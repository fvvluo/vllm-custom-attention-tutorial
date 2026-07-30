# SPDX-License-Identifier: Apache-2.0
"""
分页版投机验证 attention（verify）——把 spec_decode/verify_attention 桥接到 vLLM 分页布局。
================================================================================

verify 场景：一次前向验证 K 个候选 token。每个候选 attend：
  - **history**（cache_len 个已确认 token）：对所有候选可见；存在 vLLM 分页池
    `[num_blocks, kv_heads, PAGE, d]` + block_table。
  - **K 个候选自身的 KV**：候选间按 chain-causal（j<=t）或 tree_mask 可见；候选 KV 是本步新算的、
    **尚未写入分页 cache**，作为独立连续张量 `cand_k/cand_v (1, kv_heads, K, d)` 传入。

与 sparse_paged 一致：history 按 SPARSE_BLOCK=64=4×PAGE 的块粒度读，选中块（dense verify 时全选、
sparse verify 时 Quest 选 top-k）展开成物理页 gather。候选段单独一个 split 处理。
q pack 成 pack_rows = group*K 行（K 候选 × group 个 q 头）。

产出 paged 版 `verify_paged`（dense history）与 `sparse_verify_paged`（稀疏 history）。
只离线验证/桥接；接 vLLM spec-decode 调度是后续阶段。
"""
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_FLASH = Path(__file__).resolve().parents[2]
_ATT = _FLASH / "attention-test"
for _p in (str(_ATT), str(_ATT / "sparse_attn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from select_blocks import select_blocks  # noqa: E402

PAGE = 16


@triton.jit
def _verify_paged_split_kernel(
    Q, KcacheP, VcacheP, BlockTable, CandK, CandV, TreeMask,
    Selected, NumSel, OutP, MlkP,
    sm_scale,
    cache_len, K, kv_heads, num_splits, n_hist_blocks,
    pack_rows, q_len,
    PAGES_PER_SB: tl.constexpr,
    stride_qb, stride_qh, stride_qm,
    stride_sel_h, stride_sel_n,
    stride_ns_h,
    kc_b, kc_h, kc_s,
    vc_b, vc_h, vc_s,
    bt_stride,
    ck_h, ck_n,               # CandK (kv_heads, K, d) strides
    cv_h, cv_n,
    tm_r, tm_j,
    stride_oh, stride_os, stride_om,
    stride_eh, stride_es, stride_em,
    BLOCK_M: tl.constexpr,
    SPARSE_BLOCK: tl.constexpr,
    PAGE_N: tl.constexpr,
    HDIM: tl.constexpr,
    TREE: tl.constexpr,
    K_POW2: tl.constexpr,     # >=K 的 2 的幂（候选段列宽）
):
    # grid = (kv_heads, num_splits+1)：split<num_splits 处理 history 选中块的一段；
    # 最后一个 split(==num_splits) 专门处理 K 个候选段（chain/tree mask）。单序列 b=0。
    h = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HDIM)
    m_mask = offs_m < pack_rows
    t_row = offs_m % q_len                       # 候选行 t（pack 行 r=g*q_len+t）

    q_ptrs = Q + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HDIM), tl.float32)

    is_cand_split = pid_s == num_splits
    if is_cand_split:
        # ---- 候选段：读 CandK/CandV 的 K 列，chain/tree mask ----
        cols = tl.arange(0, K_POW2)              # 候选局部下标 j
        j_valid = cols < K
        k_ptrs = CandK + h * ck_h + cols[:, None] * ck_n + offs_d[None, :]
        kc = tl.load(k_ptrs, mask=j_valid[:, None], other=0.0)   # (K_POW2, HDIM)
        s = tl.dot(q, tl.trans(kc), out_dtype=tl.float32) * sm_scale
        if TREE:
            tmp = TreeMask + t_row[:, None] * tm_r + cols[None, :] * tm_j
            tmask = tl.load(tmp, mask=j_valid[None, :], other=0)
            vis = (tmask > 0) & j_valid[None, :]
        else:
            vis = (cols[None, :] <= t_row[:, None]) & j_valid[None, :]
        s = tl.where(vis, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v_ptrs = CandV + h * cv_h + cols[:, None] * cv_n + offs_d[None, :]
        vc = tl.load(v_ptrs, mask=j_valid[:, None], other=0.0)
        acc += tl.dot(p.to(vc.dtype), vc, out_dtype=tl.float32)
        m_i = m_new
    else:
        # ---- history 段：读选中块（本 split 负责一段），全对候选可见 ----
        n_sel = tl.load(NumSel + h * stride_ns_h)
        sel_per = tl.cdiv(n_sel, num_splits)
        sidx_start = pid_s * sel_per
        sidx_end = tl.minimum(sidx_start + sel_per, n_sel)
        offs_pg = tl.arange(0, PAGES_PER_SB)
        offs_sl = tl.arange(0, PAGE_N)
        sel_base = Selected + h * stride_sel_h
        for sidx in range(sidx_start, sidx_end):
            blk = tl.load(sel_base + sidx * stride_sel_n)
            n0 = blk * SPARSE_BLOCK
            logical_pg = blk * PAGES_PER_SB + offs_pg
            phys = tl.load(BlockTable + logical_pg)
            k_ptrs = (KcacheP + phys[:, None, None] * kc_b + h * kc_h
                      + offs_sl[None, :, None] * kc_s + offs_d[None, None, :])
            v_ptrs = (VcacheP + phys[:, None, None] * vc_b + h * vc_h
                      + offs_sl[None, :, None] * vc_s + offs_d[None, None, :])
            k = tl.load(k_ptrs).reshape(SPARSE_BLOCK, HDIM)
            v = tl.load(v_ptrs).reshape(SPARSE_BLOCK, HDIM)
            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
            cols = n0 + tl.arange(0, SPARSE_BLOCK)
            s = tl.where((cols < cache_len)[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
            m_i = m_new

    o_ptrs = OutP + h * stride_oh + pid_s * stride_os + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=m_mask[:, None])
    e_ptrs = MlkP + h * stride_eh + pid_s * stride_es + offs_m * stride_em
    tl.store(e_ptrs, m_i, mask=m_mask)
    tl.store(e_ptrs + stride_em * BLOCK_M, l_i, mask=m_mask)


@triton.jit
def _verify_paged_combine_kernel(
    OutP, MlkP, Out, total_splits, kv_heads,
    stride_oh, stride_os, stride_om,
    stride_eh, stride_es, stride_em,
    stride_fh, stride_fm,
    BLOCK_M: tl.constexpr, HDIM: tl.constexpr,
    MAX_SPLITS: tl.constexpr, SPLIT_CHUNK: tl.constexpr,
):
    h = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_d = tl.arange(0, HDIM)
    sp_offs = tl.arange(0, MAX_SPLITS)
    sp_mask = sp_offs < total_splits
    e_ptrs = MlkP + h * stride_eh + sp_offs * stride_es + pid_m * stride_em
    m_s = tl.load(e_ptrs, mask=sp_mask, other=float("-inf"))
    l_s = tl.load(e_ptrs + stride_em * BLOCK_M, mask=sp_mask, other=0.0)
    m_max = tl.max(m_s, 0)
    l_sum = tl.sum(tl.where(sp_mask, tl.exp(m_s - m_max), 0.0) * l_s, 0)
    inv_l = 1.0 / tl.maximum(l_sum, 1e-30)
    acc = tl.zeros((HDIM,), tl.float32)
    for s0 in range(0, total_splits, SPLIT_CHUNK):
        offs_s = s0 + tl.arange(0, SPLIT_CHUNK)
        s_mask = offs_s < total_splits
        ec = MlkP + h * stride_eh + offs_s * stride_es + pid_m * stride_em
        m_c = tl.load(ec, mask=s_mask, other=float("-inf"))
        w = tl.where(s_mask, tl.exp(m_c - m_max) * inv_l, 0.0)
        o_ptrs = OutP + h * stride_oh + offs_s[:, None] * stride_os + pid_m * stride_om + offs_d[None, :]
        o = tl.load(o_ptrs, mask=s_mask[:, None], other=0.0)
        acc += tl.sum(o * w[:, None], 0)
    f_ptrs = Out + h * stride_fh + pid_m * stride_fm + offs_d
    tl.store(f_ptrs, acc.to(Out.dtype.element_ty))


_WS: dict = {}


def _get_ws(kv_heads, ns, block_m, d, dev):
    key = (kv_heads, ns, block_m, d, str(dev))
    ws = _WS.get(key)
    if ws is None:
        if len(_WS) > 32:
            _WS.clear()
        op = torch.empty((kv_heads, ns, block_m, d), dtype=torch.bfloat16, device=dev)
        mp = torch.empty((kv_heads, ns, 2 * block_m), dtype=torch.float32, device=dev)
        ws = (op, mp)
        _WS[key] = ws
    return ws


@torch.no_grad()
def verify_paged(q, key_cache, value_cache, block_table, cand_k, cand_v, cache_len,
                 sm_scale=None, tree_mask=None, sparse_block=64,
                 selected=None, num_sel=None, num_splits=None):
    """分页投机验证 attention。

    Args:
        q: (1, q_heads, K, d)  —— K 候选 query（BHSD, b=1）。
        key_cache/value_cache: [num_blocks, kv_heads, PAGE, d] 分页 history 池。
        block_table: (max_blocks,) int32，单序列。
        cand_k/cand_v: (kv_heads, K, d) —— K 候选自身 KV（连续，未入池）。
        cache_len: history 长度。
        selected/num_sel: 稀疏 history 选块（None 时 dense 全选 history 块）。
    Returns: (1, q_heads, K, d)。
    """
    _, q_heads, K, d = q.shape
    kv_heads = key_cache.shape[1]
    group = q_heads // kv_heads
    pages_per_sb = sparse_block // PAGE
    pack_rows = group * K
    block_m = max(16, triton.next_power_of_2(pack_rows))
    if sm_scale is None:
        sm_scale = d ** -0.5
    dev = q.device
    n_hist_blocks = (cache_len + sparse_block - 1) // sparse_block

    if selected is None:
        # dense：history 全选 0..n_hist_blocks-1
        selected = torch.arange(n_hist_blocks, dtype=torch.int32, device=dev).view(1, n_hist_blocks)
        selected = selected.expand(kv_heads, n_hist_blocks).contiguous()
        num_sel = torch.full((kv_heads,), n_hist_blocks, dtype=torch.int32, device=dev)

    if num_splits is None:
        try:
            sm = torch.cuda.get_device_properties(dev).multi_processor_count
        except Exception:  # noqa: BLE001
            sm = 78
        nsel = int(num_sel.max().item())
        num_splits = max(1, min(nsel, (12 * sm) // kv_heads))
    total_splits = num_splits + 1   # +1 给候选段

    sqb, sqh, sqm = q.stride(0), group * q.stride(1), q.stride(2)
    op, mp = _get_ws(kv_heads, total_splits, block_m, d, dev)
    out = torch.empty((1, kv_heads, pack_rows, d), dtype=q.dtype, device=dev)
    TREE = tree_mask is not None
    tm = tree_mask if TREE else torch.empty((1, 1), dtype=torch.int8, device=dev)
    K_pow2 = max(16, triton.next_power_of_2(K))   # tl.dot 收缩维需 >=16

    with torch.cuda.device(dev):
        _verify_paged_split_kernel[(kv_heads, total_splits)](
            q, key_cache, value_cache, block_table, cand_k, cand_v, tm,
            selected, num_sel, op, mp,
            sm_scale, cache_len, K, kv_heads, num_splits, n_hist_blocks,
            pack_rows, K,
            pages_per_sb,
            sqb, sqh, sqm,
            selected.stride(0), selected.stride(1),
            num_sel.stride(0),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0),
            cand_k.stride(0), cand_k.stride(1),
            cand_v.stride(0), cand_v.stride(1),
            tm.stride(0), tm.stride(1),
            op.stride(0), op.stride(1), op.stride(2),
            mp.stride(0), mp.stride(1), mp.stride(2),
            BLOCK_M=block_m, SPARSE_BLOCK=sparse_block, PAGE_N=PAGE, HDIM=d,
            TREE=TREE, K_POW2=K_pow2, num_warps=4, num_stages=2,
        )
        _verify_paged_combine_kernel[(kv_heads, pack_rows)](
            op, mp, out, total_splits, kv_heads,
            op.stride(0), op.stride(1), op.stride(2),
            mp.stride(0), mp.stride(1), mp.stride(2),
            out.stride(1), out.stride(2),   # 输出 (1,kvh,pack_rows,d) 的 head/row 步长
            BLOCK_M=block_m, HDIM=d, MAX_SPLITS=triton.next_power_of_2(total_splits),
            SPLIT_CHUNK=min(32, triton.next_power_of_2(total_splits)), num_warps=4,
        )
    # out (1, kv_heads, pack_rows=group*K, d) -> (1, q_heads, K, d)
    return out.view(1, kv_heads, group, K, d).reshape(1, q_heads, K, d)


@torch.no_grad()
def sparse_verify_paged(q, key_cache, value_cache, block_table, cand_k, cand_v, cache_len,
                        k_min, k_max, sm_scale=None, tree_mask=None, sparse_block=64,
                        sparsity=0.25, n_sink=1, recent_window_blocks=8, num_splits=None):
    """稀疏 history 选块 + 分页 verify（sparse×verify 的 paged 版）。

    k_min/k_max: (1, kv_heads, n_hist_blocks, d) history 块摘要（sparse_block 粒度）。
    """
    _, q_heads, K, d = q.shape
    kv_heads = key_cache.shape[1]
    n_hist_blocks = (cache_len + sparse_block - 1) // sparse_block
    budget = max(n_sink + recent_window_blocks + 1, int(round(n_hist_blocks * sparsity)))
    budget = min(budget, n_hist_blocks)
    # 用候选里第一个位置的 q 打分（或对 K 取 max）；这里用 q 的 group×K 展平后按 kv_head 选。
    # select_blocks 期望 q:(b,q_heads,1,d)——用最后一个候选(最接近未来)的 query 做选块代表。
    q_sel = q[:, :, -1:, :].contiguous()   # (1, q_heads, 1, d)
    selected, num_sel = select_blocks(q_sel, k_min, k_max, n_hist_blocks, budget,
                                      n_sink=n_sink, recent_window_blocks=recent_window_blocks)
    selected = selected[0]     # (kv_heads, budget)
    num_sel = num_sel[0]       # (kv_heads,)
    return verify_paged(q, key_cache, value_cache, block_table, cand_k, cand_v, cache_len,
                        sm_scale=sm_scale, tree_mask=tree_mask, sparse_block=sparse_block,
                        selected=selected, num_sel=num_sel, num_splits=num_splits)
