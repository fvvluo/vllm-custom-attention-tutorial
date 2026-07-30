# SPDX-License-Identifier: Apache-2.0
"""
分页注意力 kernel：causal + GQA + 分页 KV + stride 寻址 + 原地写。

分 prefill / decode 两条路径：
  - prefill（query_len 大）：FlashAttention-2 风格，query 分块 × KV 分块，tl.dot 走 Tensor Core，online softmax。
  - decode（query_len==1）：split-KV，每 split 算 partial+LSE 再 combine。

分页寻址：logical block -> block_table -> physical block。
"""
import torch
import triton
import triton.language as tl


# ====== PREFILL (FlashAttention-2 + GQA 打包) ======
@triton.jit
def _prefill_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    query_start_loc_ptr, seq_lens_ptr, block_table_ptr,
    scale,
    num_heads: tl.constexpr, num_kv_heads: tl.constexpr, GROUP: tl.constexpr,
    HEAD_SIZE: tl.constexpr, KV_BLOCK: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    max_num_blocks: tl.constexpr,
    q_stride_t, q_stride_h,
    kc_stride_b, kc_stride_h, kc_stride_s,
    vc_stride_b, vc_stride_h, vc_stride_s,
    o_stride_t, o_stride_h, block_table_stride,
    num_seqs,
):
    # grid = (max_query_blocks, num_kv_heads, num_seqs)
    # M = BLOCK_Q * GROUP：BLOCK_Q 个 query token × GROUP 个 q_head 打包进 M 维，共享同一 kv_head 的 K/V。
    q_block = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    q_start = tl.load(query_start_loc_ptr + seq_idx)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1)
    query_len = q_end - q_start
    if q_block * BLOCK_Q >= query_len:
        return
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - query_len

    offs_m = tl.arange(0, BLOCK_M)
    query_local = offs_m // GROUP                     # 该行属于哪个 query token
    h_local = offs_m % GROUP                          # 该行属于哪个 q_head
    q_pos_rel = q_block * BLOCK_Q + query_local        # query 在本请求内的相对位置
    head_ids = kv_head_idx * GROUP + h_local
    offs_d = tl.arange(0, HEAD_SIZE)

    q_valid = q_pos_rel < query_len
    q_rows = q_start + q_pos_rel
    q = tl.load(q_ptr + q_rows[:, None] * q_stride_t + head_ids[:, None] * q_stride_h + offs_d[None, :],
                mask=q_valid[:, None], other=0.0)      # [BLOCK_M, HS]
    abs_pos = context_len + q_pos_rel                  # [BLOCK_M] causal 上界（每行按其 query）

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    # causal 剪枝：只遍历到本块最大 query 的绝对位置
    max_qpos = q_block * BLOCK_Q + BLOCK_Q
    max_kv = context_len + tl.minimum(max_qpos, query_len)
    n_kv_blocks = (max_kv + BLOCK_N - 1) // BLOCK_N
    for nb in range(0, n_kv_blocks):
        kv_pos = nb * BLOCK_N + offs_n                 # [BLOCK_N]
        kv_valid = kv_pos < seq_len
        logical_blk = kv_pos // KV_BLOCK
        slot = kv_pos % KV_BLOCK
        pb = tl.load(block_table_ptr + seq_idx * block_table_stride + logical_blk,
                     mask=(logical_blk < max_num_blocks) & kv_valid, other=0)
        k_off = pb[:, None] * kc_stride_b + kv_head_idx * kc_stride_h + slot[:, None] * kc_stride_s + offs_d[None, :]
        v_off = pb[:, None] * vc_stride_b + kv_head_idx * vc_stride_h + slot[:, None] * vc_stride_s + offs_d[None, :]
        k = tl.load(k_cache_ptr + k_off, mask=kv_valid[:, None], other=0.0)   # [BLOCK_N, HS]
        v = tl.load(v_cache_ptr + v_off, mask=kv_valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale            # [BLOCK_M, BLOCK_N]
        causal = abs_pos[:, None] >= kv_pos[None, :]
        valid = causal & kv_valid[None, :]
        qk = tl.where(valid, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)          # online softmax 累加
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(out_ptr + q_rows[:, None] * o_stride_t + head_ids[:, None] * o_stride_h + offs_d[None, :],
             out.to(out_ptr.dtype.element_ty), mask=q_valid[:, None])


# ====== SPARSE PREFILL (StreamingLLM / A-shape：sink 前缀 + local 窗口) ======
@triton.jit
def _prefill_sparse_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    query_start_loc_ptr, seq_lens_ptr, block_table_ptr,
    scale,
    num_heads: tl.constexpr, num_kv_heads: tl.constexpr, GROUP: tl.constexpr,
    HEAD_SIZE: tl.constexpr, KV_BLOCK: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SINK: tl.constexpr, LOCAL: tl.constexpr,
    max_num_blocks: tl.constexpr,
    q_stride_t, q_stride_h,
    kc_stride_b, kc_stride_h, kc_stride_s,
    vc_stride_b, vc_stride_h, vc_stride_s,
    o_stride_t, o_stride_h, block_table_stride,
    num_seqs,
):
    # 每个 query 只 attend：sink 前缀 [0, SINK) ∪ 局部窗口 [p-LOCAL, p]。SINK/LOCAL 为 BLOCK_N 的倍数。
    q_block = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    q_start = tl.load(query_start_loc_ptr + seq_idx)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1)
    query_len = q_end - q_start
    if q_block * BLOCK_Q >= query_len:
        return
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - query_len

    offs_m = tl.arange(0, BLOCK_M)
    query_local = offs_m // GROUP
    h_local = offs_m % GROUP
    q_pos_rel = q_block * BLOCK_Q + query_local
    head_ids = kv_head_idx * GROUP + h_local
    offs_d = tl.arange(0, HEAD_SIZE)

    q_valid = q_pos_rel < query_len
    q_rows = q_start + q_pos_rel
    q = tl.load(q_ptr + q_rows[:, None] * q_stride_t + head_ids[:, None] * q_stride_h + offs_d[None, :],
                mask=q_valid[:, None], other=0.0)
    abs_pos = context_len + q_pos_rel

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)

    # 本块 query 的绝对位置范围
    min_abs = context_len + q_block * BLOCK_Q
    max_abs = context_len + tl.minimum(q_block * BLOCK_Q + BLOCK_Q, query_len)

    # 局部窗口起始 KV 块（对齐到 BLOCK_N），覆盖 [min_abs-LOCAL, max_abs)
    local_start = min_abs - LOCAL
    local_start = tl.maximum(local_start, 0)
    local_start_blk = local_start // BLOCK_N
    n_sink_blk = SINK // BLOCK_N                         # sink 覆盖前 SINK 个位置
    max_kv_blk = (max_abs + BLOCK_N - 1) // BLOCK_N

    # ---- 阶段 1：sink 前缀块 [0, n_sink_blk) ----
    for nb in range(0, n_sink_blk):
        kv_pos = nb * BLOCK_N + offs_n
        kv_valid = kv_pos < seq_len
        logical_blk = kv_pos // KV_BLOCK
        slot = kv_pos % KV_BLOCK
        pb = tl.load(block_table_ptr + seq_idx * block_table_stride + logical_blk,
                     mask=(logical_blk < max_num_blocks) & kv_valid, other=0)
        k_off = pb[:, None] * kc_stride_b + kv_head_idx * kc_stride_h + slot[:, None] * kc_stride_s + offs_d[None, :]
        v_off = pb[:, None] * vc_stride_b + kv_head_idx * vc_stride_h + slot[:, None] * vc_stride_s + offs_d[None, :]
        k = tl.load(k_cache_ptr + k_off, mask=kv_valid[:, None], other=0.0)
        v = tl.load(v_cache_ptr + v_off, mask=kv_valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale
        causal = abs_pos[:, None] >= kv_pos[None, :]
        valid = causal & kv_valid[None, :]
        qk = tl.where(valid, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # ---- 阶段 2：局部窗口块 [local_start_blk, max_kv_blk)，但跳过已算的 sink 块 ----
    for nb in range(local_start_blk, max_kv_blk):
        if nb >= n_sink_blk:                            # 避免与 sink 段重复计算
            kv_pos = nb * BLOCK_N + offs_n
            kv_valid = kv_pos < seq_len
            logical_blk = kv_pos // KV_BLOCK
            slot = kv_pos % KV_BLOCK
            pb = tl.load(block_table_ptr + seq_idx * block_table_stride + logical_blk,
                         mask=(logical_blk < max_num_blocks) & kv_valid, other=0)
            k_off = pb[:, None] * kc_stride_b + kv_head_idx * kc_stride_h + slot[:, None] * kc_stride_s + offs_d[None, :]
            v_off = pb[:, None] * vc_stride_b + kv_head_idx * vc_stride_h + slot[:, None] * vc_stride_s + offs_d[None, :]
            k = tl.load(k_cache_ptr + k_off, mask=kv_valid[:, None], other=0.0)
            v = tl.load(v_cache_ptr + v_off, mask=kv_valid[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * scale
            causal = abs_pos[:, None] >= kv_pos[None, :]
            # 局部窗口下界：kv_pos >= abs_pos - LOCAL
            in_window = kv_pos[None, :] >= (abs_pos[:, None] - LOCAL)
            valid = causal & kv_valid[None, :] & in_window
            qk = tl.where(valid, qk, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(out_ptr + q_rows[:, None] * o_stride_t + head_ids[:, None] * o_stride_h + offs_d[None, :],
             out.to(out_ptr.dtype.element_ty), mask=q_valid[:, None])



# ============================ DECODE (split-KV) ============================
@triton.jit
def _decode_split_kernel(
    part_o_ptr, part_lse_ptr,
    q_ptr, k_cache_ptr, v_cache_ptr,
    query_start_loc_ptr, seq_lens_ptr, token_seq_idx_ptr, block_table_ptr,
    scale,
    num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
    HEAD_SIZE: tl.constexpr, KV_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr, max_num_blocks: tl.constexpr, NUM_SPLITS: tl.constexpr,
    q_stride_t, q_stride_h,
    kc_stride_b, kc_stride_h, kc_stride_s,
    vc_stride_b, vc_stride_h, vc_stride_s,
    block_table_stride,
    po_stride_t, po_stride_h, po_stride_s, pl_stride_t, pl_stride_h,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    seq_idx = tl.load(token_seq_idx_ptr + token_idx)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    kv_upper = seq_len                                  # decode: query 是最后一个，能看全部
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    per = tl.cdiv(kv_upper, NUM_SPLITS)
    split_start = split_idx * per
    split_end = tl.minimum(split_start + per, kv_upper)

    offs_d = tl.arange(0, HEAD_SIZE)
    q = tl.load(q_ptr + token_idx * q_stride_t + head_idx * q_stride_h + offs_d).to(tl.float32) * scale

    m_i = -float("inf"); l_i = 0.0
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    nb_start = split_start // BLOCK_N
    nb_end = tl.cdiv(split_end, BLOCK_N)
    for nb in range(nb_start, nb_end):
        kv_pos = nb * BLOCK_N + offs_n
        valid = (kv_pos >= split_start) & (kv_pos < split_end)
        logical_blk = kv_pos // KV_BLOCK
        slot = kv_pos % KV_BLOCK
        pb = tl.load(block_table_ptr + seq_idx * block_table_stride + logical_blk,
                     mask=(logical_blk < max_num_blocks) & valid, other=0)
        k_off = pb[:, None] * kc_stride_b + kv_head_idx * kc_stride_h + slot[:, None] * kc_stride_s + offs_d[None, :]
        v_off = pb[:, None] * vc_stride_b + kv_head_idx * vc_stride_h + slot[:, None] * vc_stride_s + offs_d[None, :]
        k = tl.load(k_cache_ptr + k_off, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + v_off, mask=valid[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1)         # [BLOCK_N]
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    safe = l_i > 0.0
    o_s = tl.where(safe, acc / tl.where(safe, l_i, 1.0), 0.0)
    lse_s = tl.where(safe, m_i + tl.log(tl.where(safe, l_i, 1.0)), -float("inf"))
    base = token_idx * po_stride_t + head_idx * po_stride_h + split_idx * po_stride_s
    tl.store(part_o_ptr + base + offs_d, o_s)
    tl.store(part_lse_ptr + token_idx * pl_stride_t + head_idx * pl_stride_h + split_idx, lse_s)


@triton.jit
def _decode_gqa_kernel(
    part_o_ptr, part_lse_ptr,
    q_ptr, k_cache_ptr, v_cache_ptr,
    seq_lens_ptr, token_seq_idx_ptr, block_table_ptr,
    scale,
    num_heads: tl.constexpr, num_kv_heads: tl.constexpr, GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HEAD_SIZE: tl.constexpr, KV_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr, max_num_blocks: tl.constexpr, NUM_SPLITS: tl.constexpr,
    DECODE_WINDOW: tl.constexpr,
    q_stride_t, q_stride_h,
    kc_stride_b, kc_stride_h, kc_stride_s,
    vc_stride_b, vc_stride_h, vc_stride_s,
    block_table_stride,
    po_stride_t, po_stride_h, po_stride_s, pl_stride_t, pl_stride_h,
):
    # grid = (num_tokens, num_kv_heads, NUM_SPLITS)
    # 一个 program 处理一个 kv_head 的全部 GROUP 个 q_head：K/V 只读一次。
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)
    seq_idx = tl.load(token_seq_idx_ptr + token_idx)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    kv_upper = seq_len

    # decode 窗口稀疏：只 attend 最近 DECODE_WINDOW 个 key，split 在 [win_start, kv_upper) 内划分。
    win_lo = 0
    if DECODE_WINDOW > 0:
        cand = kv_upper - DECODE_WINDOW
        win_lo = tl.where(cand > 0, cand, 0)
    span = kv_upper - win_lo
    per = tl.cdiv(span, NUM_SPLITS)
    split_start = win_lo + split_idx * per
    split_end = tl.minimum(split_start + per, kv_upper)

    offs_d = tl.arange(0, HEAD_SIZE)
    offs_g = tl.arange(0, GROUP_PAD)                    # pad 到 2 的幂给 tl.dot
    g_valid = offs_g < GROUP
    head_ids = kv_head_idx * GROUP + offs_g
    q = tl.load(q_ptr + token_idx * q_stride_t + head_ids[:, None] * q_stride_h + offs_d[None, :],
                mask=g_valid[:, None], other=0.0)   # [GROUP_PAD, HEAD_SIZE]

    m_i = tl.zeros([GROUP_PAD], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_SIZE], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    nb_start = split_start // BLOCK_N
    nb_end = tl.cdiv(split_end, BLOCK_N)
    for nb in range(nb_start, nb_end):
        kv_pos = nb * BLOCK_N + offs_n
        valid = (kv_pos >= split_start) & (kv_pos < split_end)
        logical_blk = kv_pos // KV_BLOCK
        slot = kv_pos % KV_BLOCK
        pb = tl.load(block_table_ptr + seq_idx * block_table_stride + logical_blk,
                     mask=(logical_blk < max_num_blocks) & valid, other=0)
        k_off = pb[:, None] * kc_stride_b + kv_head_idx * kc_stride_h + slot[:, None] * kc_stride_s + offs_d[None, :]
        v_off = pb[:, None] * vc_stride_b + kv_head_idx * vc_stride_h + slot[:, None] * vc_stride_s + offs_d[None, :]
        k = tl.load(k_cache_ptr + k_off, mask=valid[:, None], other=0.0)   # [BLOCK_N, HS]
        v = tl.load(v_cache_ptr + v_off, mask=valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale             # [GROUP_PAD, BLOCK_N]
        qk = tl.where(valid[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])                 # [GROUP_PAD, BLOCK_N]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    safe = l_i > 0.0
    o_s = tl.where(safe[:, None], acc / tl.where(safe, l_i, 1.0)[:, None], 0.0)
    lse_s = tl.where(safe, m_i + tl.log(tl.where(safe, l_i, 1.0)), -float("inf"))
    # 一次性写 GROUP 个 head 的 part_o / part_lse（mask 掉 pad 行）
    o_base = (token_idx * po_stride_t + head_ids[:, None] * po_stride_h
              + split_idx * po_stride_s + offs_d[None, :])
    tl.store(part_o_ptr + o_base, o_s, mask=g_valid[:, None])
    l_base = token_idx * pl_stride_t + head_ids * pl_stride_h + split_idx
    tl.store(part_lse_ptr + l_base, lse_s, mask=g_valid)


@triton.jit
def _combine_kernel(
    out_ptr, part_o_ptr, part_lse_ptr,
    HEAD_SIZE: tl.constexpr, NUM_SPLITS: tl.constexpr,
    o_stride_t, o_stride_h, po_stride_t, po_stride_h, po_stride_s, pl_stride_t, pl_stride_h,
):
    token_idx = tl.program_id(0); head_idx = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_SIZE); s_offs = tl.arange(0, NUM_SPLITS)
    lse = tl.load(part_lse_ptr + token_idx * pl_stride_t + head_idx * pl_stride_h + s_offs)
    m = tl.max(lse, axis=0)
    w = tl.exp(lse - m); denom = tl.sum(w, axis=0)
    base = token_idx * po_stride_t + head_idx * po_stride_h
    o_all = tl.load(part_o_ptr + base + s_offs[:, None] * po_stride_s + offs_d[None, :])
    acc = tl.sum(w[:, None] * o_all, axis=0)
    out = acc / denom
    tl.store(out_ptr + token_idx * o_stride_t + head_idx * o_stride_h + offs_d,
             out.to(out_ptr.dtype.element_ty))


import os
NUM_SPLITS = int(os.environ.get("DEC_SPLITS", 32))  # decode split-KV 段数
BLOCK_M = 64
BLOCK_N = 32
BLOCK_N_DECODE = int(os.environ.get("DEC_BLOCK_N", BLOCK_N))  # decode KV 分块
BLOCK_Q = 16          # prefill：每 program 处理 BLOCK_Q 个 query token
BLOCK_Q = int(os.environ.get("PF_BLOCK_Q", BLOCK_Q))
BLOCK_N_PREFILL = int(os.environ.get("PF_BLOCK_N", 32))
PF_WARPS = int(os.environ.get("PF_WARPS", 8))
PF_STAGES = int(os.environ.get("PF_STAGES", 2))
DEC_WARPS = int(os.environ.get("DEC_WARPS", 4))     # decode kernel warp 数
# decode 窗口稀疏：只 attend 最近 N 个 key，0=关闭(dense)。
DECODE_WINDOW = int(os.environ.get("DECODE_WINDOW", 0))


def paged_attention_triton(query, key_cache, value_cache, output,
                           query_start_loc, seq_lens, token_seq_idx, block_table, scale):
    num_tokens, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    kv_block = key_cache.shape[2]
    max_num_blocks = block_table.shape[1]
    num_seqs = seq_lens.shape[0]

    # 判断 prefill vs decode：用张量 shape（CPU 已知，无 GPU 同步，兼容 CUDA graph）。
    # decode: num_tokens == num_seqs；prefill/mixed: num_tokens > num_seqs。
    is_decode_only = (num_tokens == num_seqs)

    if not is_decode_only:
        # ---- prefill / mixed 路径：GQA 打包的 FlashAttention-2 ----
        group = num_heads // num_kv_heads
        BLOCK_M_prefill = BLOCK_Q * group
        # grid dim0 用 num_tokens 作上界（kernel 内 q_block*BLOCK_Q>=query_len 会提前 return）
        max_qblocks = (num_tokens + BLOCK_Q - 1) // BLOCK_Q
        grid = (max_qblocks, num_kv_heads, num_seqs)
        _prefill_kernel[grid](
            output, query, key_cache, value_cache,
            query_start_loc, seq_lens, block_table, scale,
            num_heads=num_heads, num_kv_heads=num_kv_heads, GROUP=group,
            HEAD_SIZE=head_size, KV_BLOCK=kv_block,
            BLOCK_Q=BLOCK_Q, BLOCK_M=BLOCK_M_prefill, BLOCK_N=BLOCK_N_PREFILL, max_num_blocks=max_num_blocks,
            q_stride_t=query.stride(0), q_stride_h=query.stride(1),
            kc_stride_b=key_cache.stride(0), kc_stride_h=key_cache.stride(1), kc_stride_s=key_cache.stride(2),
            vc_stride_b=value_cache.stride(0), vc_stride_h=value_cache.stride(1), vc_stride_s=value_cache.stride(2),
            o_stride_t=output.stride(0), o_stride_h=output.stride(1), block_table_stride=block_table.stride(0),
            num_seqs=num_seqs,
            num_warps=PF_WARPS, num_stages=PF_STAGES,
        )
    else:
        # ---- 纯 decode 路径：split-KV ----
        part_o = torch.empty(num_tokens, num_heads, NUM_SPLITS, head_size,
                             device=query.device, dtype=torch.float32)
        part_lse = torch.empty(num_tokens, num_heads, NUM_SPLITS,
                               device=query.device, dtype=torch.float32)
        group = num_heads // num_kv_heads
        _decode_gqa_kernel[(num_tokens, num_kv_heads, NUM_SPLITS)](
            part_o, part_lse, query, key_cache, value_cache,
            seq_lens, token_seq_idx, block_table, scale,
            num_heads=num_heads, num_kv_heads=num_kv_heads, GROUP=group, GROUP_PAD=16,
            HEAD_SIZE=head_size, KV_BLOCK=kv_block,
            BLOCK_N=BLOCK_N_DECODE, max_num_blocks=max_num_blocks, NUM_SPLITS=NUM_SPLITS,
            DECODE_WINDOW=DECODE_WINDOW,
            q_stride_t=query.stride(0), q_stride_h=query.stride(1),
            kc_stride_b=key_cache.stride(0), kc_stride_h=key_cache.stride(1), kc_stride_s=key_cache.stride(2),
            vc_stride_b=value_cache.stride(0), vc_stride_h=value_cache.stride(1), vc_stride_s=value_cache.stride(2),
            block_table_stride=block_table.stride(0),
            po_stride_t=part_o.stride(0), po_stride_h=part_o.stride(1), po_stride_s=part_o.stride(2),
            pl_stride_t=part_lse.stride(0), pl_stride_h=part_lse.stride(1),
            num_warps=DEC_WARPS,
        )
        _combine_kernel[(num_tokens, num_heads)](
            output, part_o, part_lse,
            HEAD_SIZE=head_size, NUM_SPLITS=NUM_SPLITS,
            o_stride_t=output.stride(0), o_stride_h=output.stride(1),
            po_stride_t=part_o.stride(0), po_stride_h=part_o.stride(1), po_stride_s=part_o.stride(2),
            pl_stride_t=part_lse.stride(0), pl_stride_h=part_lse.stride(1),
        )
    return output


# 稀疏 prefill 窗口参数：SINK=前缀全局块，LOCAL=局部窗口，均为 BLOCK_N 倍数。
SPARSE_SINK = int(os.environ.get("SP_SINK", 4096))
SPARSE_LOCAL = int(os.environ.get("SP_LOCAL", 8192))


def paged_sparse_prefill(query, key_cache, value_cache, output,
                         query_start_loc, seq_lens, block_table, scale,
                         sink=None, local=None):
    """稀疏 prefill：每 query 只 attend sink 前缀 + local 窗口。降 O(N^2)→O(N*(sink+local))。"""
    num_tokens, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    kv_block = key_cache.shape[2]
    max_num_blocks = block_table.shape[1]
    num_seqs = seq_lens.shape[0]
    sink = SPARSE_SINK if sink is None else sink
    local = SPARSE_LOCAL if local is None else local
    group = num_heads // num_kv_heads
    BLOCK_M_prefill = BLOCK_Q * group
    max_qblocks = (num_tokens + BLOCK_Q - 1) // BLOCK_Q
    grid = (max_qblocks, num_kv_heads, num_seqs)
    _prefill_sparse_kernel[grid](
        output, query, key_cache, value_cache,
        query_start_loc, seq_lens, block_table, scale,
        num_heads=num_heads, num_kv_heads=num_kv_heads, GROUP=group,
        HEAD_SIZE=head_size, KV_BLOCK=kv_block,
        BLOCK_Q=BLOCK_Q, BLOCK_M=BLOCK_M_prefill, BLOCK_N=BLOCK_N_PREFILL,
        SINK=sink, LOCAL=local, max_num_blocks=max_num_blocks,
        q_stride_t=query.stride(0), q_stride_h=query.stride(1),
        kc_stride_b=key_cache.stride(0), kc_stride_h=key_cache.stride(1), kc_stride_s=key_cache.stride(2),
        vc_stride_b=value_cache.stride(0), vc_stride_h=value_cache.stride(1), vc_stride_s=value_cache.stride(2),
        o_stride_t=output.stride(0), o_stride_h=output.stride(1), block_table_stride=block_table.stride(0),
        num_seqs=num_seqs, num_warps=PF_WARPS, num_stages=PF_STAGES,
    )
    return output
