# SPDX-License-Identifier: Apache-2.0
"""
自定义分页注意力（paged attention）实现 —— 高性能版
====================================================

接口约定（保持不变，见 README Part 3.1）：
  - query:  [num_tokens, num_heads, head_size]
  - key_cache / value_cache: [num_blocks, num_kv_heads, block_size, head_size]
    （按 stride 寻址，不假设物理布局；调用方传入的可能是非连续视图）
  - output: [num_tokens, num_heads, head_size]（原地写入并返回）
  - query_start_loc: [num_seqs + 1] 每条请求 query 的起始偏移
  - seq_lens: [num_seqs] 每条请求总长度（context + query）
  - token_seq_idx: [num_tokens] 每个 token 属于哪条请求（本实现未用到）
  - block_table: [num_seqs, max_num_blocks] 逻辑块 -> 物理块

实现：把 ~/attention-test/ops/chh_flash_attention.py 的两个 flash-attention
内核移植到**分页 KV cache** 寻址上：

  1. `_paged_prefill_kernel` —— prefill 内核（任意 n_q）。
     q/o 用 TMA descriptor（每条请求内部的 q 行是连续的）；K/V 按
     block_table gather 出物理块号后按 stride 计算行地址做向量 load。
     两段式 KV 循环：完全可见段无掩码，对角线/尾部段加 causal + 边界掩码；
     fp32 online softmax（exp2 域）。
  2. `_paged_decode_fused_kernel` —— decode 内核（n_q == 1 且长 context）。
     FlashDecoding split-K：grid = (splits, num_kv_heads)，每个 program 负责
     一个 kv head 的一段 KV，同组全部 g 个 q head 共享同一份 K/V（GQA 流量
     只读一次）；partial (o, m, l) 写全局暂存，组内最后一个 program 用
     log-sum-exp 合并并复位信号量（单次 kernel 启动完成 partial + merge）。

与 chh 原版的差异（除分页寻址外）：
  - autotune 不再以 n_q/n_kv/chunk 为 key：服务时 n_kv 每步递增，会导致
    每个 decode step 重新调参。改为按 ('D') / ('D','M_PAD') 调一次。
  - decode 的 splits 用确定性启发式（目标 program 数 ≈ 2x SM 数），
    不做逐形状的 graph 计时调参。
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# device-side TMA descriptor 需要的全局显存暂存区分配器（与 chh 相同）
# ============================================================================
_TMA_SCRATCH = [None]


def _triton_alloc(size: int, alignment: int, stream):
    buf = _TMA_SCRATCH[0]
    if buf is None or buf.numel() < size:
        buf = torch.empty(size, dtype=torch.uint8, device='cuda')
        _TMA_SCRATCH[0] = buf
    return buf


triton.set_allocator(_triton_alloc)


def _prefill_configs():
    """prefill kernel 的 autotune 候选。

    只保留 H20 上实测最优的几个配置（micro-benchmark：n_q=8192、
    n_kv 8k~95k 全范围 (128,32,8,2) 稳定最优 ~138 TFLOPS，几个候选
    彼此差距 <4%），避免 autotune 在小形状首次调用上误选差配置
    （如 (128,128,4,2) 仅 ~96 TFLOPS）。
    """
    return [
        triton.Config({'BR': 128, 'BC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BR': 128, 'BC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BR': 128, 'BC': 32}, num_warps=8, num_stages=3),
    ]


def _decode_configs():
    """decode fused kernel 的 autotune 候选。

    只保留 H20 上实测最优的几个（n_kv~95k、sp=38 扫参：
    (64,4,3)=2.51 TB/s、(128,8,2)=2.44、(64,4,2)=2.43，彼此 <3%）。
    """
    return [
        triton.Config({'BC': 64}, num_warps=4, num_stages=3),
        triton.Config({'BC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BC': 128}, num_warps=8, num_stages=2),
    ]


# ============================================================================
# 1. prefill 内核（分页 KV 版）
# ============================================================================
@triton.jit
def _paged_fa_inner(o, l, m, q, k_base, v_base, bt_ptr,
                    n_kv, rows, offset, qk_scale, j_lo, j_hi,
                    kc_stride_b, kc_stride_s, vc_stride_b, vc_stride_s,
                    D: tl.constexpr, BC: tl.constexpr, BS: tl.constexpr,
                    MASKED: tl.constexpr, CAUSAL: tl.constexpr):
    """遍历 kv block [j_lo, j_hi)，online-softmax 累积（分页寻址）。

    MASKED=False 的段对所有 query 行完全可见且完全在界内，不加任何掩码；
    块表 gather 出来的物理块号 pb 一定有效（cols < n_kv 保证）。
    """
    d_offs = tl.arange(0, D)
    for j in range(j_lo, j_hi):
        cols = j * BC + tl.arange(0, BC)
        lb = cols // BS
        slot = cols % BS
        if MASKED:
            cvalid = cols < n_kv
            pb = tl.load(bt_ptr + lb, mask=cvalid, other=0).to(tl.int64)
            k = tl.load(k_base + pb[:, None] * kc_stride_b
                        + slot[:, None] * kc_stride_s + d_offs[None, :],
                        mask=cvalid[:, None], other=0.0)
        else:
            pb = tl.load(bt_ptr + lb).to(tl.int64)
            k = tl.load(k_base + pb[:, None] * kc_stride_b
                        + slot[:, None] * kc_stride_s + d_offs[None, :])
        s = tl.dot(q, tl.trans(k)) * qk_scale  # fp32
        if MASKED:
            mask = (cols < n_kv)[None, :]
            if CAUSAL:
                mask = mask & (cols[None, :] <= (rows + offset)[:, None])
            s = tl.where(mask, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.math.exp2(m - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        if MASKED:
            v = tl.load(v_base + pb[:, None] * vc_stride_b
                        + slot[:, None] * vc_stride_s + d_offs[None, :],
                        mask=(cols < n_kv)[:, None], other=0.0)
        else:
            v = tl.load(v_base + pb[:, None] * vc_stride_b
                        + slot[:, None] * vc_stride_s + d_offs[None, :])
        o = o * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    return o, l, m


@triton.autotune(configs=_prefill_configs(), key=['D'])
@triton.jit
def _paged_prefill_kernel(q_ptr, k_ptr, v_ptr, o_ptr, bt_ptr,
                          sqh, sqm, soh, som,
                          kc_stride_b, kc_stride_h, kc_stride_s,
                          vc_stride_b, vc_stride_h, vc_stride_s,
                          n_q, n_kv, h_q, g, qk_scale,
                          D: tl.constexpr, BS: tl.constexpr,
                          BR: tl.constexpr, BC: tl.constexpr,
                          CAUSAL: tl.constexpr):
    # grid = (cdiv(n_q, BR), q_heads)；batch=1（每次调用处理一条请求）
    i = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // g

    q_desc = tl.make_tensor_descriptor(q_ptr + h * sqh, shape=[n_q, D],
                                       strides=[sqm, 1], block_shape=[BR, D])
    o_desc = tl.make_tensor_descriptor(o_ptr + h * soh, shape=[n_q, D],
                                       strides=[som, 1], block_shape=[BR, D])
    k_base = k_ptr + hk * kc_stride_h
    v_base = v_ptr + hk * vc_stride_h

    q = q_desc.load([i * BR, 0])

    o = tl.zeros([BR, D], dtype=tl.float32)
    l = tl.zeros([BR], dtype=tl.float32)
    m = tl.full([BR], float("-inf"), dtype=tl.float32)

    # query 位置 i 的绝对位置为 i + offset（query 对齐到 kv 末尾）
    offset = n_kv - n_q
    rows = i * BR + tl.arange(0, BR)

    if CAUSAL:
        full = tl.minimum(n_kv, i * BR + offset + 1) // BC
        limit = tl.minimum(n_kv, i * BR + BR + offset)
    else:
        full = n_kv // BC
        limit = n_kv
    hi = tl.cdiv(limit, BC)

    o, l, m = _paged_fa_inner(o, l, m, q, k_base, v_base, bt_ptr,
                              n_kv, rows, offset, qk_scale, 0, full,
                              kc_stride_b, kc_stride_s, vc_stride_b, vc_stride_s,
                              D, BC, BS, False, CAUSAL)
    o, l, m = _paged_fa_inner(o, l, m, q, k_base, v_base, bt_ptr,
                              n_kv, rows, offset, qk_scale, full, hi,
                              kc_stride_b, kc_stride_s, vc_stride_b, vc_stride_s,
                              D, BC, BS, True, CAUSAL)

    o = o / l[:, None]
    o_desc.store([i * BR, 0], o.to(o_ptr.dtype.element_ty))


# ============================================================================
# 2. decode 内核（分页 KV 版 FlashDecoding split-K + 融合 merge）
# ============================================================================
@triton.autotune(configs=_decode_configs(), key=['D', 'M_PAD'])
@triton.jit
def _paged_decode_fused_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                               op_ptr, mp_ptr, lp_ptr, cnt_ptr, bt_ptr,
                               bhn, n_q, n_kv, h_kv, g, qk_scale, chunk, splits,
                               kc_stride_b, kc_stride_h, kc_stride_s,
                               vc_stride_b, vc_stride_h, vc_stride_s,
                               D: tl.constexpr, BS: tl.constexpr,
                               M_PAD: tl.constexpr, BC: tl.constexpr,
                               S_PAD: tl.constexpr, CAUSAL: tl.constexpr):
    """grid = (splits, num_kv_heads)。batch=1、n_q==1（见 host 端分发）。

    阶段 1：每个 program 负责一个 kv head 的 kv [lo, hi) 段，一次性载入该组
    全部 g 个 q head（GNQ = g*n_q 行，q 内存中连续）与同一份 K/V 做
    attention（K/V 只从 HBM 读一次），partial (o, m, l)（dtype 随输入）
    写入全局暂存。
    阶段 2：组内最后一个完成 partial 的 program（atomic 计数）负责
    log-sum-exp 合并、归一化写出，并复位计数器。
    """
    s_id = tl.program_id(0)
    bhk = tl.program_id(1)

    k_base = k_ptr + bhk * kc_stride_h
    v_base = v_ptr + bhk * vc_stride_h

    GNQ = g * n_q
    row0 = bhk * GNQ
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[bhn, D], strides=[D, 1],
                                       block_shape=[M_PAD, D])
    q = q_desc.load([row0, 0])  # 超过 GNQ 的 padding 行自动补零

    lo = s_id * chunk
    hi = tl.minimum(n_kv, lo + chunk)

    # 中间量精度随输入（bf16 输入 -> bf16）
    o = tl.zeros([M_PAD, D], dtype=q.dtype)
    l = tl.zeros([M_PAD], dtype=q.dtype)
    m = tl.full([M_PAD], float("-inf"), dtype=q.dtype)

    offset = n_kv - n_q
    rows = tl.arange(0, M_PAD)
    r_local = rows % n_q
    d_offs = tl.arange(0, D)

    for j0 in range(lo, hi, BC):
        cols = j0 + tl.arange(0, BC)
        cvalid = cols < hi
        lb = cols // BS
        slot = cols % BS
        pb = tl.load(bt_ptr + lb, mask=cvalid, other=0).to(tl.int64)
        k = tl.load(k_base + pb[:, None] * kc_stride_b
                    + slot[:, None] * kc_stride_s + d_offs[None, :],
                    mask=cvalid[:, None], other=0.0)
        # 16bit 输入的 dot 输出恒为 fp32，显式转回
        s = (tl.dot(q, tl.trans(k)) * qk_scale).to(q.dtype)
        mask = cvalid[None, :]
        if CAUSAL:
            mask = mask & (cols[None, :] <= (r_local + offset)[:, None])
        s = tl.where(mask, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m - m_safe).to(q.dtype)
        p = tl.math.exp2(s - m_safe[:, None]).to(q.dtype)
        l = l * alpha + tl.sum(p, axis=1)
        v = tl.load(v_base + pb[:, None] * vc_stride_b
                    + slot[:, None] * vc_stride_s + d_offs[None, :],
                    mask=cvalid[:, None], other=0.0)
        o = o * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(q.dtype)
        m = m_new.to(q.dtype)

    pid = bhk * splits + s_id
    valid = rows < GNQ
    tl.store(op_ptr + (pid * GNQ + rows)[:, None] * D + d_offs[None, :],
             o.to(op_ptr.dtype.element_ty), mask=valid[:, None])
    tl.store(mp_ptr + pid * GNQ + rows, m, mask=valid)
    tl.store(lp_ptr + pid * GNQ + rows, l, mask=valid)

    # ---- 阶段 2：组内最后一个 program 做 merge ----
    tl.debug_barrier()
    old = tl.atomic_add(cnt_ptr + bhk, 1, sem="acq_rel", scope="gpu")
    if old == splits - 1:
        sids = tl.arange(0, S_PAD)
        smask = (sids < splits)[:, None] & valid[None, :]
        pids = bhk * splits + sids
        m_all = tl.load(mp_ptr + pids[:, None] * GNQ + rows[None, :],
                        mask=smask, other=float("-inf"))
        l_all = tl.load(lp_ptr + pids[:, None] * GNQ + rows[None, :],
                        mask=smask, other=0.0)
        m_star = tl.max(m_all, axis=0)
        m_gsafe = tl.where(m_star == float("-inf"), 0.0, m_star)
        w_all = tl.math.exp2(m_all - m_gsafe[None, :]).to(q.dtype)
        l_g = tl.sum(l_all * w_all, axis=0)

        o_g = tl.zeros([M_PAD, D], dtype=q.dtype)
        for sid in range(splits):
            pid2 = bhk * splits + sid
            m_s = tl.load(mp_ptr + pid2 * GNQ + rows, mask=valid,
                          other=float("-inf"))
            w = tl.math.exp2(m_s - m_gsafe).to(q.dtype)
            o_s = tl.load(op_ptr + (pid2 * GNQ + rows)[:, None] * D
                          + d_offs[None, :], mask=valid[:, None], other=0.0)
            o_g += w[:, None] * o_s

        o_g = o_g / l_g[:, None]  # l_g==0 只在 valid=False 的 padding 行
        tl.store(o_ptr + (row0 + rows)[:, None] * D + d_offs[None, :],
                 o_g.to(o_ptr.dtype.element_ty), mask=valid[:, None])
        tl.store(cnt_ptr + bhk, 0)  # 复位供下一次调用


# ============================================================================
# host 端
# ============================================================================
_DECODE_SCRATCH = {}
_N_SM = [None]


def _decode_scratch(BHK, splits, GNQ, d, device, dtype):
    key = (BHK, splits, GNQ, d, device, dtype)
    if len(_DECODE_SCRATCH) > 32:
        _DECODE_SCRATCH.clear()
    if key not in _DECODE_SCRATCH:
        o_part = torch.empty(BHK * splits * GNQ * d, device=device, dtype=dtype)
        m_part = torch.empty(BHK * splits * GNQ, device=device, dtype=dtype)
        l_part = torch.empty(BHK * splits * GNQ, device=device, dtype=dtype)
        cnt = torch.zeros(BHK, device=device, dtype=torch.int32)
        _DECODE_SCRATCH[key] = (o_part, m_part, l_part, cnt)
    return _DECODE_SCRATCH[key]


def _decode_one(q_s, k_cache, v_cache, o_s, bt_s, n_kv, qk_scale):
    """单条请求的 decode（n_q == 1）。q_s/o_s: [1, h, d] 视图。"""
    h = q_s.shape[1]
    h_kv = k_cache.shape[1]
    g = h // h_kv
    d = q_s.shape[2]
    bs = k_cache.shape[2]
    GNQ = g  # n_q = 1
    M_PAD = max(16, triton.next_power_of_2(GNQ))
    BHK = h_kv
    if _N_SM[0] is None:
        _N_SM[0] = torch.cuda.get_device_properties(q_s.device).multi_processor_count
    # 确定性 splits 启发式：目标 program 数 ≈ 4x SM 数
    # （H20、n_kv~95k 实测 splits∈[27,39] 最优 ~1.78 TB/s，取 4x SM）
    cap = min(triton.cdiv(n_kv, 256), 64)
    splits = min(max(triton.cdiv(4 * _N_SM[0], BHK), 1), cap)
    chunk = triton.cdiv(triton.cdiv(n_kv, splits), 128) * 128
    sp = triton.cdiv(n_kv, chunk)
    o_part, m_part, l_part, cnt = _decode_scratch(BHK, sp, GNQ, d,
                                                  q_s.device, q_s.dtype)
    _paged_decode_fused_kernel[(sp, BHK)](
        q_s, k_cache, v_cache, o_s,
        o_part, m_part, l_part, cnt, bt_s,
        h, 1, n_kv, h_kv, g, qk_scale, chunk, sp,
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        D=d, BS=bs, M_PAD=M_PAD, S_PAD=64, CAUSAL=True)


def _prefill_one(q_s, k_cache, v_cache, o_s, bt_s, n_q, n_kv, qk_scale):
    """单条请求的 prefill（任意 n_q）。q_s/o_s: [n_q, h, d] 视图。"""
    h = q_s.shape[1]
    h_kv = k_cache.shape[1]
    g = h // h_kv
    d = q_s.shape[2]
    bs = k_cache.shape[2]
    grid = lambda meta: (triton.cdiv(n_q, meta['BR']), h)
    _paged_prefill_kernel[grid](
        q_s, k_cache, v_cache, o_s, bt_s,
        q_s.stride(1), q_s.stride(0),
        o_s.stride(1), o_s.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        n_q, n_kv, h, g, qk_scale,
        D=d, BS=bs, CAUSAL=True)


# decode 走 split-K 内核的最小 context 长度（短 context 用 prefill 内核更划算）
_DECODE_MIN_KV = 1024


# 可选的 CPU 元数据侧通道（避免每次调用对 GPU 张量 .tolist() 造成 64 次/步的
# 流同步，从而让 host 端 Python/launch 开销与 GPU 计算重叠）。
# 由 CustomTritonImpl.forward 在调用前设置；未设置时回退为 .tolist()。
_CPU_META = None


def set_cpu_metadata(query_start_loc_cpu, seq_lens_cpu) -> None:
    """预置本步的 CPU 侧元数据（list）。任一参数为 None 时清除。"""
    global _CPU_META
    if query_start_loc_cpu is None or seq_lens_cpu is None:
        _CPU_META = None
    else:
        _CPU_META = (query_start_loc_cpu.tolist(), seq_lens_cpu.tolist())


def paged_attention_triton(
    query: torch.Tensor,        # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size]
    output: torch.Tensor,       # [num_tokens, num_heads, head_size]  (原地写入)
    query_start_loc: torch.Tensor,  # [num_seqs + 1] int32
    seq_lens: torch.Tensor,         # [num_seqs] int32
    token_seq_idx: torch.Tensor,    # [num_tokens] int32：每个 token 属于哪条请求
    block_table: torch.Tensor,      # [num_seqs, max_num_blocks] int32
    scale: float,
) -> torch.Tensor:
    """
    分页注意力（causal, GQA, prefill+decode 通用）。output 原地写入并返回。

    按请求逐条分发：n_q == 1 且 context 较长 -> split-K decode 内核；
    其余 -> prefill 内核（causal 时 query 对齐到 kv 末尾，语义一致）。
    """
    num_tokens, num_heads, head_size = query.shape
    if num_tokens == 0:
        return output
    num_kv_heads = key_cache.shape[1]
    g = num_heads // num_kv_heads
    qk_scale = scale * 1.4426950408889634  # 乘入 log2(e)，kernel 内用 exp2

    if _CPU_META is not None:
        qsl, sls = _CPU_META
    else:
        qsl = query_start_loc.tolist()
        sls = seq_lens.tolist()
    for s in range(len(sls)):
        qs, qe = qsl[s], qsl[s + 1]
        n_q = qe - qs
        n_kv = sls[s]
        if n_q <= 0:
            continue
        bt_s = block_table[s]
        q_s = query[qs:qe]
        o_s = output[qs:qe]
        if n_q == 1 and n_kv >= _DECODE_MIN_KV and g <= 128:
            _decode_one(q_s, key_cache, value_cache, o_s, bt_s, n_kv, qk_scale)
        else:
            _prefill_one(q_s, key_cache, value_cache, o_s, bt_s, n_q, n_kv,
                         qk_scale)
    return output
