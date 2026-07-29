# SPDX-License-Identifier: Apache-2.0
"""分页 StreamingLLM 注意力 v3p2 —— 在 v3p1 基础上优化 decode 循环（其余不变）。

v3p1 的 decode 用 `for n in range(0, ceil(qpos/BLOCK_N))` 扫全程再靠 if 跳空洞块：
seq=100000 时循环体走 ~1563 次，只有 ~17 次真干活，1546 次纯循环控制/判断开销。
v3p2 改成【只迭代 sink 段 + window 段两小段】——循环次数直接降到 ~(SINK+WINDOW)/BLOCK_N，
无空转、无逐块分支判断，decode 更快；数值/稀疏语义与 v3p1 完全一致。

对外只暴露 paged_attention_triton(...)，签名/语义与 README Part 3.1 完全一致
（causal + GQA + 分页寻址，output 原地写入并返回）。想调稀疏度改 _SINK / _WINDOW。
    cp sota_dsl_v3p2.py custom_backend/triton_attention.py
"""
import math

import torch
import triton
import triton.language as tl

# ---- StreamingLLM 超参：每个 query 只看 sink(前 _SINK 个) + 最近 _WINDOW 个 KV ----
_SINK = 4
_WINDOW = 1024

# ---- 分块常量 ----
_BLOCK_M = 64          # prefill: 每个 program 处理的 query 数
_BLOCK_N = 64          # KV tile 宽度（prefill）
_BLOCK_N_DECODE = 64   # KV tile 宽度（decode）


@triton.jit
def _paged_prefill_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    query_start_loc_ptr, seq_lens_ptr, seq_id_map_ptr, block_table_ptr,
    QK_SCALE,                              # scale * log2(e)，预乘进 q（log2 域）
    num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SINK: tl.constexpr, WINDOW: tl.constexpr, max_num_blocks: tl.constexpr,
    q_stride_t, q_stride_h, q_stride_d, o_stride_t, o_stride_h, o_stride_d,
    kc_stride_b, kc_stride_h, kc_stride_s, kc_stride_d,
    vc_stride_b, vc_stride_h, vc_stride_s, vc_stride_d,
    block_table_stride,
):
    # grid = (num_m_blocks, num_heads, num_prefill_seqs)。每个 program 算 1 个 q_head 的
    # BLOCK_M 个 query，tl.dot 吃 tensor core；K/V 用 block_table 逐列 gather 成连续 tile，
    # 只遍历 sink+window 块（跳空洞 -> 稀疏收益）。
    pid_m = tl.program_id(0)
    q_head = tl.program_id(1)
    seq_idx = tl.load(seq_id_map_ptr + tl.program_id(2))
    kv_head = q_head // (num_heads // num_kv_heads)

    q_start = tl.load(query_start_loc_ptr + seq_idx)
    q_end = tl.load(query_start_loc_ptr + seq_idx + 1)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    q_len = q_end - q_start
    context = seq_len - q_len

    if pid_m * BLOCK_M < q_len:
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        m_in = offs_m < q_len
        tok = q_start + tl.where(m_in, offs_m, 0)          # 越界行指针 clamp 到 q_start
        qpos = context + offs_m                            # [BLOCK_M] 绝对位置

        q_ptrs = q_ptr + tok[:, None] * q_stride_t + q_head * q_stride_h + offs_d[None, :] * q_stride_d
        q = (tl.load(q_ptrs) * QK_SCALE).to(q_ptr.dtype.element_ty)   # [BLOCK_M, HEAD_DIM]

        m_i = tl.full([BLOCK_M], -1e30, tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

        min_qpos = context + pid_m * BLOCK_M
        max_qpos = context + tl.minimum((pid_m + 1) * BLOCK_M, q_len) - 1
        win_lo = min_qpos - WINDOW + 1                     # 本 m-block 最宽 window 下界
        # 【v3p2 优化】不扫全程(n_end~ceil(seq/BLOCK_N))再 if 跳块——那样循环次数只随 seq、
        # 与 WINDOW 无关(512/1024 空转次数相同 → 一样慢)。改成用索引重映射只迭代
        # sink 段 [0,SINK) + window 段 [win_start, max_qpos]，循环次数 ~(SINK+WINDOW)/BLOCK_N。
        n_causal = (max_qpos + BLOCK_N) // BLOCK_N                        # causal 上界块数
        sink_blocks = tl.minimum((SINK + BLOCK_N - 1) // BLOCK_N, n_causal)  # sink 段块数(clamp 到 causal)
        ws_block = tl.maximum(win_lo, sink_blocks * BLOCK_N) // BLOCK_N   # window 段起始块(对齐, >=sink段末尾)
        n_win = n_causal - ws_block                                      # window 段块数(到 causal 上界)

        for idx in range(0, sink_blocks + n_win):
            n = tl.where(idx < sink_blocks, idx, ws_block + (idx - sink_blocks))  # 重映射到 sink/window 块
            block_lo = n * BLOCK_N
            offs_n = block_lo + tl.arange(0, BLOCK_N)
            col_in = offs_n < seq_len
            logical_block = offs_n // BLOCK_SIZE
            slot = offs_n % BLOCK_SIZE
            pb = tl.load(block_table_ptr + seq_idx * block_table_stride + logical_block,
                         mask=col_in & (logical_block < max_num_blocks), other=0)
            # K tile [HEAD_DIM, BLOCK_N]（逐列分页 gather，转置布局供 tl.dot）
            k_ptrs = (k_cache_ptr + pb[None, :] * kc_stride_b + kv_head * kc_stride_h
                      + slot[None, :] * kc_stride_s + offs_d[:, None] * kc_stride_d)
            k = tl.load(k_ptrs, mask=col_in[None, :], other=0.0)
            qk = tl.dot(q, k)                          # [BLOCK_M, BLOCK_N]，已含 scale

            j = offs_n[None, :]
            causal = j <= qpos[:, None]
            stream = (j < SINK) | (j > (qpos[:, None] - WINDOW))
            valid = causal & stream & col_in[None, :]
            qk = tl.where(valid, qk, -float("inf"))

            m_next = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_next[:, None])
            alpha = tl.math.exp2(m_i - m_next)
            acc = acc * alpha[:, None]
            v_ptrs = (v_cache_ptr + pb[:, None] * vc_stride_b + kv_head * vc_stride_h
                      + slot[:, None] * vc_stride_s + offs_d[None, :] * vc_stride_d)
            v = tl.load(v_ptrs, mask=col_in[:, None], other=0.0)   # [BLOCK_N, HEAD_DIM]
            acc = tl.dot(p.to(v.dtype), v, acc)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_next

        out = acc / l_i[:, None]
        o_ptrs = out_ptr + tok[:, None] * o_stride_t + q_head * o_stride_h + offs_d[None, :] * o_stride_d
        tl.store(o_ptrs, out.to(out_ptr.dtype.element_ty), mask=m_in[:, None])


@triton.jit
def _decode_block(block_lo, q, m_i, l_i, acc, seq_idx, kv_head, qpos, seq_len,
                  block_table_ptr, k_cache_ptr, v_cache_ptr, block_table_stride,
                  kc_stride_b, kc_stride_h, kc_stride_s, kc_stride_d,
                  vc_stride_b, vc_stride_h, vc_stride_s, vc_stride_d,
                  m_valid, offs_d, SINK: tl.constexpr, WINDOW: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, max_num_blocks: tl.constexpr):
    """在一个 KV block [block_lo, block_lo+BLOCK_N) 上做一步 online-softmax，返回更新后的 (m_i,l_i,acc)。
    streaming mask 精确判 sink/window/causal，故调用方按块边界迭代即可（对齐到块不影响数值）。"""
    offs_n = block_lo + tl.arange(0, BLOCK_N)
    col_in = offs_n < seq_len
    logical_block = offs_n // BLOCK_SIZE
    slot = offs_n % BLOCK_SIZE
    pb = tl.load(block_table_ptr + seq_idx * block_table_stride + logical_block,
                 mask=col_in & (logical_block < max_num_blocks), other=0)
    k_ptrs = (k_cache_ptr + pb[None, :] * kc_stride_b + kv_head * kc_stride_h
              + slot[None, :] * kc_stride_s + offs_d[:, None] * kc_stride_d)
    k = tl.load(k_ptrs, mask=col_in[None, :], other=0.0)   # [HEAD_DIM, BLOCK_N]
    qk = tl.dot(q, k)                                      # [PAD_M, BLOCK_N]
    j = offs_n[None, :]
    stream = (j < SINK) | (j > (qpos - WINDOW))
    valid = m_valid[:, None] & col_in[None, :] & (j <= qpos) & stream
    qk = tl.where(valid, qk, -float("inf"))
    m_next = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_next[:, None])
    alpha = tl.math.exp2(m_i - m_next)
    acc = acc * alpha[:, None]
    v_ptrs = (v_cache_ptr + pb[:, None] * vc_stride_b + kv_head * vc_stride_h
              + slot[:, None] * vc_stride_s + offs_d[None, :] * vc_stride_d)
    v = tl.load(v_ptrs, mask=col_in[:, None], other=0.0)   # [BLOCK_N, HEAD_DIM]
    acc = tl.dot(p.to(v.dtype), v, acc)
    l_i = l_i * alpha + tl.sum(p, 1)
    return m_next, l_i, acc


@triton.jit
def _paged_decode_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    query_start_loc_ptr, seq_lens_ptr, seq_id_map_ptr, block_table_ptr,
    QK_SCALE,
    num_heads: tl.constexpr, num_kv_heads: tl.constexpr, GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
    PAD_M: tl.constexpr, SINK: tl.constexpr, WINDOW: tl.constexpr, max_num_blocks: tl.constexpr,
    q_stride_t, q_stride_h, q_stride_d, o_stride_t, o_stride_h, o_stride_d,
    kc_stride_b, kc_stride_h, kc_stride_s, kc_stride_d,
    vc_stride_b, vc_stride_h, vc_stride_s, vc_stride_d,
    block_table_stride,
):
    # grid = (num_kv_heads, num_decode_seqs)。GQA group 打包 + tl.dot。
    # 【v3p2 优化】不扫全程再跳块，而是【只迭代 sink 段 + window 段】：循环次数 ~(SINK+WINDOW)/BLOCK_N，
    # 无空转、无逐块 if。两段起点按 BLOCK_N 对齐，window 段起点 clamp 到 >=sink 段末尾避免重复计算。
    kv_head = tl.program_id(0)
    seq_idx = tl.load(seq_id_map_ptr + tl.program_id(1))
    q_start = tl.load(query_start_loc_ptr + seq_idx)       # decode: q_len==1，token = q_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    qpos = seq_len - 1

    offs_m = tl.arange(0, PAD_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < GROUP
    q_head = kv_head * GROUP + tl.where(m_valid, offs_m, 0)
    q_ptrs = q_ptr + q_start * q_stride_t + q_head[:, None] * q_stride_h + offs_d[None, :] * q_stride_d
    q = (tl.load(q_ptrs) * QK_SCALE).to(q_ptr.dtype.element_ty)   # [PAD_M, HEAD_DIM]

    m_i = tl.full([PAD_M], -1e30, tl.float32)
    l_i = tl.zeros([PAD_M], tl.float32)
    acc = tl.zeros([PAD_M, HEAD_DIM], tl.float32)

    args = (q_start, kv_head, qpos, seq_len, block_table_ptr, k_cache_ptr, v_cache_ptr,
            block_table_stride, kc_stride_b, kc_stride_h, kc_stride_s, kc_stride_d,
            vc_stride_b, vc_stride_h, vc_stride_s, vc_stride_d, m_valid, offs_d)
    # sink 段: block [0, SINK)，共 ceil(SINK/BLOCK_N) 个块（SINK 小，通常 1 个）。
    sink_blocks = (SINK + BLOCK_N - 1) // BLOCK_N
    for nb in range(0, sink_blocks):
        m_i, l_i, acc = _decode_block(
            nb * BLOCK_N, q, m_i, l_i, acc, *args,
            SINK=SINK, WINDOW=WINDOW, BLOCK_SIZE=BLOCK_SIZE, BLOCK_N=BLOCK_N, max_num_blocks=max_num_blocks)
    # window 段: 从 max(win_lo, sink 段末尾) 对齐块起，到 qpos。次数 ~ WINDOW/BLOCK_N。
    win_lo = qpos - WINDOW + 1
    win_start = tl.maximum(win_lo, sink_blocks * BLOCK_N)
    win_start = (win_start // BLOCK_N) * BLOCK_N                # 对齐块边界
    n_win = (qpos - win_start + BLOCK_N) // BLOCK_N            # ceil((qpos-win_start+1)/BLOCK_N)
    for nb in range(0, n_win):
        m_i, l_i, acc = _decode_block(
            win_start + nb * BLOCK_N, q, m_i, l_i, acc, *args,
            SINK=SINK, WINDOW=WINDOW, BLOCK_SIZE=BLOCK_SIZE, BLOCK_N=BLOCK_N, max_num_blocks=max_num_blocks)

    out = acc / l_i[:, None]
    o_head = kv_head * GROUP + offs_m
    o_ptrs = out_ptr + q_start * o_stride_t + o_head[:, None] * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptrs, out.to(out_ptr.dtype.element_ty), mask=m_valid[:, None])


def paged_attention_triton(
    query,            # [num_tokens, num_heads, head_size]
    key_cache,        # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache,      # [num_blocks, num_kv_heads, block_size, head_size]
    output,           # [num_tokens, num_heads, head_size]  (原地写入)
    query_start_loc,  # [num_seqs + 1] int32
    seq_lens,         # [num_seqs] int32
    token_seq_idx,    # [num_tokens] int32（本实现不用，仅对齐 README 接口）
    block_table,      # [num_seqs, max_num_blocks] int32
    scale,
):
    """host 按 q_len 分派：prefill(q_len>1) 走 _paged_prefill_kernel，decode(q_len==1)
    走 _paged_decode_kernel。全程分页寻址、无稠密化；sink/window 取 _SINK/_WINDOW。"""
    _, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]
    max_num_blocks = block_table.shape[1]
    group = num_heads // num_kv_heads
    qk_scale = scale * math.log2(math.e)          # 预乘进 q，内循环用 exp2（log2 域）
    dev = query.device

    q_lens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()   # 一次 host 同步
    prefill_ids = [i for i, L in enumerate(q_lens) if L > 1]
    decode_ids = [i for i, L in enumerate(q_lens) if L == 1]

    common = dict(
        q_stride_t=query.stride(0), q_stride_h=query.stride(1), q_stride_d=query.stride(2),
        o_stride_t=output.stride(0), o_stride_h=output.stride(1), o_stride_d=output.stride(2),
        kc_stride_b=key_cache.stride(0), kc_stride_h=key_cache.stride(1),
        kc_stride_s=key_cache.stride(2), kc_stride_d=key_cache.stride(3),
        vc_stride_b=value_cache.stride(0), vc_stride_h=value_cache.stride(1),
        vc_stride_s=value_cache.stride(2), vc_stride_d=value_cache.stride(3),
        block_table_stride=block_table.stride(0),
    )

    if prefill_ids:
        seq_map = torch.tensor(prefill_ids, device=dev, dtype=torch.int32)
        max_m = (max(q_lens[i] for i in prefill_ids) + _BLOCK_M - 1) // _BLOCK_M
        _paged_prefill_kernel[(max_m, num_heads, len(prefill_ids))](
            output, query, key_cache, value_cache,
            query_start_loc, seq_lens, seq_map, block_table, qk_scale,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            HEAD_DIM=head_size, BLOCK_SIZE=block_size, BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N,
            SINK=_SINK, WINDOW=_WINDOW, max_num_blocks=max_num_blocks, **common,
        )

    if decode_ids:
        seq_map = torch.tensor(decode_ids, device=dev, dtype=torch.int32)
        pad_m = max(16, triton.next_power_of_2(group))
        _paged_decode_kernel[(num_kv_heads, len(decode_ids))](
            output, query, key_cache, value_cache,
            query_start_loc, seq_lens, seq_map, block_table, qk_scale,
            num_heads=num_heads, num_kv_heads=num_kv_heads, GROUP=group,
            HEAD_DIM=head_size, BLOCK_SIZE=block_size, BLOCK_N=_BLOCK_N_DECODE,
            PAD_M=pad_m, SINK=_SINK, WINDOW=_WINDOW, max_num_blocks=max_num_blocks, **common,
        )

    return output


# ============================================================================
# 自测（可选，需 CUDA GPU）：python sota_dsl_v3p1.py
# 构造分页 KV cache（block_table 打乱物理块以验证寻址），对拍朴素 streaming 参考，
# 覆盖 prefill / decode / 变长混合 batch。
# ============================================================================
def _run_paged_case(name, reqs, block_size=16, sink=4, window=64,
                    dtype=torch.bfloat16, rtol=2e-2, atol=2e-2):
    global _SINK, _WINDOW
    saved = (_SINK, _WINDOW)
    _SINK, _WINDOW = sink, window
    try:
        torch.manual_seed(0)
        dev, h, hkv, d = "cuda", 64, 8, 128
        g = h // hkv
        seq_lens = [c + q for c, q in reqs]
        q_lens = [q for _, q in reqs]
        qsl = [0]
        for q in q_lens:
            qsl.append(qsl[-1] + q)
        token_seq_idx = [i for i, q in enumerate(q_lens) for _ in range(q)]
        nlog = [(sl + block_size - 1) // block_size for sl in seq_lens]
        max_logical, total = max(nlog), sum(nlog) + 4
        perm = torch.randperm(total, device=dev).to(torch.int32)
        kc = torch.randn(total, hkv, block_size, d, device=dev, dtype=dtype)
        vc = torch.randn(total, hkv, block_size, d, device=dev, dtype=dtype)
        bt = torch.zeros(len(reqs), max_logical, device=dev, dtype=torch.int32)
        k_log, v_log, ptr = [], [], 0
        for r in range(len(reqs)):
            sl, nb = seq_lens[r], nlog[r]
            blk = perm[ptr:ptr + nb]
            ptr += nb
            bt[r, :nb] = blk
            kl = torch.randn(hkv, sl, d, device=dev, dtype=dtype)
            vl = torch.randn(hkv, sl, d, device=dev, dtype=dtype)
            k_log.append(kl)
            v_log.append(vl)
            for j in range(sl):
                pb = int(blk[j // block_size])
                kc[pb, :, j % block_size, :] = kl[:, j, :]
                vc[pb, :, j % block_size, :] = vl[:, j, :]
        query = torch.randn(qsl[-1], h, d, device=dev, dtype=dtype)
        output = torch.empty_like(query)
        scale = 1.0 / math.sqrt(d)
        paged_attention_triton(
            query, kc, vc, output,
            torch.tensor(qsl, device=dev, dtype=torch.int32),
            torch.tensor(seq_lens, device=dev, dtype=torch.int32),
            torch.tensor(token_seq_idx, device=dev, dtype=torch.int32), bt, scale)
        ref = torch.empty(qsl[-1], h, d, device=dev, dtype=torch.float32)
        for r in range(len(reqs)):
            sl, ctx = seq_lens[r], seq_lens[r] - q_lens[r]
            jv = torch.arange(sl, device=dev)
            for i in range(q_lens[r]):
                tok, qpos = qsl[r] + i, ctx + i
                valid = (jv <= qpos) & ((jv < sink) | (jv > qpos - window))
                for hi in range(h):
                    sc = (query[tok, hi].float() * scale) @ k_log[r][hi // g].float().T
                    sc = sc.masked_fill(~valid, float("-inf"))
                    ref[tok, hi] = torch.softmax(sc, -1) @ v_log[r][hi // g].float()
        err = (output.float() - ref).abs().max().item()
        ok = torch.allclose(output.float(), ref, rtol=rtol, atol=atol)
        print(f"[{name}] max_abs_err={err:.4e} -> {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        _SINK, _WINDOW = saved


def _main():
    ok = True
    ok &= _run_paged_case("paged-prefill", [(0, 48)])
    ok &= _run_paged_case("paged-prefill-multiblock", [(0, 130)])
    ok &= _run_paged_case("paged-decode-long", [(300, 1)])
    ok &= _run_paged_case("paged-mixed", [(0, 40), (300, 1), (100, 16), (20, 1)])
    print("=" * 40)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
