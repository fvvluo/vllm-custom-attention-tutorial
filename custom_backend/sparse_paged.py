# SPDX-License-Identifier: Apache-2.0
"""稀疏分页 decode —— query-aware 块稀疏，接入 vLLM CUSTOM 后端。

在 triton_attention.py 的 fp8 分页 decode 之上加一层"只读关键块"：
  Phase A（score kernel）：对每个 (token, kv_head)，用**块内 mean-pooled K** 估每块分数
     score[blk] = max_over_group( q · kmean[blk] )，选 top-k 块（+sink+recent window）。
     只读 K（不读 V、不做 softmax/PV），比全 attention 轻。
  Phase B（sparse decode kernel）：只遍历 selected_blocks 里的物理块做 flash-attention。

设计取舍（诚实）：
  - vLLM 的 KV cache 由 reshape_and_cache 增量写，没有现成的 per-block K min/max 摘要，
    故这里在 kernel 内**在线 mean-pool** 打分（读一遍 K）。总访存 = 打分读 K(≈半个KV) +
    decode 读选中块 K+V(sparsity×KV)。sparsity 足够小时仍省。
  - 只用于纯 decode（q_len==1）；prefill / chunked 走原路径（不稀疏）。
  - 走 bf16 路径（KV cache bf16, auto dtype）：稀疏与 fp8 正交，先验证稀疏端到端；
    fp8+sparse 可后续叠加。
  - 质量：真实模型注意力天然稀疏，top-k 应保住质量（smoke=42 验证），不同于 randn。

env: CUSTOM_SPARSE=1 开启；CUSTOM_SPARSITY（默认0.25）、CUSTOM_SPARSE_MIN_LEN（默认2048，
     短于此不稀疏，固定开销不划算）。
"""
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _score_blocks_kernel(
    q_ptr,                # [num_tokens, num_heads, head_size]
    k_cache_ptr,          # [num_blocks, num_kv_heads, block_size, head_size]
    token_seq_idx_ptr,    # [num_tokens]
    seq_lens_ptr,         # [num_seqs]
    block_table_ptr,      # [num_seqs, max_num_blocks]
    score_ptr,            # [num_tokens, num_kv_heads, max_num_blocks] fp32
    scale,
    num_kv_heads: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    max_num_blocks: tl.constexpr,
    n_sink: tl.constexpr,
    recent_win: tl.constexpr,
    q_stride_t, q_stride_h,
    kc_stride_b, kc_stride_h, kc_stride_s,
    block_table_stride,
    sc_st, sc_sh,
):
    # grid = (num_tokens, num_kv_heads, num_block_tiles)
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    blk_tile = tl.program_id(2)   # 每个 program 处理 BLOCK_TILE 个块中的 1 个块

    seq_idx = tl.load(token_seq_idx_ptr + token_idx)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_valid_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    blk = blk_tile
    if blk >= num_valid_blocks:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_s = tl.arange(0, BLOCK_SIZE)
    m_valid = offs_m < GROUP
    head_ids = kv_head_idx * GROUP + offs_m

    # Q tile [BLOCK_M, HEAD_SIZE]
    q = tl.load(q_ptr + token_idx * q_stride_t + head_ids[:, None] * q_stride_h + offs_d[None, :],
                mask=m_valid[:, None], other=0.0).to(tl.float32)

    # 该逻辑块的物理块号
    pb = tl.load(block_table_ptr + seq_idx * block_table_stride + blk,
                 mask=blk < max_num_blocks, other=0)
    # 读块内 K [BLOCK_SIZE, HEAD_SIZE]，mean-pool 到 [HEAD_SIZE]
    k_ptrs = (k_cache_ptr + pb * kc_stride_b + kv_head_idx * kc_stride_h
              + offs_s[:, None] * kc_stride_s + offs_d[None, :])
    # 屏蔽块内越过 seq_len 的槽位
    slot_pos = blk * BLOCK_SIZE + offs_s
    slot_valid = slot_pos < seq_len
    k = tl.load(k_ptrs, mask=slot_valid[:, None], other=0.0).to(tl.float32)
    cnt = tl.maximum(tl.sum(slot_valid.to(tl.float32)), 1.0)
    kmean = tl.sum(k, axis=0) / cnt            # [HEAD_SIZE]

    # score = max_over_group( q · kmean ) * scale
    qk = tl.sum(q * kmean[None, :], axis=1)    # [BLOCK_M]
    qk = tl.where(m_valid, qk, -float("inf"))
    score = tl.max(qk, axis=0) * scale

    # sink / recent window 强制入选（分数抬到 +inf）
    is_sink = blk < n_sink
    is_recent = blk >= (num_valid_blocks - recent_win)
    score = tl.where(is_sink | is_recent, float("inf"), score)

    tl.store(score_ptr + token_idx * sc_st + kv_head_idx * sc_sh + blk, score)


@triton.jit
def _sparse_paged_attn_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    token_seq_idx_ptr, seq_lens_ptr, block_table_ptr,
    selected_ptr,         # [num_tokens, num_kv_heads, budget] int32 选中块号
    num_sel_ptr,          # [num_tokens, num_kv_heads] int32
    scale,
    num_kv_heads: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP: tl.constexpr,
    budget: tl.constexpr,
    max_num_blocks: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    q_stride_t, q_stride_h,
    kc_stride_b, kc_stride_h, kc_stride_s,
    vc_stride_b, vc_stride_h, vc_stride_s,
    block_table_stride,
    sel_st, sel_sh,
    ns_st, ns_sh,
    op_ptr, ml_ptr,
    op_st, op_sh, op_ss, op_sm,
    ml_st, ml_sh, ml_ss, ml_se,
):
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    pid_split = tl.program_id(2)

    seq_idx = tl.load(token_seq_idx_ptr + token_idx)
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_s = tl.arange(0, BLOCK_SIZE)
    m_valid = offs_m < GROUP
    head_ids = kv_head_idx * GROUP + offs_m

    q = tl.load(q_ptr + token_idx * q_stride_t + head_ids[:, None] * q_stride_h + offs_d[None, :],
                mask=m_valid[:, None], other=0.0).to(tl.bfloat16)

    n_sel = tl.load(num_sel_ptr + token_idx * ns_st + kv_head_idx * ns_sh)
    # 本 split 负责选中块列表 [s_start, s_end)
    sel_per = (n_sel + NUM_SPLITS - 1) // NUM_SPLITS
    s_start = pid_split * sel_per
    s_end = tl.minimum(s_start + sel_per, n_sel)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], tl.float32)

    sel_base = selected_ptr + token_idx * sel_st + kv_head_idx * sel_sh
    for sidx in range(s_start, s_end):
        blk = tl.load(sel_base + sidx)            # 选中的逻辑块号
        pb = tl.load(block_table_ptr + seq_idx * block_table_stride + blk,
                     mask=blk < max_num_blocks, other=0)
        slot_pos = blk * BLOCK_SIZE + offs_s
        col_active = slot_pos < seq_len
        k_ptrs = (k_cache_ptr + pb * kc_stride_b + kv_head_idx * kc_stride_h
                  + offs_s[:, None] * kc_stride_s + offs_d[None, :])
        v_ptrs = (v_cache_ptr + pb * vc_stride_b + kv_head_idx * vc_stride_h
                  + offs_s[:, None] * vc_stride_s + offs_d[None, :])
        k_bf = tl.load(k_ptrs, mask=col_active[:, None], other=0.0).to(tl.bfloat16)
        v_bf = tl.load(v_ptrs, mask=col_active[:, None], other=0.0).to(tl.bfloat16)
        qk = tl.dot(q, k_bf.T, out_dtype=tl.float32) * scale       # [BLOCK_M, BLOCK_SIZE]
        qk = tl.where(col_active[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v_bf, out_dtype=tl.float32)
        m_i = m_new

    op_base = op_ptr + token_idx * op_st + kv_head_idx * op_sh + pid_split * op_ss
    tl.store(op_base + offs_m[:, None] * op_sm + offs_d[None, :], acc, mask=m_valid[:, None])
    ml_base = ml_ptr + token_idx * ml_st + kv_head_idx * ml_sh + pid_split * ml_ss
    tl.store(ml_base + 0 * ml_se + offs_m, m_i, mask=m_valid)
    tl.store(ml_base + 1 * ml_se + offs_m, l_i, mask=m_valid)


def sparse_paged_decode(query, key_cache, value_cache, output,
                        seq_lens, token_seq_idx, block_table, scale,
                        combine_kernel, sparsity=0.25, n_sink=1, recent_win=8,
                        return_selected=False):
    """稀疏分页 decode（纯 decode，q_len==1）。原地写 output，返回 output。

    combine_kernel: 复用 triton_attention._fp8_decode_combine_kernel。
    """
    num_tokens, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]
    max_num_blocks = block_table.shape[1]
    group = num_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group))
    dev = query.device

    # 有效块数（用 seq_lens 的 host 值算 budget；decode eager 路径允许）
    max_seq = int(seq_lens.max().item())
    num_valid_blocks = (max_seq + block_size - 1) // block_size
    budget = max(n_sink + recent_win + 1, int(round(num_valid_blocks * sparsity)))
    budget = min(budget, num_valid_blocks)

    # ---- Phase A: 打分 ----
    score = torch.full((num_tokens, num_kv_heads, max_num_blocks), -float("inf"),
                       dtype=torch.float32, device=dev)
    grid_score = (num_tokens, num_kv_heads, num_valid_blocks)
    _score_blocks_kernel[grid_score](
        query, key_cache, token_seq_idx, seq_lens, block_table, score, scale,
        num_kv_heads=num_kv_heads, HEAD_SIZE=head_size, BLOCK_SIZE=block_size,
        GROUP=group, BLOCK_M=block_m, max_num_blocks=max_num_blocks,
        n_sink=n_sink, recent_win=recent_win,
        q_stride_t=query.stride(0), q_stride_h=query.stride(1),
        kc_stride_b=key_cache.stride(0), kc_stride_h=key_cache.stride(1),
        kc_stride_s=key_cache.stride(2),
        block_table_stride=block_table.stride(0),
        sc_st=score.stride(0), sc_sh=score.stride(1),
        num_warps=4, num_stages=2,
    )
    # ---- 选块（topk，不排序）----
    selected = torch.topk(score, budget, dim=-1, sorted=False).indices.to(torch.int32)
    num_sel = torch.full((num_tokens, num_kv_heads), budget, dtype=torch.int32, device=dev)

    # ---- Phase B: 稀疏 decode + combine ----
    try:
        num_sm = torch.cuda.get_device_properties(dev).multi_processor_count
    except Exception:
        num_sm = 78
    target_ctas = 8 * num_sm
    denom = max(1, num_tokens * num_kv_heads)
    num_splits = max(1, min(target_ctas // denom, budget))
    num_splits = 1 << (num_splits.bit_length() - 1)

    op = torch.empty((num_tokens, num_kv_heads, num_splits, block_m, head_size),
                     dtype=torch.float32, device=dev)
    ml = torch.empty((num_tokens, num_kv_heads, num_splits, 2, block_m),
                     dtype=torch.float32, device=dev)
    grid = (num_tokens, num_kv_heads, num_splits)
    _sparse_paged_attn_kernel[grid](
        output, query, key_cache, value_cache,
        token_seq_idx, seq_lens, block_table, selected, num_sel, scale,
        num_kv_heads=num_kv_heads, HEAD_SIZE=head_size, BLOCK_SIZE=block_size,
        BLOCK_M=block_m, GROUP=group, budget=budget, max_num_blocks=max_num_blocks,
        NUM_SPLITS=num_splits,
        q_stride_t=query.stride(0), q_stride_h=query.stride(1),
        kc_stride_b=key_cache.stride(0), kc_stride_h=key_cache.stride(1),
        kc_stride_s=key_cache.stride(2),
        vc_stride_b=value_cache.stride(0), vc_stride_h=value_cache.stride(1),
        vc_stride_s=value_cache.stride(2),
        block_table_stride=block_table.stride(0),
        sel_st=selected.stride(0), sel_sh=selected.stride(1),
        ns_st=num_sel.stride(0), ns_sh=num_sel.stride(1),
        op_ptr=op, ml_ptr=ml,
        op_st=op.stride(0), op_sh=op.stride(1), op_ss=op.stride(2), op_sm=op.stride(3),
        ml_st=ml.stride(0), ml_sh=ml.stride(1), ml_ss=ml.stride(2), ml_se=ml.stride(3),
        num_warps=4, num_stages=2,
    )
    combine_kernel[(num_tokens, num_kv_heads)](
        output, op, ml,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        HEAD_SIZE=head_size, BLOCK_M=block_m, GROUP=group, NUM_SPLITS=num_splits,
        o_stride_t=output.stride(0), o_stride_h=output.stride(1),
        op_st=op.stride(0), op_sh=op.stride(1), op_ss=op.stride(2), op_sm=op.stride(3),
        ml_st=ml.stride(0), ml_sh=ml.stride(1), ml_ss=ml.stride(2), ml_se=ml.stride(3),
    )
    if return_selected:
        return output, selected, num_sel
    return output
