# SPDX-License-Identifier: Apache-2.0
"""
简易 Triton attention kernel（教学示例）
============================================

这个文件是【学生需要替换的部分】。它实现了一个最小可用的、支持
**分页 KV cache（paged KV cache）** 的 attention，覆盖 prefill 与 decode。

设计目标：
  - 接口清晰：只要你的 kernel 满足 `paged_attention_triton(...)` 的输入/输出约定，
    就能直接替换本实现并接入 vLLM。
  - 正确性优先，不追求极致性能：每个 query token 一个 Triton program，
    在 kernel 内沿 KV 序列做在线 softmax（flash-attention 风格的数值稳定累加）。

KV cache 布局（与 vLLM TRITON_ATTN 后端一致）：
  kv_cache 逻辑形状 = (num_blocks, num_kv_heads, block_size, 2 * head_size)
  最后一维前 head_size 是 K，后 head_size 是 V。
  本模块在调用前已把它拆成 key_cache / value_cache 两个
  (num_blocks, num_kv_heads, block_size, head_size) 视图传入。

如何映射一个 token 到 cache 中的物理位置：
  对第 req 条请求的第 j 个（全局）位置：
    block_table[req, j // block_size] -> 物理 block 号 pb
    槽位 = j % block_size
  即该 (K,V) 存在 key_cache[pb, kv_head, 槽位, :]。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_kernel(
    # 输出： [num_tokens, num_heads, head_size]
    out_ptr,
    # query： [num_tokens, num_heads, head_size]
    q_ptr,
    # 分页 KV cache： [num_blocks, num_kv_heads, block_size, head_size]
    k_cache_ptr,
    v_cache_ptr,
    # 元数据
    query_start_loc_ptr,  # [num_seqs + 1] 每条请求 query 在 flatten 后的起始位置
    seq_lens_ptr,         # [num_seqs]     每条请求的总长度（context + 本次 query）
    token_seq_idx_ptr,    # [num_tokens]   每个 token 属于哪条请求（预计算好）
    block_table_ptr,      # [num_seqs, max_num_blocks] 逻辑块 -> 物理块
    # 形状 / 步长（标量）
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    max_num_blocks: tl.constexpr,
    # query 张量步长
    q_stride_t, q_stride_h,
    # out 张量步长
    o_stride_t, o_stride_h,
    # kv cache 步长
    kc_stride_b, kc_stride_h, kc_stride_s,
    vc_stride_b, vc_stride_h, vc_stride_s,
    block_table_stride,
):
    # grid = (num_tokens, num_heads)
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # ---- 1. 定位该 token 属于哪条请求、以及它在该请求内的绝对位置 ----
    # token->请求 的映射已在 host 端预计算好，直接读即可（避免 kernel 内查找）。
    seq_idx = tl.load(token_seq_idx_ptr + token_idx)

    q_start = tl.load(query_start_loc_ptr + seq_idx)
    seq_len = tl.load(seq_lens_ptr + seq_idx)         # 该请求总长度
    query_len = tl.load(query_start_loc_ptr + seq_idx + 1) - q_start
    # 该 token 在本请求内是第几个 query（0-based）
    idx_in_query = token_idx - q_start
    # 该 token 对应的绝对位置（causal 上界）：context 部分 + 该 query 偏移
    context_len = seq_len - query_len
    abs_pos = context_len + idx_in_query

    # ---- 2. 载入 query 向量 ----
    d_offs = tl.arange(0, HEAD_SIZE)
    q = tl.load(q_ptr + token_idx * q_stride_t + head_idx * q_stride_h + d_offs)
    q = q.to(tl.float32) * scale

    # GQA：多个 Q 头共享一个 KV 头
    kv_head_idx = head_idx // (num_heads // num_kv_heads)

    # ---- 3. 沿 KV 序列做在线 softmax（flash 风格）----
    m_i = -float("inf")     # running max
    l_i = 0.0               # running sum of exp
    acc = tl.zeros([HEAD_SIZE], dtype=tl.float32)

    # 只需注意到 abs_pos（含）为止（causal）
    kv_upper = abs_pos + 1
    for kv_pos in range(0, seq_len):
        active = kv_pos < kv_upper
        # 该 kv 位置的物理地址
        logical_block = kv_pos // BLOCK_SIZE
        slot = kv_pos % BLOCK_SIZE
        pb = tl.load(
            block_table_ptr + seq_idx * block_table_stride + logical_block,
            mask=logical_block < max_num_blocks,
            other=0,
        )
        k_off = pb * kc_stride_b + kv_head_idx * kc_stride_h + slot * kc_stride_s + d_offs
        v_off = pb * vc_stride_b + kv_head_idx * vc_stride_h + slot * vc_stride_s + d_offs
        k = tl.load(k_cache_ptr + k_off).to(tl.float32)
        v = tl.load(v_cache_ptr + v_off).to(tl.float32)

        qk = tl.sum(q * k, axis=0)
        qk = tl.where(active, qk, -float("inf"))

        m_new = tl.maximum(m_i, qk)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        l_i = l_i * alpha + p
        acc = acc * alpha + p * v
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + token_idx * o_stride_t + head_idx * o_stride_h + d_offs,
             out.to(out_ptr.dtype.element_ty))


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
    分页注意力（causal, GQA, prefill+decode 通用）。

    这是【接口约定】：学生把自己的 kernel 实现成同样的签名与语义即可替换。
    - causal：每个 query token 只能看到不超过自身绝对位置的 KV。
    - GQA：num_heads 可以是 num_kv_heads 的整数倍，Q 头映射到共享的 KV 头。
    - output 原地写入并返回。
    """
    from .cute_attention_kernel import paged_attention

    return paged_attention(
        query, key_cache, value_cache, output, query_start_loc, seq_lens,
        token_seq_idx, block_table, scale,
    )
