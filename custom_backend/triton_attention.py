# SPDX-License-Identifier: Apache-2.0
"""
paged_attention_triton —— wzc kernel 接入点（遵循 README Part 2.5 契约）
==========================================================================

按教程契约：**接入自定义 kernel 只改本文件的 `paged_attention_triton` 函数体**，
其余（`custom_triton_backend.py` / `plugin.py` / `__init__.py` / `pyproject.toml`）
保持原样不动。`tests/test_paged_attn_correctness.py` 直接 import 本函数，因此这里
接入的 kernel 会被正确性测试覆盖（correctness.png 测的就是它）。

本实现把 vLLM 的分页/变长 batch 适配到 wzc 的 CuTe DSL kernel（见
`wzc_sparse_attention.py` 的实现与逐请求路由）：
  - 纯 prefill（q_len==seq_len）           -> block-top-k 稀疏 prefill kernel
  - chunked prefill（q_len<seq_len, ctx%128==0） -> 矩形 causal 稀疏 prefill kernel
  - decode（q_len==1）                     -> paged decode kernel（block_size==128
                                              时零拷贝；否则 gather+repack）
  - 其余（非对齐/不支持形状）              -> 精确 torch 回退（分块，显存安全）
KV cache 已由上游 `CustomTritonImpl.forward` 写好；本函数只负责读 cache 算 attention，
结果原地写进 `output` 并 return。签名/语义见 README Part 3.1。
"""

import torch

from .wzc_sparse_attention import paged_attention_wzc


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
    """分页注意力（causal, GQA, prefill+decode 通用）。委托给 wzc 适配器。

    - causal：每个 query token 只能看到不超过自身绝对位置的 KV。
    - GQA：num_heads 可以是 num_kv_heads 的整数倍，Q 头映射到共享的 KV 头。
    - output 原地写入并返回。
    """
    return paged_attention_wzc(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        output=output,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_seq_idx=token_seq_idx,
        block_table=block_table,
        scale=scale,
    )
