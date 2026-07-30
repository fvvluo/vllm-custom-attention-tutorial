# SPDX-License-Identifier: Apache-2.0
"""
自定义分页注意力接入点。

把请求分派到自研 kernel：
  - prefill / chunked-prefill：CuTe WGMMA prefill，失败回退 Triton prefill。
  - 纯 decode：Triton split-KV decode。
接口签名与语义为 causal / GQA / 分页寻址。
"""

import os

import torch

from .my_attention import paged_attention_triton as _triton_paged_attention


def _try_cute_prefill(query, key_cache, value_cache, output,
                      query_start_loc, seq_lens, block_table, scale):
    """尝试 CuTe WGMMA prefill，成功返回 True，否则 False（回退 Triton）。"""
    num_tokens, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    num_seqs = seq_lens.shape[0]

    # block-sparse 分支（PREFILL_BLOCK_SPARSE=1 启用）：sink 前缀 + 局部窗口。
    if os.environ.get("PREFILL_BLOCK_SPARSE", "0") == "1":
        try:
            from .my_attention import paged_sparse_prefill
            paged_sparse_prefill(
                query=query, key_cache=key_cache, value_cache=value_cache,
                output=output, query_start_loc=query_start_loc,
                seq_lens=seq_lens, block_table=block_table, scale=scale,
            )
            return True
        except Exception as e:
            import sys
            print(f"[CUSTOM] block-sparse fallback: {type(e).__name__}: {e}",
                  file=sys.stderr)

    # CuTe prefill 适用：单请求、GQA 64/8/128、bf16。
    use_cute = (
        num_seqs == 1
        and num_heads == 64 and num_kv_heads == 8 and head_size == 128
        and query.dtype == torch.bfloat16
    )
    if not use_cute:
        return False
    try:
        from .cute_prefill import cute_prefill_paged
        cute_prefill_paged(
            query=query, key_cache=key_cache, value_cache=value_cache,
            output=output, query_start_loc=query_start_loc,
            seq_lens=seq_lens, block_table=block_table, scale=scale,
        )
        return True
    except Exception as e:  # CuTe 失败即回退 Triton
        import sys
        print(f"[CUSTOM] CuTe prefill fallback to Triton: {type(e).__name__}: {e}",
              file=sys.stderr)
        return False


def paged_attention_triton(
    query: torch.Tensor,        # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size]
    output: torch.Tensor,       # [num_tokens, num_heads, head_size]（原地写入）
    query_start_loc: torch.Tensor,  # [num_seqs + 1] int32
    seq_lens: torch.Tensor,         # [num_seqs] int32
    token_seq_idx: torch.Tensor,    # [num_tokens] int32：每个 token 属于哪条请求
    block_table: torch.Tensor,      # [num_seqs, max_num_blocks] int32
    scale: float,
) -> torch.Tensor:
    """分页注意力（causal, GQA, prefill+decode 通用）总入口。

    - causal：每个 query token 只能看到不超过自身绝对位置的 KV。
    - GQA：num_heads 是 num_kv_heads 的整数倍，多个 Q 头共享一个 KV 头。
    - output 原地写入并返回。
    """
    num_tokens = query.shape[0]
    num_seqs = seq_lens.shape[0]
    is_decode_only = (num_tokens == num_seqs)  # 用 shape 判断，无 GPU 同步

    # prefill / chunked-prefill：优先 CuTe WGMMA，失败回退 Triton。
    if not is_decode_only:
        if _try_cute_prefill(query, key_cache, value_cache, output,
                             query_start_loc, seq_lens, block_table, scale):
            return output

    # decode，或 CuTe 不适用/失败 -> 自研 Triton kernel（含 prefill/decode 分派）。
    return _triton_paged_attention(
        query=query, key_cache=key_cache, value_cache=value_cache,
        output=output, query_start_loc=query_start_loc, seq_lens=seq_lens,
        token_seq_idx=token_seq_idx, block_table=block_table, scale=scale,
    )
