# SPDX-License-Identifier: Apache-2.0
"""
分页注意力（paged attention）正确性检测
==========================================

用途：把本教程的简易 Triton attention（custom_backend.triton_attention）作为
baseline，和一份【朴素 PyTorch 参考实现】逐元素比对，验证其在分页 KV cache
上的正确性。学生把自己的 kernel 实现成同样的接口后，跑本测试即可校验正确性。

覆盖两种典型场景：
  - prefill：一条请求一次性喂入多条 token（query_len == seq_len）。
  - decode ：请求已有 context，本次只喂 1 个 token（query_len == 1）。
以及两者混合的 batch。

运行：
  PYTHONPATH=/dockerdata/landojiang/vllm_src python tests/test_paged_attn_correctness.py
"""

import torch

from custom_backend.triton_attention import paged_attention_triton


def make_paged_kv_cache(
    kv_data,        # list[(K_i, V_i)]，K_i/V_i: [seq_len_i, num_kv_heads, head_size]
    num_kv_heads,
    head_size,
    block_size,
    device,
    dtype,
):
    """把每条请求的连续 K/V 摆放进分页 KV cache，返回 cache + block_table + slot_mapping。"""
    # 统计需要多少物理块
    blocks_per_seq = [
        (len(k) + block_size - 1) // block_size for (k, v) in kv_data
    ]
    total_blocks = sum(blocks_per_seq) + 2  # 多留 2 块，模拟非满块
    max_num_blocks = max(blocks_per_seq)

    key_cache = torch.zeros(
        (total_blocks, num_kv_heads, block_size, head_size), device=device, dtype=dtype
    )
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros(
        (len(kv_data), max_num_blocks), device=device, dtype=torch.int32
    )

    next_block = 0
    for si, (k, v) in enumerate(kv_data):
        seq_len = len(k)
        for j in range(seq_len):
            lb = j // block_size
            slot = j % block_size
            if slot == 0:
                # 分配一个新物理块给该逻辑块
                block_table[si, lb] = next_block
                next_block += 1
            pb = block_table[si, lb]
            key_cache[pb, :, slot, :] = k[j]
            value_cache[pb, :, slot, :] = v[j]
    return key_cache, value_cache, block_table


def naive_reference(kv_data, queries, scale):
    """朴素 causal GQA attention 参考实现（float32），返回 list[out_i]。"""
    outs = []
    for (k, v), q in zip(kv_data, queries):
        # q: [q_len, num_heads, hs]; k/v: [seq_len, num_kv_heads, hs]
        q_len, num_heads, hs = q.shape
        seq_len, num_kv_heads, _ = k.shape
        context_len = seq_len - q_len
        group = num_heads // num_kv_heads
        out = torch.empty((q_len, num_heads, hs), dtype=torch.float32, device=q.device)
        qf = q.float()
        kf = k.float()
        vf = v.float()
        for h in range(num_heads):
            kvh = h // group
            for i in range(q_len):
                abs_pos = context_len + i
                scores = (qf[i, h] * scale) @ kf[: abs_pos + 1, kvh].T  # [abs_pos+1]
                w = torch.softmax(scores, dim=-1)
                out[i, h] = w @ vf[: abs_pos + 1, kvh]
        outs.append(out)
    return outs


def run_case(name, seqs, num_heads, num_kv_heads, head_size, block_size,
             device, dtype, rtol, atol):
    """seqs: list[(seq_len, query_len)]"""
    torch.manual_seed(0)
    kv_data = []
    queries = []
    for (seq_len, q_len) in seqs:
        k = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
        v = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
        q = torch.randn(q_len, num_heads, head_size, device=device, dtype=dtype)
        kv_data.append((k, v))
        queries.append(q)

    scale = 1.0 / (head_size ** 0.5)

    # 组装分页 KV cache
    key_cache, value_cache, block_table = make_paged_kv_cache(
        kv_data, num_kv_heads, head_size, block_size, device, dtype
    )

    # flatten queries -> [num_tokens, num_heads, hs]，构造元数据
    q_lens = [q.shape[0] for q in queries]
    seq_lens = torch.tensor([s for (s, _) in seqs], device=device, dtype=torch.int32)
    query_start_loc = torch.zeros(len(seqs) + 1, device=device, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(q_lens, device=device).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    query = torch.cat(queries, dim=0)
    token_seq_idx = torch.searchsorted(
        query_start_loc[1:], torch.arange(num_tokens, device=device, dtype=torch.int32),
        right=True,
    ).to(torch.int32)

    output = torch.empty(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    paged_attention_triton(
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

    ref = torch.cat(naive_reference(kv_data, queries, scale), dim=0)
    got = output.float()
    max_err = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
    print(f"[{name}] max_abs_err={max_err:.4e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    assert torch.cuda.is_available(), "需要 GPU"
    device = "cuda"
    dtype = torch.bfloat16
    num_heads, num_kv_heads, head_size, block_size = 64, 8, 128, 16
    rtol, atol = 2e-2, 2e-2  # bf16 容差

    all_ok = True
    # 1) 纯 prefill：整条 query
    all_ok &= run_case("prefill", [(37, 37), (16, 16)], num_heads, num_kv_heads,
                        head_size, block_size, device, dtype, rtol, atol)
    # 2) 纯 decode：已有 context，本次 1 token
    all_ok &= run_case("decode", [(40, 1), (17, 1), (128, 1)], num_heads, num_kv_heads,
                       head_size, block_size, device, dtype, rtol, atol)
    # 3) prefill + decode 混合
    all_ok &= run_case("mixed", [(50, 50), (33, 1), (8, 8), (65, 1)], num_heads,
                       num_kv_heads, head_size, block_size, device, dtype, rtol, atol)

    print("=" * 50)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
