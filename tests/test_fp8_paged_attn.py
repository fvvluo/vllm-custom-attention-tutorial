# SPDX-License-Identifier: Apache-2.0
"""
FP8(e4m3) 分页注意力正确性检测
=================================

针对本仓库替换进来的 FP8 分页 kernel（custom_backend.triton_attention.paged_attention_triton）
做正确性验证。复用教程 test_paged_attn_correctness.py 的分页 KV 组装 + 朴素 fp32 参考。

为什么单独一份测试：教程默认 kernel 是 bf16 计算，容差 2e-2（逐元素 max）。而 FP8(e4m3)
只有 3 位尾数，是"精度换速度"的路径——在随机高斯输入下，softmax 近均匀、输出接近 0，
少数近零元素上的 fp8 量化噪声会把**逐元素 max 绝对误差**放大到 ~0.1 量级（这是 e4m3 的固有
特性，不是寻址/布局 bug）。因此对 fp8 用**贴合 fp8 的判据**：
  - bf16 对照路径（use_fp8=False）：仍用严格 max_abs<=2e-2，**证明分页寻址/gather 完全正确**；
  - fp8 路径（use_fp8=True）：用 **mean 相对误差<=2e-1**（与 fp8_attn/test_fp8.py 一致的 fp8 判据），
    同时打印 max_abs / mean_abs / max_rel / mean_rel 供人工核对。

运行：
  PYTHONPATH=/dockerdata/landojiang/vllm_src:. python tests/test_fp8_paged_attn.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 便于直接 import 同目录测试
from test_paged_attn_correctness import make_paged_kv_cache, naive_reference  # noqa: E402

from custom_backend.triton_attention import paged_attention_triton  # noqa: E402

BF16_ATOL = 2e-2          # bf16 对照路径：严格逐元素判据
FP8_MEAN_REL_TOL = 2e-1   # fp8 路径：mean 相对误差判据（fp8 换速度的标准判法）


def _build(seqs, nh, nkv, hs, bs, dev, dtype):
    torch.manual_seed(0)
    kv, qs = [], []
    for (sl, ql) in seqs:
        k = torch.randn(sl, nkv, hs, device=dev, dtype=dtype)
        v = torch.randn(sl, nkv, hs, device=dev, dtype=dtype)
        q = torch.randn(ql, nh, hs, device=dev, dtype=dtype)
        kv.append((k, v)); qs.append(q)
    scale = 1.0 / hs ** 0.5
    kc, vc, bt = make_paged_kv_cache(kv, nkv, hs, bs, dev, dtype)
    qsl = torch.zeros(len(seqs) + 1, device=dev, dtype=torch.int32)
    qsl[1:] = torch.tensor([q.shape[0] for q in qs], device=dev).cumsum(0)
    nt = int(qsl[-1])
    query = torch.cat(qs, 0)
    sl_t = torch.tensor([s for s, _ in seqs], device=dev, dtype=torch.int32)
    tsi = torch.searchsorted(qsl[1:], torch.arange(nt, device=dev, dtype=torch.int32),
                             right=True).to(torch.int32)
    ref = torch.cat(naive_reference(kv, qs, scale), 0)
    return query, kc, vc, qsl, sl_t, tsi, bt, scale, ref, nt


def run_case(name, seqs, nh, nkv, hs, bs, dev, dtype):
    query, kc, vc, qsl, sl_t, tsi, bt, scale, ref, nt = _build(
        seqs, nh, nkv, hs, bs, dev, dtype)

    # (1) bf16 对照：证明分页寻址/gather 完全正确
    out_bf = torch.empty(nt, nh, hs, device=dev, dtype=dtype)
    paged_attention_triton(query, kc, vc, out_bf, qsl, sl_t, tsi, bt, scale, use_fp8=False)
    e_bf = (out_bf.float() - ref).abs().max().item()
    ok_bf = e_bf <= BF16_ATOL

    # (2) fp8 路径（bf16 KV + kernel 内动态量化）：贴合 fp8 的 mean-rel 判据
    out_fp8 = torch.empty(nt, nh, hs, device=dev, dtype=dtype)
    paged_attention_triton(query, kc, vc, out_fp8, qsl, sl_t, tsi, bt, scale, use_fp8=True)
    e = (out_fp8.float() - ref).abs()
    mean_rel = (e / ref.abs().clamp_min(1e-2)).mean().item()
    ok_fp8 = mean_rel <= FP8_MEAN_REL_TOL

    # (3) 预量化 fp8 KV cache 路径：把分页 KV cache 量化成 e4m3 常驻，传 descale 标量。
    #     模拟 vLLM --kv-cache-dtype fp8：K/V 只在写入时量化一次，kernel 直接读 fp8。
    e4m3 = torch.float8_e4m3fn
    k_amax = kc.abs().amax().clamp_min(1e-12)
    v_amax = vc.abs().amax().clamp_min(1e-12)
    k_scale = (k_amax / 448.0).float()
    v_scale = (v_amax / 448.0).float()
    kc_fp8 = (kc.float() / k_scale).clamp(-448, 448).to(e4m3)
    vc_fp8 = (vc.float() / v_scale).clamp(-448, 448).to(e4m3)
    out_pq = torch.empty(nt, nh, hs, device=dev, dtype=dtype)
    paged_attention_triton(query, kc_fp8, vc_fp8, out_pq, qsl, sl_t, tsi, bt, scale,
                           k_descale=k_scale.view(1), v_descale=v_scale.view(1))
    e_pq = (out_pq.float() - ref).abs()
    mean_rel_pq = (e_pq / ref.abs().clamp_min(1e-2)).mean().item()
    ok_pq = mean_rel_pq <= FP8_MEAN_REL_TOL

    print(f"[{name}] bf16对照 max_abs={e_bf:.3e} ({'PASS' if ok_bf else 'FAIL'})  |  "
          f"fp8(动态) mean_rel={mean_rel:.3e} ({'PASS' if ok_fp8 else 'FAIL'})  |  "
          f"fp8(预量化KV) max_abs={e_pq.max().item():.3e} mean_rel={mean_rel_pq:.3e} "
          f"({'PASS' if ok_pq else 'FAIL'})")
    return ok_bf and ok_fp8 and ok_pq


def main():
    assert torch.cuda.is_available(), "需要 GPU"
    dev = "cuda"; dtype = torch.bfloat16
    nh, nkv, hs, bs = 64, 8, 128, 16
    all_ok = True
    all_ok &= run_case("prefill", [(37, 37), (16, 16)], nh, nkv, hs, bs, dev, dtype)
    all_ok &= run_case("decode", [(40, 1), (17, 1), (128, 1)], nh, nkv, hs, bs, dev, dtype)
    all_ok &= run_case("mixed", [(50, 50), (33, 1), (8, 8), (65, 1)], nh, nkv, hs, bs, dev, dtype)
    print("=" * 60)
    print("ALL PASS" if all_ok else "SOME FAILED")
    print("说明：bf16 对照路径过严格 2e-2 -> 分页寻址正确；fp8 路径按 mean-rel<=2e-1 判 -> "
          "e4m3 精度换速度，逐元素 max 绝对误差偏大属正常。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
