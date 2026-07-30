# SPDX-License-Identifier: Apache-2.0
"""
分页版投机验证（verify）正确性验证。

对齐 spec_decode/test_verify.py 思路：paged verify 输出 vs 朴素 masked-dense
（history 全可见 + K 候选 chain-causal）。分页 history 用**乱序物理页**摆放，证明
页展开 gather + 候选段 + chain mask + split/combine 复合算对。

运行：PYTHONPATH=/dockerdata/landojiang/vllm_src:. python tests/test_verify_paged.py --gpu N
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_backend.verify_paged import (  # noqa: E402
    verify_paged, sparse_verify_paged, PAGE,
)


def ref_verify(q, khist, vhist, cand_k, cand_v, cache_len, scale):
    """朴素 masked-dense 参考。q:(qh,K,d) khist/vhist:(cache_len,kvh,d) cand:(K,kvh,d)."""
    qh, K, d = q.shape
    kvh = khist.shape[1]
    group = qh // kvh
    out = torch.empty(qh, K, d, dtype=torch.float32, device=q.device)
    for h in range(qh):
        kvi = h // group
        kfull = torch.cat([khist[:, kvi], cand_k[:, kvi]], 0).float()   # (cache_len+K, d)
        vfull = torch.cat([vhist[:, kvi], cand_v[:, kvi]], 0).float()
        for t in range(K):
            s = (q[h, t].float() * scale) @ kfull.T                     # (cache_len+K,)
            # history 全可见；候选 j 可见 iff j<=t（chain）
            vis = torch.zeros(cache_len + K, dtype=torch.bool, device=q.device)
            vis[:cache_len] = True
            vis[cache_len:cache_len + t + 1] = True
            s = s.masked_fill(~vis, float("-inf"))
            w = torch.softmax(s, -1)
            out[h, t] = w @ vfull
    return out


def build_paged(khist_bhsd, dev):
    """(cache_len, kvh, d) -> 分页池 [npages, kvh, PAGE, d] + block_table(乱序)."""
    cache_len, kvh, d = khist_bhsd.shape
    npages = (cache_len + PAGE - 1) // PAGE
    total = npages + 3
    kc = torch.zeros(total, kvh, PAGE, d, device=dev, dtype=khist_bhsd.dtype)
    perm = list(range(total)); random.Random(2).shuffle(perm)
    bt = torch.tensor(perm[:npages], device=dev, dtype=torch.int32)
    return kc, bt, npages, perm


def test_dense(gpu, cache_len=8192, K=8, sparse_block=64):
    dev = f"cuda:{gpu}"
    torch.manual_seed(0)
    qh, kvh, d = 64, 8, 128
    q = torch.randn(qh, K, d, device=dev, dtype=torch.bfloat16)
    khist = torch.randn(cache_len, kvh, d, device=dev, dtype=torch.bfloat16)
    vhist = torch.randn(cache_len, kvh, d, device=dev, dtype=torch.bfloat16)
    cand_k = torch.randn(K, kvh, d, device=dev, dtype=torch.bfloat16)
    cand_v = torch.randn(K, kvh, d, device=dev, dtype=torch.bfloat16)
    scale = d ** -0.5

    # 摆 history 进乱序分页池
    kc, bt, npages, perm = build_paged(khist, dev)
    vc = torch.zeros_like(kc)
    for lp in range(npages):
        pb = perm[lp]; s0 = lp * PAGE; s1 = min(s0 + PAGE, cache_len)
        kc[pb, :, :s1 - s0] = khist[s0:s1].transpose(0, 1)
        vc[pb, :, :s1 - s0] = vhist[s0:s1].transpose(0, 1)

    out = verify_paged(q.view(1, qh, K, d), kc, vc, bt,
                       cand_k.transpose(0, 1).contiguous(),   # (kvh, K, d)
                       cand_v.transpose(0, 1).contiguous(),
                       cache_len, sm_scale=scale, sparse_block=sparse_block)[0]  # (qh,K,d)
    ref = ref_verify(q, khist, vhist, cand_k, cand_v, cache_len, scale)
    err = (out.float() - ref).abs().max().item()
    ok = err < 2e-2
    print(f"[dense verify] cache_len={cache_len} K={K}: max_abs={err:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_sparse_verify(gpu, cache_len=16384, K=8, sparse_block=64, sparsity=0.25):
    """sparse history 选块 + 分页 verify（sparse×verify 的 paged 版）vs 选中块+chain 参考。"""
    from custom_backend.verify_paged import sparse_verify_paged, select_blocks
    from custom_backend.sparse_paged import build_paged_block_summary
    dev = f"cuda:{gpu}"
    torch.manual_seed(0)
    qh, kvh, d = 64, 8, 128
    group = qh // kvh
    scale = d ** -0.5
    q = torch.randn(qh, K, d, device=dev, dtype=torch.bfloat16)
    khist = torch.randn(cache_len, kvh, d, device=dev, dtype=torch.bfloat16)
    vhist = torch.randn(cache_len, kvh, d, device=dev, dtype=torch.bfloat16)
    cand_k = torch.randn(K, kvh, d, device=dev, dtype=torch.bfloat16)
    cand_v = torch.randn(K, kvh, d, device=dev, dtype=torch.bfloat16)
    kc, bt, npages, perm = build_paged(khist, dev)
    vc = torch.zeros_like(kc)
    for lp in range(npages):
        pb = perm[lp]; s0 = lp * PAGE; s1 = min(s0 + PAGE, cache_len)
        kc[pb, :, :s1 - s0] = khist[s0:s1].transpose(0, 1)
        vc[pb, :, :s1 - s0] = vhist[s0:s1].transpose(0, 1)
    kmin, kmax, _ = build_paged_block_summary(kc, bt, cache_len, kvh, d, sparse_block, sparse_block // PAGE)
    out = sparse_verify_paged(q.view(1, qh, K, d), kc, vc, bt,
                              cand_k.transpose(0, 1).contiguous(), cand_v.transpose(0, 1).contiguous(),
                              cache_len, kmin, kmax, sm_scale=scale,
                              sparse_block=sparse_block, sparsity=sparsity)[0]
    # 参考：只在选中 history 块 + chain 候选
    qsel = q.view(1, qh, K, d)[:, :, -1:, :].contiguous()
    nhb = (cache_len + sparse_block - 1) // sparse_block
    budget = min(max(1 + 8 + 1, int(round(nhb * sparsity))), nhb)
    sel, ns = select_blocks(qsel, kmin, kmax, nhb, budget, n_sink=1, recent_window_blocks=8)
    sel, ns = sel[0], ns[0]
    o = torch.empty(qh, K, d, dtype=torch.float32, device=dev)
    for h in range(qh):
        kvi = h // group
        idx = []
        for b in sel[kvi][:ns[kvi]].tolist():
            s0 = b * sparse_block; s1 = min(s0 + sparse_block, cache_len); idx += list(range(s0, s1))
        idx = sorted(set(idx)); nh = len(idx)
        kh = torch.cat([khist[idx, kvi], cand_k[:, kvi]], 0).float()
        vh = torch.cat([vhist[idx, kvi], cand_v[:, kvi]], 0).float()
        for t in range(K):
            s = (q[h, t].float() * scale) @ kh.T
            vis = torch.zeros(nh + K, dtype=torch.bool, device=dev); vis[:nh] = True; vis[nh:nh + t + 1] = True
            o[h, t] = torch.softmax(s.masked_fill(~vis, float("-inf")), -1) @ vh
    err = (out.float() - o).abs().max().item()
    ok = err < 2e-2
    print(f"[sparse verify] cache_len={cache_len} K={K} sparsity={sparsity}: "
          f"max_abs(vs 选中块+chain)={err:.3e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(args.gpu)
    ok = True
    ok &= test_dense(args.gpu, cache_len=8192, K=8)
    ok &= test_dense(args.gpu, cache_len=4096, K=4)
    ok &= test_dense(args.gpu, cache_len=16384, K=8)
    ok &= test_sparse_verify(args.gpu, cache_len=16384, K=8, sparsity=0.25)
    print("=" * 40); print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
