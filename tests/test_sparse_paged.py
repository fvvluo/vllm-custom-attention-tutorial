"""稀疏分页 decode kernel 单元测试（vLLM 布局），对比 dense 参考。

用 vLLM 的 KV cache 布局 [num_blocks, num_kv_heads, block_size, head_size]，
单序列 decode（num_tokens=1，q_len=1）。结构化输入下测 sparse vs full dense 逼近；
另测 sparse vs "同选中块 dense"（隔离 kernel 正确性）。
"""
import argparse
import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))  # 让 custom_backend 可导入

from custom_backend.sparse_paged import sparse_paged_decode  # noqa: E402
from custom_backend.triton_attention import _fp8_decode_combine_kernel  # noqa: E402


def build_cache(seq_len, num_kv_heads, block_size, head_size, dtype, dev, n_hot, q):
    """造 KV cache（结构化：少数块热点，与 q 对齐）。返回 kc, vc, block_table, k_full, v_full。"""
    num_blocks = (seq_len + block_size - 1) // block_size
    kc = torch.randn(num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=dev) * 0.3
    vc = torch.randn(num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=dev)
    group = q.shape[1] // num_kv_heads
    g = torch.Generator(device=dev).manual_seed(0)
    for h in range(num_kv_heads):
        hot = torch.randperm(num_blocks, generator=g, device=dev)[:n_hot]
        qdir = q[0, h * group, :]   # (head_size,)
        for blk in hot.tolist():
            kc[blk, h] = qdir.view(1, head_size) * 2.0 + torch.randn(
                block_size, head_size, dtype=dtype, device=dev, generator=g) * 0.1
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(1, num_blocks)
    return kc, vc, block_table, num_blocks


def dense_ref(q, kc, vc, seq_len, block_size, scale, sel=None):
    """dense：把 cache 摊平成 [seq_len, kv_heads, d] 做 full attention。sel!=None 只算选中块。"""
    num_blocks, num_kv_heads, _, d = kc.shape
    group = q.shape[1] // num_kv_heads
    k_full = kc.permute(1, 0, 2, 3).reshape(num_kv_heads, num_blocks * block_size, d)[:, :seq_len]
    v_full = vc.permute(1, 0, 2, 3).reshape(num_kv_heads, num_blocks * block_size, d)[:, :seq_len]
    out = torch.empty(1, q.shape[1], d, dtype=torch.float32, device=q.device)
    for h in range(num_kv_heads):
        mask = torch.ones(seq_len, dtype=torch.bool, device=q.device)
        if sel is not None:
            mask = torch.zeros(seq_len, dtype=torch.bool, device=q.device)
            for blk in sel[h]:
                s = blk * block_size
                mask[s:min(s + block_size, seq_len)] = True
        for g in range(group):
            qh = h * group + g
            qi = q[0, qh].float()
            sc = (k_full[h].float() @ qi) * scale
            sc = torch.where(mask, sc, torch.tensor(-float("inf"), device=q.device))
            w = torch.softmax(sc, dim=0)
            out[0, qh] = w @ v_full[h].float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    dtype = torch.bfloat16
    num_heads, num_kv_heads, d = 64, 8, 128
    block_size = 16
    scale = 1.0 / (d ** 0.5)

    print("=== 稀疏分页 decode kernel 正确性（vLLM 布局）===")
    for seq_len, n_hot, sparsity in [(4096, 4, 0.25), (8192, 6, 0.2)]:
        q = torch.randn(1, num_heads, d, dtype=dtype, device=dev)
        kc, vc, bt, num_blocks = build_cache(seq_len, num_kv_heads, block_size, d, dtype, dev, n_hot, q)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
        token_seq_idx = torch.zeros(1, dtype=torch.int32, device=dev)
        out = torch.empty(1, num_heads, d, dtype=dtype, device=dev)

        # 拿到 kernel 选的块（复算 selection 便于对比 masked dense）
        out, selected, num_sel = sparse_paged_decode(
            q, kc, vc, out, seq_lens, token_seq_idx, bt, scale,
            _fp8_decode_combine_kernel, sparsity=sparsity,
            n_sink=1, recent_win=4, return_selected=True)
        ref_full = dense_ref(q, kc, vc, seq_len, block_size, scale)
        e = (out.float() - ref_full).abs()
        mrel = (e / (ref_full.abs() + 1e-4)).mean().item()
        # 隔离 kernel 正确性：vs 同选中块 masked-dense（应逐元素一致）
        sel_list = [selected[0, h][selected[0, h] >= 0].tolist() for h in range(num_kv_heads)]
        ref_masked = dense_ref(q, kc, vc, seq_len, block_size, scale, sel=sel_list)
        ek = (out.float() - ref_masked).abs().max().item()
        print(f"  seq_len={seq_len:5d} sparsity={sparsity} budget≈{max(6,int(num_blocks*sparsity))}/{num_blocks} "
              f"| vs full: max_abs={e.max():.4f} mean_rel={mrel:.4f} "
              f"| kernel vs 同块dense: max_abs={ek:.5f} {'PASS' if ek < 2e-2 else 'FAIL'}")

    print("\n注：结构化热点输入下 mean_rel 应较小（top-k 召回热点块）；真实模型 attention 更稀疏。")


if __name__ == "__main__":
    main()
