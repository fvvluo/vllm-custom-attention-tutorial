# V2 — Direct Paged-KV warp-MMA Decode Prototype (Liu Xiaochen)

**V2 is an independent decode-only paged-KV prototype. It is NOT registered as
the active CUSTOM backend.** It is a standalone correctness/perf study that reads
the vLLM-style paged KV-cache directly (via `block_table`), using a warp-MMA
Tensor-Core kernel. Sync-load only (no cp.async / PDL yet — that is V3).

## Support domain (fixed target)
Decode-only: `q_len == 1` per sequence, `num_tokens == num_seqs` (packed in
request order). BF16, `D=128`, `Hq=64`, `Hkv=8` (GQA group 8), `block_size=16`,
SM90/H20, CUDA, contiguous or strided cache (stride-aware). Anything else raises
`TypeError`/`ValueError` (no silent fallback, no gather, no K/V copy).

## Paged addressing (in-kernel, no gather)
For logical token position `pos` of sequence `seq`, kv_head `kvh`, dim `d`:
```
logical_block  = pos // block_size          # block_size = 16
slot           = pos %  block_size
physical_block = block_table[seq, logical_block]
K = key_cache[physical_block, kvh, slot, d]  # read by real strides
V = value_cache[physical_block, kvh, slot, d]
```
A 64-token tile (`n_block=64`) spans 4 logical 16-token blocks, which may map to
non-contiguous physical blocks; SMEM rows 0..63 are filled in **logical** order.
Out-of-range tokens (`pos >= seq_len`) are zero-filled in SMEM and masked to
`-inf` in the score tile.

## MMA mapping
`warp.MmaF16BF16Op(bf16, Float32, (16,8,16))` — m16n8k16 HMMA, FP32 acc.
- Q tile `[16,128]`: 8 q_heads of this kv_head in rows 0..7, rows 8..15 zero pad.
- K tile `[64,128]`, score `[16,64]` FP32, P `[16,64]` BF16, V `[64,128]`, O `[16,128]` FP32.
- Both QK and PV run on Tensor Core; no SIMT dot/FMA main path.

## Split-KV + combine
Grid `[num_splits_max, kv_heads, num_seqs]`, one warp (32 threads) per
(split, kv_head, seq). `split_size_tokens` fixed; `valid_splits = ceil(seq_len/split_size)`,
`num_splits_max = ceil(max_seq_len/split_size)`; splits beyond a seq's valid range
write neutral (`lse=-inf`, `partial_o=0`). Each CTA processes `split_size/64`
tiles (e.g. split=256 → 4 tiles/CTA).
- Stage-1 partial: `partial_o [num_seqs,Hq,num_splits,D] FP32`, `partial_lse [num_seqs,Hq,num_splits] FP32`.
- Combine: grid `[Hq, num_seqs]`, 128 threads; `LSE-weighted` merge; writes BF16
  to `output[query_start_loc[seq], head, d]` (packed query token). Pure GPU.

## Correctness (rtol=atol=2e-2 tutorial tol; strict 5e-3-vs-ref diagnostic)
ALL PASS vs tutorial PyTorch reference AND tutorial Triton:
- basic seq_len ∈ {1,15,16,17,63,64,65,128,512} × seed {0,1,2026}
- tutorial decode [40,17,128]; irregular [1,17,65,257] / [127,128,129,1023]
- shuffled block_table; padded-leading-stride cache; explicit scale (default & 0.05)
- big: [40,512,8192], [131072], [8192,32768,131072] (shuffled) — max_abs ≤ 2.3e-4 vs ref.

## Microbenchmark (H20, warmup=10/iters=100/rounds=5, CUDA-event, same input;
tutorial Triton is the paged decode baseline)

### 128K single, split sweep (best = split 256)
| split | num_splits | Stage-1 ms | combine ms | end-to-end ms | speedup vs Triton | unique-KV(K+V) GB/s | workspace |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **256** | **512** | 3.195 | 0.011 | **3.221** | **24.3x** | 167 | 16.9 MB |
| 512 | 256 | 3.513 | 0.007 | 3.530 | 22.2x | 152 | 8.5 MB |
| 1024 | 128 | 9.215 | 0.004 | 9.227 | 18.2x | 58 | 4.2 MB |

**Combine is ~0.3% of total; Stage-1 dominates (~99%).**

### earlier single-seq (cleaner GPU)
| seq_len | Triton ms | V2 e2e ms | speedup |
|---:|---:|---:|---:|
| 128 | 0.050 | 0.099 | 0.50x (fixed-overhead bound) |
| 512 | 0.192 | 0.171 | 1.12x |
| 8192 | 3.72 | 0.189 | 19.7x |
| 32768 | 19.6 | 1.07 | 18.3x |

### multi-sequence (seq_len=8192, split=256)
| num_seqs | end-to-end ms | speedup vs Triton | tokens/s |
|---:|---:|---:|---:|
| 1 | 0.189 | 19.7x | ~5300 |
| 4 | 2.340 | 4.19x | 1709 |
| 16 | 7.124 | 1.59x | 2246 |
V2 latency grows with num_seqs (more heavy warps, sync-load bound); Triton
parallelizes over more query tokens, so V2's relative lead shrinks.

### block_table identity vs randomized (same cache size)
- 8192: identity 0.418 ms vs shuffled 0.448 ms → **~7% penalty**.
- 32768: numbers contended on a shared GPU (identity run anomalously high);
  treated as indicative only. **We only claim block_table lookup is NOT a
  dominant bottleneck; we do not assert it is negligible without cleaner profiling.**

## Non-strict comparison to continuous B5/B7 (different repo/framework)
Under the unified K+V-bytes GB/s convention (128K K+V = 512 MiB):
- continuous B5 (sync) ≈ **1.81 TB/s**, B7 (cp.async+PDL) ≈ **2.88 TB/s**.
- V2 paged (sync) 128K ≈ **167 GB/s** — ~10-17x lower bandwidth utilization.
This is expected: V2 is a **synchronous-load** paged prototype. It far outpaces
the tutorial Triton (per-token serial-KV, ~24x at 128K), but is nowhere near the
HBM roofline because loads are not overlapped with compute.

## Bottleneck
Stage-1 synchronous paged K/V load — GB/s ≤167 (128K), far below H20 ~4 TB/s.
Combine negligible. block_table lookup not dominant (indicative). Latency-bound
on load, exactly the B5→B6 lesson.

## Next: V3 cp.async
Add cp.async double-buffered paged K/V staging (analogous to old B5→B6, ~1.5x
there) to overlap load with warp-MMA. V2 (this doc) is the frozen sync baseline.

## Limitations
Decode-only; fixed 64/8/128/16 shape; sync load; no cp.async/PDL; not wired to
CUSTOM backend; multi-seq scaling weak.
