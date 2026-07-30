# FP8 分页 Attention 接入 vLLM —— 实现与测试结果

> **decode 性能瓶颈诊断与修复（split-KV / flash-decoding）见文末"decode 100k 瓶颈诊断"一节——
> 这是把 decode 从 3.8 → 24.8 tok/s（100k）的关键。**

把本教程的可替换 kernel（`custom_backend/triton_attention.py` 的 `paged_attention_triton`）
替换为 **FP8(e4m3) + 分页(block-table 间接寻址)** 实现，并接入 vLLM CUSTOM 后端，
在 Qwen3-32B 上验证正确性与性能。

## 做了什么
1. **FP8 attention kernel**（`custom_backend/triton_attention.py`）
   - 沿用本仓库 `fp8_attn/` 思路：Q/K 量化到 float8_e4m3，用 fp8 做 QK^T（H20 上 fp8
     吞吐约 bf16 2x），descale 恢复量级；PV 保持 fp32 累加（P 靠近 0，量化会放大误差）。
   - 移植本仓库 `paged_kv/` 的**分页读取**：按 `block_table` 把逻辑 KV 位置跳映射到物理块，
     沿 KV **分块**（BLOCK_N=block_size）读取，而非教学版逐 KV 位置串行标量读。
   - 保留 `use_fp8=False` 的 bf16 对照路径用于验证分页寻址正确性。
   - 接口/语义与教程完全一致，`CustomTritonImpl.forward` 无需改动即自动用上（默认 use_fp8=True）。

2. **接入 vLLM**：`pip install -e .` 注册 `vllm.general_plugins` 入口，`--attention-backend CUSTOM`
   选中本后端；vLLM 管理的分页 KV 按 `[num_blocks, num_kv_heads, block_size, head_size]`
   布局传入，kernel 按 stride 寻址。

## 正确性
### 单元测试（无需起服务）
- 教程原测试 `tests/test_paged_attn_correctness.py`（bf16 容差 2e-2）：fp8 路径逐元素 max
  绝对误差 ~0.16 会 FAIL —— 这是 e4m3 只有 3 位尾数的**固有精度特性**（随机高斯输入下
  softmax 近均匀、输出近 0，近零元素上量化噪声被放大），不是寻址/布局 bug。
- 新增 `tests/test_fp8_paged_attn.py`（fp8 贴合判据）：
  ```
  [prefill] bf16对照 max_abs=7.8e-03 (PASS) | fp8 max_abs=1.6e-01 mean_abs=8.3e-03 mean_rel=7.8e-02 (PASS)
  [decode]  bf16对照 max_abs=3.9e-03 (PASS) | fp8 max_abs=5.2e-02 mean_abs=6.0e-03 mean_rel=8.3e-02 (PASS)
  [mixed]   bf16对照 max_abs=7.8e-03 (PASS) | fp8 max_abs=1.6e-01 mean_abs=7.6e-03 mean_rel=7.9e-02 (PASS)
  ALL PASS
  ```
  - **bf16 对照路径过严格 2e-2** → 证明分页寻址/gather 完全正确；
  - **fp8 路径 mean-rel ≤ 2e-1**（与 `fp8_attn/test_fp8.py` 一致的 fp8 判据）→ 通过。
  运行：`PYTHONPATH=/dockerdata/landojiang/vllm_src:. python tests/test_fp8_paged_attn.py`

### 端到端（Qwen3-32B 服务）
- `GPU=4 PORT=8004 bash scripts/serve_qwen3_custom.sh` 起服务，日志确认
  `Using AttentionBackendEnum.CUSTOM backend.`；
- `smoke_test.py`（17+25=42）：**PASS，回答 42**。fp8 attention 若算错模型会输出乱码而非正确答案，
  故这是强端到端正确性信号。

## 性能（单张 H20，~1960 token 输入，output 32，中位数）
| 后端 | TTFT | decode 吞吐 |
| --- | --- | --- |
| `flash_attn`（baseline） | **0.98 s** | **43.9 tok/s** |
| `CUSTOM`（本 FP8 分页 kernel） | **5.85 s** | **26.2 tok/s** |
| 教程默认教学 kernel（README 实测） | 9.12 s | 9.8 tok/s |

- 本 FP8 分页 kernel 比教程默认教学 kernel **快约 1.6x（TTFT）/ 2.7x（decode）**——得益于按块读取 +
  fp8 QK，而非逐 token 逐位置串行标量。
- 仍慢于 `flash_attn`：因为 grid 仍是 `(num_tokens × num_heads)`、每 program 串行扫 KV，且
  `--enforce-eager` 无 CUDA graph。要追平需进一步做 KV 分块并行 + split-KV + 支持 CUDA graph
  （方向同任务一的高效 decode kernel）。100k 长输入下该朴素 grid 不实用，故只在小长度对比。

## kernel-level 微基准（tensor-core tl.dot 优化后，scripts/bench_paged_kernel.py）
把 kernel 从"逐 token 标量 reduce"重构为 **tensor-core `tl.dot`**（GQA group 当 M 维、grid
收紧为 (num_tokens × num_kv_heads)、KV tile 聚合 8 个物理块凑 N=128），绝对速度大幅提升：

| 场景 | 标量版 fp8 | tl.dot 版 fp8 | 加速 |
| --- | --- | --- | --- |
| prefill s=8192 | 2070 ms | **87 ms** | ~24x |
| decode ctx=32768 bs=8 | 11.8 ms | **0.82 ms** | ~14x |

但同 kernel 内 **fp8 仍略慢于 bf16（0.28–0.78x），fp8 未兑现加速**。原因是结构性的、非 bug：
- 本分页/decode 场景 **M 维 = GQA group = 8（pad 到 16）恒定很小**，矩阵乘是**延迟受限**而非
  吞吐受限；fp8 tensor-core 的 2x 吞吐只有在大 M、吞吐受限时才显现，M=16 时看不到。
- fp8 每个 KV tile 要付 **量化开销**（算 amax/缩放/cast fp32→fp8），且分页 K 是**非连续 gather
  载入**，这些固定成本盖过了 fp8 MMA 的收益。
- 这与 `fp8_attn/` 里 fp8 prefill ~2x 加速不矛盾：那是 FA3 CUDA 大 M-tile、KV cache 预量化为
  e4m3（不在 kernel 内反复量化）、吞吐受限的场景。

**结论**：在"单 token/小 M 的分页 decode + kernel 内动态量化"这一设定下，fp8 无法快过 bf16；
要让 fp8 真正提速需要：(a) 把 KV cache **预量化为 e4m3 常驻**（省去每步重量化、并省一半 HBM 带宽，
这才是 decode 的真正红利），(b) 增大 M（pack 更多 query 行/token）让矩阵乘吞吐受限。fp8 当前的
价值主要在 **显存/带宽**（KV 字节减半），而非这个小-M kernel 的算力。

## KV cache 预量化 e4m3（--kv-cache-dtype fp8）
已实现 KV cache 预量化路径：
- backend `supported_kv_cache_dtypes` 加 `fp8/fp8_e4m3`；`forward` 里 fp8 时把 cache `.view(fp8)`、
  用 `layer._k_scale/_v_scale` 让 `reshape_and_cache` 在写入时量化成 e4m3（**只量化一次**），
  descale 标量传给 kernel。
- kernel 新增 `KV_IS_FP8` 路径：**直接读 fp8 K/V（每字节减半、不重量化）**，QK fp8 tensor-core、
  V 反量化后 bf16 PV。
- 正确性：新增 tests/test_fp8_paged_attn.py 第 (3) 路"预量化 KV"全 PASS（mean_rel ~0.10）。

预量化 vs 动态量化 vs bf16（kernel-level，相对 bf16 倍数）：
| 场景 | fp8 动态量化KV | fp8 预量化KV常驻 |
| --- | --- | --- |
| prefill s=8192 | 0.61x | **0.80x** |
| decode ctx=32768 bs=8 | 0.78x | **0.86x** |
| decode ctx=131072 bs=1 | — | **0.84x** |

## 128k 长输入端到端对比（vLLM 服务，YaRN factor=4.0，~95.6k token 输入）
单张 H20、开 YaRN 扩到 128k+、`--max-model-len 102400`：

| 后端 | TTFT（prefill 95.6k） | decode 吞吐 |
| --- | --- | --- |
| `flash_attn`（FA3 baseline） | **111.8 s** | 28.8 tok/s |
| `CUSTOM` v1（per-token 逐 token 扫 KV） | 705.9 s | 3.8 tok/s |
| **`CUSTOM` v2（query-tiled prefill kernel）** | **200.8 s** | 3.8 tok/s |

### prefill 优化：query-tiled flash-attention（把 6.3x 压到 1.8x）
v1 根因：grid=(num_tokens×num_kv_heads)，**每 query token 一个 program 串行扫整条 KV**，
无 query 批处理、无三角裁剪。v2 重写 prefill（`_fp8_prefill_kernel`）：
- grid=(num_seqs, q_tiles, num_q_heads)，每 program 处理 **TILE_Q=64 个 query token**（大 M tile
  打满 tensor core），沿 KV 分块 flash-attention；
- **causal 三角裁剪**：本 tile 最远可见位置决定 KV 扫描上界，只扫到那里；
- 每个 KV block 只被本 q-tile 读一次（而非每 token 读一次）；
- 复用 3D 合并分页载入 + fp8 预量化 KV。

效果（kernel-level 单层 128k prefill）：**10.5s（v1）→ 2.56s（v2 fp8）= 4.1x**；
GFLOP/s 从 ~28 提到 **67-70**（prefill s≥8k）。prefill 预量化 fp8 甚至比 bf16 快 **1.33-1.53x**。
端到端 TTFT **705.9s → 200.8s（3.5x）**，与 flash_attn 差距 **6.3x → 1.8x**。

### 仍存在的 decode 差距（framework，非本 kernel）
decode 仍 3.8 tok/s（本次只优化 prefill，decode 走原 per-token kernel）。它比 flash_attn 慢主要
**不是 kernel 问题**（decode kernel 已 0.98x bf16），而是：CUSTOM 后端 `_cudagraph_support=NEVER`、
服务用 `--enforce-eager`，64 层逐步 eager launch 开销主导；flash_attn 用 CUDA graph 消掉了这块。
要追平 decode 需让后端支持 CUDA graph（去 --enforce-eager），属 framework 层改造。

### 结论
- **prefill 已显著逼近 flash_attn（1.8x 内）**，验证了 query-tiled + fp8 预量化 + 合并访存的组合有效。
- 进一步完全追平/超过 FA3 需：TMA 整块并行 + warp-specialized WGMMA + CUDA graph（生产级 FA3 的做法，
  即本仓库 paged_kv/ 128k decode 3.1 TB/s 的路线），超出教程 per-token 接口与 eager 约束。

## decode 优化：给 CUSTOM 后端加 CUDA graph 支持
给后端加了 **decode CUDA graph** 支持（framework 层改造）：
- `CustomTritonMetadataBuilder._cudagraph_support = UNIFORM_SINGLE_TOKEN_DECODE`、`reorder_batch_threshold=1`、
  `build()` 用 `common_attn_metadata.max_query_len`（host int）预算 `is_prefill`、加 `build_for_cudagraph_capture`。
- kernel 去 host-sync 以兼容捕获：`paged_attention_triton` 增 `is_prefill` 参数（替代 `qlens.max().item()`）；
  decode 路径 fp8 descale 改走**张量指针**（kernel 内 `tl.load`，替代 `.item()`）。prefill 仍走 eager。
- 起服务去掉 `--enforce-eager`，加 `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`。

实测结果：
- **CUDA graph 捕获成功**（"Capturing CUDA graphs (FULL): 100%|51/51"，占 0.37 GiB），启动正常，
  **smoke_test=42 PASS**（graph 捕获的 decode 路径数值正确）——framework 层改造本身完全跑通。
- **严谨对照（同一 GPU、同一 ~28.7k context、均含 warmup，repeat=2 稳定）**：
  | 模式 | decode 吞吐 | TTFT |
  | --- | --- | --- |
  | eager（--enforce-eager） | 10.4 tok/s | 27.5s |
  | CUDA graph（FULL_DECODE_ONLY） | 10.7 tok/s | 27.3s |
  → **二者基本持平（~3%）**。即在本场景（单请求、bs=1、长 context）下 CUDA graph 收益很小。
- **纠正之前的错误结论**：先前"eager 3.8 → graph 10.6，~2.8x"是**测量口径错误**——拿 eager@95k(3.8)
  和 graph@28.7k(10.7) 比了**不同 context**。同 context 对照后，graph 几乎无提升。
- **真正主导 decode 吞吐的是 context 长度**（attention 每步要扫的 KV 长度），而非逐层 launch 开销：
  bs=1 时每步只有一条序列，CPU 调度/launch 已能与 GPU 计算重叠，graph 省的那点开销被长 KV 扫描掩盖。
  CUDA graph 的收益要在**大 batch 小 context（launch 开销占比高）**时才明显——本长上下文单请求场景不是。
- 注：本机多租户竞争极严重（GPU 反复被抢 85-96GB），100k 全量端到端不稳定，故对照在空闲卡的
  ~28.7k context 上完成。

## 总体进展（逐轮优化，均为诚实实测）
| 阶段 | prefill TTFT vs FA3 | decode（同 ctx 对照）|
| --- | --- | --- |
| 初版 per-token 标量 kernel | 6.3x 慢 (128k TTFT 706s) | — |
| query-tiled prefill kernel | **1.8x 慢** (128k TTFT 200.8s) | — |
| decode: eager vs CUDA graph @28.7k | — | 10.4 vs **10.7 tok/s（持平）** |

- **prefill 已显著逼近 FA3（1.8x 内）** —— 这是本轮优化的主要成果，来自 query-tiled flash-attention
  + fp8 预量化 + 3D 合并访存。
- **decode**：CUDA graph 框架支持已跑通且正确，但在长上下文单请求场景收益甚微（与 eager 持平）；
  decode 与 FA3 的差距主要在 kernel 工艺（FA3 的 split-KV/TMA/warp-spec），而非 launch 开销。
  完全追平/超过 FA3 需 paged_kv/ 那种 TMA 整块并行 + split-KV 的 decode kernel（128k 实测 3.1 TB/s）。

## KV 访存布局优化：3D 合并载入（关键突破）
上一版每个 KV tile 用 **per-column 指针 gather**（`pb[:, None]`，[BLOCK_N,HEAD_SIZE] 逐列 scatter），
跨 8 个物理块非连续，打不满 HBM → fp8 半字节红利被 gather 开销吃掉。

优化：改成 **3D tile 载入 `[PAGES, BLOCK_SIZE, HEAD_SIZE]`**——block_table 的 gather 只发生在
**page 维**（每 tile 只查 PAGES_PER_TILE=8 个物理块号），而每个物理块内部的 16×128 在显存里**连续**
（布局 (num_blocks,kv_heads,block_size,head_size) 下同一 (block,kv_head) 连续），Triton 可对页内做
**合并/向量化访存**，载入后 reshape 成 [BLOCK_N,HEAD_SIZE]。正确性不变（reshape 保序，全 PASS）。

优化后 fp8 预量化 KV **相对 bf16**（clean 实测）：
| 场景 | 优化前(逐列gather) | 优化后(3D合并) |
| --- | --- | --- |
| prefill s=2048 | 0.79x | **1.06x** ✅ 快过 bf16 |
| prefill s=8192 | 0.80x | **1.03x** ✅ |
| decode ctx=32768 bs=8 | 0.86x | **0.98x**（基本追平）|
| decode ctx=8192 bs=8 | 0.76x | 0.85x |

**结论**：合并访存让预量化 fp8 的半字节带宽红利真正兑现——在 bandwidth-bound 的 prefill(≥2k) 上
**fp8 已快过 bf16**、长 context decode 基本追平。小 context decode(8k) 仍受 M=16 小 tile 的延迟开销
限制未追平。要进一步（追平/超过 flash_attn 量级）需 TMA 整块读 + split-KV + CUDA graph（同 paged_kv/
的 128k decode 3.1 TB/s 路线）。至此 fp8 的**带宽/显存红利已在本教程接口内落地**。

## 复现命令
```bash
cd /dockerdata/liuyi/fl3/flash-attention/vllm-custom-attention-tutorial
pip install -e .                      # 注册 CUSTOM 后端插件（一次性）
# 正确性
PYTHONPATH=/dockerdata/landojiang/vllm_src:. python tests/test_fp8_paged_attn.py
# 端到端（挑空闲卡）
GPU=4 PORT=8004 bash scripts/serve_qwen3_custom.sh          # 终端1
PYTHONPATH=/dockerdata/landojiang/vllm_src python scripts/smoke_test.py --port 8004 --model qwen3-32b  # 终端2
python scripts/perf_test.py --port 8004 --input-len 2048 --output-len 32
```

## decode 100k 瓶颈诊断与修复（split-KV / flash-decoding）
**问题**：接入后单 kernel 性能与 FA 相当，但 100k 端到端 decode 只有 flash_attn(~28 tok/s)的一半。

**诊断（先定位再改）**：kernel-level 逐项测量 bs=1 decode 发现根因——
原 decode grid=`(num_tokens, num_kv_heads)`，bs=1 时只有 `1×8=8 个 CTA`，在 78-SM 的 H20 上
**~90% 的 SM 闲置**，且每个 CTA **串行**扫完整条 100k KV：
| ctx | 原始 ms/层 | grid | KV 带宽 | attn-only tok/s 上限 |
| --- | --- | --- | --- | --- |
| 100k | 2.50ms | 8 CTA | **0.16 TB/s**(峰值~3.3) | **6.2** |
即只用到 ~5% 显存带宽 → 直接把 decode 压到 flash_attn 的一半以下。flash_attn 用 flash-decoding
(split-KV) 把长 KV 扫描摊到所有 SM，我们没有。这是根因，不是"缺某个特殊优化"。

**修复**：给 decode kernel 加 **split-KV**——grid 增加一维 `NUM_SPLITS`，每个 `(token,kv_head)`
的 KV 扫描切成 NUM_SPLITS 段并行（grid 8→8×NUM_SPLITS，填满 SM），各段写未归一化的部分
(acc,m,l) 到 workspace，再由 `_fp8_decode_combine_kernel` 跨段做在线-softmax 合并。
NUM_SPLITS 按 `~8×SM/(num_tokens·kv_heads)` 取 2 的幂（bs=1→~32-64）。

**效果（kernel-level，bs=1 100k）**：2.50ms→**0.213ms/层（11.7x）**，KV 带宽 0.16→**1.92 TB/s**，
attn-only 上限 6.2→**73 tok/s**。正确性 test_fp8_paged_attn 全 PASS（含 combine 跨段合并）。

**端到端（100k，空闲 GPU3，eager）**：decode **3.8 → 24.8 tok/s（6.5x）**，达到 flash_attn 28 的 **~89%**。

> 注：GPU0 上复测得 8.1 tok/s 是**被同卡其他租户 compute job 污染**（GPU0 全程 100% util、
> 常驻他人进程），prefill/decode 一起 2-3x 变慢，非本改动问题；有效数据以空闲卡(GPU3)为准。
> 剩余到 28 的差距主要是非 attention 层的 eager 开销 + 单 kernel 细节，属正常范围。

## 集成另一 agent 的稀疏优化（sparse decode 接入 vLLM 后端）—— 阶段1 完成
把 `attention-test/sparse_attn`（Quest query-aware 块稀疏）接进 vLLM CUSTOM 后端。

**桥接**：sparse 原吃连续 BHSD KV（4D TMA 顺序读）；vLLM 传分页 [num_blocks,kvh,16,d]+block_table。
新增 `custom_backend/sparse_paged.py`：
- **稀疏块=64=4×物理页(16)**，对齐 sparse 带宽甜点；选中的稀疏块展开成 4 个物理页
  `block_table[4*blk+i]`，用 3D 合并分页载入跳读（复用 triton_attention 范式）。
- Phase A 选块直接复用 `sparse_attn.select_blocks`（纯 torch/triton、布局无关）；摘要
  `build_paged_block_summary` 从分页池在 64 粒度上建 k_min/k_max。
- backend decode 分支加 env 开关 `CUSTOM_SPARSE=1`（+ CUSTOM_SPARSITY/CUSTOM_SPARSE_MIN_LEN）；
  纯 decode + bf16 KV + 序列≥min_len 时逐序列走稀疏，否则回退已验证的 dense split-KV。默认关。

**正确性**（`tests/test_sparse_paged.py`，全 PASS）：稀疏分页 vs 选中块 masked-dense，
乱序物理页 + S=8192/32768 + sparsity=1.0(=dense)：max_abs 2.3e-4 ~ 4.6e-4 → 分页寻址/页展开/
gather/split/combine 全算对。

**端到端**（GPU2，bf16，max-len 16384，CUSTOM_SPARSE=1 sparsity=0.25）：
- 短 prompt smoke(17+25) = **42 PASS**（<8192 走 dense）。
- **长上下文 needle 测试**：14,045-token prompt 埋入 secret「7391」，decode 走稀疏路径
  （只读 25% 块）→ 模型正确答出 **7391 PASS**。证明 Quest 上界选块在真实（结构化）注意力下
  **召回关键块**，稀疏不掉质量（与 sparse_attn README recall=1.0 一致；randn 才会 FAIL，是数学性质）。

**阶段2/3（未完成）**：sparse×verify 离线复现 + verify 接 vLLM v1 spec-decode（draft model +
推测验证）——是独立大工程，待续。

## 阶段2：verify / sparse_verify kernel 桥接到 paged 布局（完成）
新增 `custom_backend/verify_paged.py`：把 spec_decode 的投机验证 attention 桥接到 vLLM 分页布局。
- verify KV 分两部分：**history**（cache_len，在分页池，按 SPARSE_BLOCK=64=4页 展开 gather）+
  **K 个候选自身 KV**（本步新算、未入池，作连续 (kvh,K,d) 传入，单独一个 split 处理）。
- grid=(kv_heads, num_splits+1)：前 num_splits 个 split 处理 history 选中块，最后 1 个 split
  处理候选段（chain-causal `j<=t` 或 tree_mask）。q pack 成 group*K 行。
- `verify_paged`（dense history 全选）+ `sparse_verify_paged`（Quest 选 history top-k）。
- 候选段 tl.dot 收缩维 K 需 pad 到 ≥16。

**正确性**（`tests/test_verify_paged.py`，全 PASS，乱序物理页）：
- dense verify cache_len=4k/8k/16k, K=4/8：max_abs 2.3e-4 ~ 3.6e-4（vs history全可见+chain masked-dense）。
- sparse verify cache_len=16k K=8 sparsity=0.25：max_abs 3.8e-4（vs 选中块+chain）。
→ 分页 history gather + 候选段 + chain mask + split/combine 复合算对。

**踩坑**：combine kernel 的输出 stride 要传 `out.stride(1),out.stride(2)`（head/row），
误传 `out.stride(0),out.stride(1)`（batch/head）会导致结果全错（NaN/大误差）——调试时先逐段
验证（history split 的 m/l/acc 对、候选 split 对、手动 combine 对）才定位到是 combine 调用的 stride。

**阶段3（未完成）**：verify 接 vLLM v1 spec-decode 调度（draft model + 推测 token 验证 + q_len=K
候选批的 metadata），需研究 vLLM v1 spec-decode 路径能否用教程简化后端承载。kernel 已就绪。

## 阶段3 研究结论：verify 接 vLLM spec-decode（关键发现，简化了整合）
研读 vLLM v1 spec-decode 源码（/dockerdata/landojiang/vllm_src）得到**决定性结论**：

**1. vLLM 的 spec-decode verify = 标准 causal 变长 q_len=K attention，attention 里不需要 tree/chain mask。**
- spec batch 是 flattened 变长：每个验证序列 num_scheduled_tokens = draft_len+1，K 候选拼进同一
  flat token 流。`query_start_loc` 已编码 per-seq q_len=K，`seq_lens`=history+K，`max_query_len`=K
  （gpu_model_runner.py:2207/2412, backend.py:420-425,440）。
- `Impl.forward` 收到 query [num_tokens, heads, d]，num_tokens=Σq_len；**causal=True 隐式 bottom-right**
  （token i 看 history + 新 token ≤ i）。
- **接受/拒绝（tree/chain）全部在 attention 之后的 RejectionSampler 做**（gpu_model_runner.py:2819-2855,
  3657-3663），attention kernel 不需要知道 tree 结构。

**2. → 我现有的 query-tiled prefill kernel 已经正好覆盖 spec verify！**
`triton_attention.py::_fp8_prefill_kernel` 里：`context_len=seq_len-query_len`、`abs_pos=context_len+rows`、
causal `cols<=abs_pos`（bottom-right）——正是 spec-decode 的 q_len=K 长 history 场景。
`paged_attention_triton` 按 `is_prefill=max_query_len>1` 分流，spec batch(max_query_len=K>1)天然走它。
**故 verify 接 vLLM 无需新 kernel**；我单独写的 verify_paged.py（显式 chain/tree mask + 候选段分离）
是"树形 spec"才需要的更强形态，vLLM 的 ngram/chain spec 用不到（但已验证正确，备用）。

**3. 无需 draft model 的最简 proposer = n-gram**（ngram_proposer.py:172-174 load 是 no-op）。
启用：`--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'`。

**4. 无任何断言要求 spec-decode 必须用 FlashAttention 后端**——proposer/verify 路径 backend 无关，
自定义 CUSTOM 后端可用。可选优化：builder 设 `reorder_batch_threshold=num_spec+1`
（`_init_reorder_batch_threshold(1, supports_spec_as_decode=True)`）让 K-token decode 归到 decode 组；
`_cudagraph_support=UNIFORM_BATCH` 做 spec batch 的全图捕获。不设也**正确**，只是 K-token 走变长/prefill 路径。

**最简落地路径**：(a) 现有 prefill 路径已处理 q_len=K varlen causal ✓；(b) 起服务加
`--speculative-config method=ngram`；(c) 接受/bonus 采样全在后端之外。**理论上现有后端直接可跑 ngram
spec-decode**，待空闲 GPU 端到端验证（多租户竞争下未跑）。sparse×verify 的更优形态可用 verify_paged
在 backend 内替换 prefill 路径的 spec 分支（未来工作）。
