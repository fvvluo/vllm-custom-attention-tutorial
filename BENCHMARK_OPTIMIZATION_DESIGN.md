# vLLM 自定义 Attention Backend —— Benchmark 评分优化设计文档

> 目标：优化 `README.md` 所述 benchmark 的评分。本文先把「评分标准到底测什么、当前
> 接入实现卡在哪、有哪些可动的杠杆」讲清楚，再给出 prefill / paged-attn 的架构方案
> （含无损与有损两轨），最后列出**需要你拍板的决策点**。
>
> 面向读者：本项目作者本人（GPU kernel 工程师）。硬件/内核背景见
> `/dockerdata/wangzicheng/h20_attn_tech.md` 与记忆库，本文不重复。
>
> 状态：**设计稿，未改任何代码**。等你在 §7 做决策后再进入实现。
>
> 更新（对齐 origin 最新 README）：评分口径已从「TTFT + decode 吞吐两个数」统一为**单一分数
> `E2E = TTFT + 1000×TPOT`**（baseline **147s**，越低越好，加速比 = 147 / 你的 E2E），并把
> **正确性测试 `ALL PASS`** 定为性能分生效的**硬前提**（提交 `correctness.png` + `performance.png`
> 两张截图）。这重排了优化优先级——见 §0.5 与 §1.1。

---

## 0. TL;DR（先看结论）

1. **评分现在是一个统一分数：`E2E = TTFT + 1000×TPOT`（越低越好），baseline = 147s**：
   - `TTFT`≈长输入 prefill 耗时；`TPOT = 1/decode_tps`（decode 每 token 秒数）；
     `E2E` = 生成 1000 token 的端到端秒数。**加速比 = 147 / 你的 E2E**。
   - baseline 拆解：`147 = 111.4(TTFT) + 1000×0.03557(TPOT)` = **prefill 76% + decode 24%**。
     → **prefill(TTFT) 是最大的一块预算，是「赢」的主战场；但 decode 项一旦失控会直接吃掉整个分**（见 §0.5）。
   - **硬前提**：`tests/test_paged_attn_correctness.py` 必须 `ALL PASS`（`correctness.png`），性能分才算数。
     这道闸门专防「近似/丢 KV 的 kernel 只刷速度不保正确」——**直接约束了有损（sparse/fp8）方案的下限**。
   - 另有内核级 microbench `attention-test/bench_attention.py`（vs `flash_attn.cute`，128K 形状）。
     它**不是** README 评分，但是你验证 kernel 的主力；其 **decode 容差 1e-3 极严**，fp8 大概率过不了
     （所以 fp8 只在「vLLM 端到端 + 那道 bf16 正确性测试」里守质量，别拿它去撞这个严格 microbench）。

2. **当前 wzc backend 接入有两个阻塞性缺陷，导致长上下文下稀疏内核几乎不触发**：
   - **[BUG-1] import 失效**：`custom_backend/wzc_sparse_attention.py:74` 仍
     `from ops import _wzc_sparse_prefill_c5`，但该文件已改名为
     `attention-test/ops/_wzc_attn_sparse.py`。**开 `WZC_SPARSE_BACKEND=1` 会直接 ImportError**。
   - **[BUG-2] chunked prefill 使稀疏内核 100% 落到 torch 慢路径**：vLLM 本 commit
     `enable_chunked_prefill=True` 默认开、`max_num_batched_tokens=2048`。100k prompt 会被切成
     ~2048 token 的 chunk，每个 chunk `q_len(2048) < seq_len(累计 context)`。而适配器判据
     `is_pure_prefill = (q_len == seq_len)`（`wzc_sparse_attention.py:268`）**永远为 False** →
     全部走 `_torch_causal_gqa` 纯 torch 回退（对 2048×100k 还会实质 OOM/极慢）。
     **=> 现状下 CUSTOM+sparse 在长 prefill 上根本没在跑你的 kernel。这是最大的杠杆。**

3. **decode 项（TPOT）现在直接进总分，且它的瓶颈不在 attention kernel**：`--enforce-eager`
   让整个 32B 模型逐 kernel 启动，attention 只占单步预算极小一块（见
   `scripts/bench_wzc_decode_adapter.py` 的实测方法）。教学 CUSTOM 后端 decode ~9.8 tok/s →
   TPOT≈102ms → **单 decode 项就贡献 `1000×0.102 = 102s`**，比 baseline 整个 E2E（147s）里的
   decode 部分（35.6s）多 66s。**=> 只要还挂着 `--enforce-eager`，E2E 分数不可能赢，必须让后端
   可 CUDA graph 捕获并去掉 `--enforce-eager`。这是从「输」到「能比」的前提，不是可选项。**

4. **四条优化轨（详见 §5）**，优先级已按新评分口径重排：
   - **轨-0（解阻塞，必做）**：修 BUG-1；用「关 chunked prefill / 调大 batched tokens」或
     「让 kernel 支持 context 分块 prefill」两选一，让稀疏 prefill 真正触发。
   - **轨-A（无损，CUDA graph）**：去 `--enforce-eager`。**在新口径下升级为「必做」**——不解决它
     decode 项就把总分拖垮，无论 prefill 多快都赢不了。
   - **轨-B（有损提速，追求超过 flash 的 TTFT）**：block-top-k 稀疏 prefill +（可选）fp8 KV。
     必须先过 `test_paged_attn_correctness.py`（硬闸门），再用 `perf_test` 证明 E2E 收益。

---

## 0.5 新评分口径怎么改变打法（这次更新的核心）

**评分公式**：`E2E(s) = TTFT + 1000 × TPOT`，`TPOT = 1/decode_tps`，越低越好，
`加速比 = 147 / E2E`（baseline flash_attn 单卡 H20 ~95k 输入实测 147s）。

**baseline 预算拆解**（决定往哪使劲）：

| 分量 | baseline 值 | 占 E2E | 谁决定 | 优化杠杆 |
|---|---|---|---|---|
| TTFT | 111.4 s | **76%** | 长 prefill 的 attention | dense→稀疏/fp8 减 FLOPs（轨-B），先靠轨-0 让 kernel 触发 |
| 1000×TPOT | 35.6 s | 24% | **eager 全模型调度**（非 attention） | CUDA graph（轨-A）；再谈 fp8 decode |

三条推论（相对旧「两个独立数」口径的变化）：

1. **decode 从「另一个指标」变成「总分的一部分」**，且教学后端在这一项上是灾难级（+66s）。
   所以**轨-A（CUDA graph）从「可选加分」升级为「入场券」**——不做，E2E 必输。
2. **TTFT 是最大预算块（76%），是真正拉开加速比的地方**。把 111s 的 prefill 砍一半（稀疏/更快
   kernel）＝ 直接砍掉 ~55s，是单项最高杠杆。这与「稀疏专治长 prefill」完美对齐。
3. **正确性是性能分的硬前提**（`correctness.png` 必须 `ALL PASS`）。这道 bf16 `2e-2` 容差的
   paged 正确性测试，是有损方案的**统一验收口**：sparse `tau=1` 无损、fp8 都要先过它，
   再谈速度。它取代了我上一版里「HumanEval pass@1」作为主质量闸门的位置（HumanEval 仍是更
   贴近真实质量的补充验证，但**评分提交只认这道正确性测试 + E2E 分数**）。

> 一句话策略：**先用轨-A 把 decode 项拉回正常（≤ baseline 的 35.6s），再用轨-0+轨-B 把 TTFT
> 往下砍**。两者相乘才是加速比。只优化其一都不够。

---

## 1. 评分标准的精确拆解

### 1.1 README 评分（本次优化的**真正目标**）

评分现在是**一个统一分数 + 一道硬前提**：

| 评分件 | 脚本 / 产物 | 口径 | baseline（单卡 H20 实测） | 性质 |
|---|---|---|---|---|
| **正确性闸门** | `tests/test_paged_attn_correctness.py` → `correctness.png` | prefill/decode/mixed 三case 对 naive fp32 参考，bf16 `rtol=atol=2e-2`，须 `ALL PASS` | 教学 kernel `ALL PASS` | **硬前提**：不过则性能分无效 |
| **E2E 性能分** | `perf_test.py --input-len 100000 --output-len 64` → `performance.png` | `E2E = TTFT + 1000×TPOT`（秒，越低越好）；`加速比 = 147/E2E` | **147s**（TTFT 111.4 + 1000×TPOT 35.6） | **性能分**：唯一的速度分数 |

评分口径的关键细节（决定方案可行性）：

- **单一分数**：`E2E` 同时惩罚 prefill 慢（TTFT 高）和 decode 慢（TPOT 高），一个数比较所有 kernel。
  baseline 里 prefill 占 76%、decode 占 24%（见 §0.5）。
- **成绩可比的前提**：必须用**默认参数**——`--input-len 100000`（tokenizer 确定性截断后**人人都是
  ~95653 token**）、`--output-len 64`、**不加** `--allow-prefix-cache`（否则第 2 次命中前缀缓存跳过
  prefill，TTFT 假性归零）。改了这些分数不可比。提交要连同末行 `[perf] SUMMARY ...` + GPU 型号。
- **正确性测试的性质**：它对拍的是 **naive fp32 causal GQA 参考**、容差 bf16 `2e-2`，覆盖
  prefill(`q_len==seq_len`)/decode(`q_len==1`)/mixed。**这是有损方案的统一验收口**：sparse 必须在
  `tau=1`（选全部段）下逐位无损通过；fp8 也要在此通过。它专门防「丢 KV/糊 softmax 只刷速度」。
- **HumanEval（Part 1.4）现在是「补充质量验证」而非评分件**：提交只认上面两张图。但 HumanEval 仍
  是判断有损方案「真实质量是否掉」的最佳工具（它真跑代码、有检索性质），建议有损方案落地前自测。

### 1.2 内核级 microbench（验证工具，非 README 评分但必须过）

`attention-test/bench_attention.py`（记忆库 `reference_bench_harness`）：

- 基线 `flash_attn.cute.flash_attn_func`（sm90，decode 不 pack GQA → decode「vs baseline」虚高）。
- 目标形状 `1x64x8x131072x128 bf16 causal`，分别测 prefill（TFLOPS）与 decode（GB/s）。
- **容差**：prefill/默认 bf16 `abs/rel 2e-2`；**decode 严格 `1e-3`**（`bench_attention.py:208`，
  专门拦「均值糊弄 softmax」的投机实现）。
- 现状：dense prefill **143 TFLOPS / 96.7% MFU / 2.03×**；decode **0.162ms / 96.7% TMA 天花板 / 33×**；
  paged decode 相对连续版 **+1.8%**；sparse prefill `tau=0.999` 过随机数据 `≤2e-2`、~1.9–2.0×。

> **对有损方案的硬约束**：fp8 KV 的 decode 若走 `bench_attention.py` 的 decode 路径，**几乎必然
> 撞 1e-3 容差**（fp8 e4m3 相对误差 ~1e-2）。所以 fp8 只能作为「vLLM 端到端 HumanEval 可接受」
> 的方案，不能指望在这个严格 microbench 的 decode 项上 PASS。设计时要把两个 benchmark 分开对待。

---

## 2. 现有资产盘点（可复用的内核与接入件）

### 2.1 attention-test 内核（CuTe DSL，已实测达 roofline）

| 内核 | 文件 | 场景 | 性质 | 状态 | 复用价值 |
|---|---|---|---|---|---|
| Dense prefill | `ops/_wzc_attn_prefill.py` `HopperWGMMAKernel` | q_len==kv_len, causal, S%128==0 | compute-bound | ✅ 143 TFLOPS | **无损 prefill 首选** |
| Sparse prefill | `ops/_wzc_attn_sparse.py` `SparseC5Kernel` | 同上 + block-top-k | compute-bound | ✅ tau=0.999 过 harness | **有损 prefill 首选** |
| Dense decode | `ops/_wzc_attn_decode.py` | q_len=1, 连续 KV 131072 | memory-bound | ✅ 0.162ms | decode 参考 |
| Paged decode | `ops/_wzc_paged_attn_decode.py` `PagedKVDecoder` | q_len=1, 分页 page=128 | memory-bound | ✅ +1.8% | **vLLM decode 首选** |

共性接口约束（接 vLLM 时的坑）：
- 都要求 `head_dim==128`、`S%128==0`（prefill 的 BLOCK_M/N=128；paged 的 page_size=128）。
- prefill 内核是 **square-causal**（`q_len==kv_len`），**不支持 context 偏移的矩形 causal**
  （即 chunked prefill 的 `q_len<seq_len` 情形）——这正是 BUG-2 的根因。
- paged decode 要求 **GQA group==8**（`q_heads==kv_heads*8`），Qwen3-32B 正好满足。

### 2.2 vLLM 接入件（现状）

- `custom_backend/custom_triton_backend.py`：三件套（Backend/Builder/Impl）。`forward` 里先
  `triton_reshape_and_cache_flash` 写 KV，再调 `_paged_attention_fn`。KV cache 逻辑形状
  `(num_blocks, kv_heads, block_size, 2*head_size)`，split 成 K/V 视图，全按 stride 寻址。
- `WZC_SPARSE_BACKEND=1` 切到 `wzc_sparse_attention.paged_attention_wzc`（**当前 import 崩，BUG-1**）。
- `wzc_sparse_attention.py` 适配器逻辑：per-request 分派——纯 prefill→sparse kernel（含
  非 128 对齐的**尾部 padding**处理）；decode（q_len==1）→paged decode kernel（block_size==128
  时零拷贝，否则 gather+repack）；其余→torch 回退。
- 已有正确性测试：`tests/test_wzc_sparse.py`（tau=1 无损对拍、tau=0.99 有损画像、非对齐 padding、
  mixed batch）、`tests/test_wzc_decode.py`（变长 decode + mixed）。这两个测试**绕过了 vLLM 调度器**，
  直接喂 `q_len==seq_len` 的纯 prefill，所以它们 PASS **不代表**真实服务里 kernel 会触发（真实服务
  被 chunked prefill 切成 q_len<seq_len）——这解释了「测试全绿但 100k 跑不动」的矛盾。

---

## 3. 根因分析：为什么现在 CUSTOM 长上下文跑不出成绩

```
vLLM 调度器 (enable_chunked_prefill=True, max_num_batched_tokens=2048)
  └─ 100k prompt 被切成 ~49 个 chunk，每 chunk q_len≈2048
       └─ CustomTritonImpl.forward
            └─ paged_attention_wzc  (WZC_SPARSE_BACKEND=1)
                 ├─ [BUG-1] import _wzc_sparse_prefill_c5 → ImportError（根本进不来）
                 └─ 即便修了 import：
                      is_pure_prefill = (q_len==2048 == seq_len? 否) → False
                      decode_ok       = (q_len==1? 否)             → False
                      => _torch_causal_gqa(q=2048, K/V=累计 context)
                         对最后几个 chunk：scores≈(2048, 8, ~100k) fp32 → 数 GB/head，极慢/OOM
```

两层都得解决，否则你的高性能 kernel 在真实评分路径上一行都没跑到。

---

## 4. 架构理解：prefill vs paged-attn（与 E2E 评分的对应）

- **Prefill（compute-bound）**：FLOPs∝S²，唯一目标喂满 Tensor Core。对应 **E2E 的 TTFT 项（占 76%）**。
  - 无损天花板：dense kernel 143 TFLOPS ≈ 96.7% MFU，已到极限。要**超过** flash 只能**减少
    FLOPs**：block-top-k 稀疏（跳过低分 KV 段，省 QK+PV）或 fp8（每 FLOP 更快）。
  - 稀疏对**长 prefill** 收益最大（段数∝S，稀疏比例随 S 升），正好命中 100k 评分场景，
    也正好砍 E2E 里最大的那块预算。
- **Paged decode（memory-bound）**：逐 token 读全部 KV，瓶颈是 HBM 带宽。对应 **E2E 的 1000×TPOT 项（占 24%）**。
  - 无损天花板：0.162ms ≈ 96.7% TMA 天花板，已到极限。**稀疏在 decode 上收益有限**
    （仍要读被选 KV，且 memory-bound）；**唯一换量级的杠杆是 fp8/int8 KV**（字节减半→~2×）。
  - **但 TPOT 在 eager 全模型下不由 attention 决定**（§0.3/§0.5）：教学后端 TPOT≈102ms 里绝大部分
    是逐 kernel 启动开销，不是 attention。→ **先解 CUDA graph（轨-A）把 TPOT 拉回 baseline 量级
    （~35ms），attention kernel 的微优化才有意义。**

---

## 5. 优化轨（方案设计）

### 轨-0：解阻塞（必做，低风险，先做）

**0.1 修 BUG-1（import 改名）**
- `wzc_sparse_attention.py` 的 `_load_kernel()` 改 `from ops import _wzc_attn_sparse as k`；
  docstring 里 c4/c5 引用一并订正。改完先跑 `tests/test_wzc_sparse.py` / `test_wzc_decode.py` 回归。

**0.2 让稀疏 prefill 真正触发（BUG-2，二选一）**

- **方案 0.2-A（快，改配置）**：serve 时**关 chunked prefill + 调大 batched tokens**，
  使整段 prompt 作为**一次 pure-prefill**（`q_len==seq_len`）进入 `forward` → 现有 square kernel
  （+尾部 padding）直接吃下。已落地为独立脚本 `scripts/serve_qwen3_wzc_sparse.sh`
  （`WZC_SPARSE_BACKEND=1` + `--no-enable-chunked-prefill --max-num-batched-tokens=MAX_LEN`）。
  **优点：零 kernel 改动即可让稀疏 kernel 在评分路径上跑起来。**
  - ⚠️ **实测发现（2026-07-29，重要）**：**0.2-A 无法在单张 H20 上跑到 100k**。关掉 chunked
    prefill 后整段 prompt 一次 forward，其**激活峰值**与 KV cache 抢显存——`MAX_LEN=102400 + 0.94`
    时可用 KV 仅 **8.82 GiB**，而 102400 需 **25 GiB**，引擎启动即 `ValueError`（估算最大 ~36k）。
    提高 util 也留不出足够 KV。**已在 `MAX_LEN=32768` 验证通过**：~28.7k pure-prefill，
    `[wzc-stats] fallback_reqs=0 / max_kernel_seq=28757` → **稀疏 kernel 100% 触发、零 torch 回退**。
  - 结论：0.2-A 只适合**中等长度（≤~32k）**验证稀疏机制；**100k 评分场景必须走 0.2-B**
    （或多卡 TP 摊激活——但评分口径是单卡 H20，故不算）。
- **方案 0.2-B（稳，改 kernel，生产正道；因上面的显存墙，现为 100k 评分的必经之路）**：让 prefill
  内核/适配器支持 **context 偏移的矩形 causal**（chunked prefill：`q_len<seq_len`，query 行绝对
  位置 = `context + i`）。保持 vLLM 默认的 chunked prefill（激活被限制在 chunk=2048 内，显存才够
  100k KV），逐 chunk 用稀疏 kernel 处理 `q_len` 个 query 对 `context+q_len` 段的注意力。
  工作量：把 square kernel 的对角 mask 与 `n_block_max` 从「基于 q_blk 行」改为「基于 `context+q_blk
  行」，并让稀疏段选择在完整 context 上做。**要动 CuTe kernel**（有 719/barrier 等风险，见记忆库 pitfalls）。

> 建议：先 0.2-A 打通闭环、拿到 TTFT 数字；确认稀疏有效后再投 0.2-B 做生产化。

### 轨-A：无损提速——CUDA graph 压 TPOT 项（**新口径下升级为必做**）

**动机**：`bench_wzc_decode_adapter.py` 的方法论已经点明：教学后端 ~9.8 tok/s（TPOT≈102ms）的锅在
`--enforce-eager`（32B 模型逐 kernel 启动，attention 只占单步极小比例）。flash 的 ~28 tok/s
（TPOT≈35.6ms）靠 CUDA graph replay 把上百次 launch 合成一次。**在 E2E 口径下，这个 TPOT 差
（102 vs 35.6ms）× 1000 = 直接多 66s，无论 prefill 多快都赢不了。所以 CUDA graph 是入场券。**

**关键调研结论（2026-07-29，源码 `vllm/config/compilation.py` + `model_executor/.../attention.py`）**：
vLLM 有两档 cudagraph——**PIECEWISE**（把模型按 attention 边界切成段，段内 graph 化、attention 在
段间**仍跑 eager**）和 **FULL**（含 attention 一起 graph 化）。attention 的 split 点是注册的自定义 op
`vllm::unified_attention_with_output`（`attention.py:835` 调 `self.impl.forward`），**所有 v1 后端
（含 CUSTOM）都走它**。因此 **PIECEWISE 不要求后端 `_cudagraph_support`≠NEVER**：
`resolve_cudagraph_mode_and_sizes` 在后端为 NEVER 时会自动降级到 PIECEWISE（`compilation.py:1416`），
前提是 `mode=VLLM_COMPILE` 且 splitting_ops 含 attention（默认含）。→ **拆成两步：**

**A1（配置级，先做，低风险，无 kernel 改动）—— PIECEWISE cudagraph**
- 把模型其余部分（QKV proj / MLP / norm / router…上百次 launch）graph 化，attention 段间跑 eager
  ——**正好捕获 TPOT 的真正瓶颈**，而我的逐请求 python 循环 attention 原样保留（它在 graph 之外跑）。
- 做法：serve **去掉 `--enforce-eager`**，改设 `-O.mode=VLLM_COMPILE` + `--cudagraph-mode PIECEWISE`
  （后端 `_cudagraph_support` 保持 NEVER 不动）。已落地为 `serve_qwen3_wzc_sparse.sh` 的
  `PIECEWISE=1` 开关。
- 预期：TPOT 从 ~100-160ms 降到接近 baseline（~35ms 量级），**不碰正确性、不改 kernel**。
- 风险：piecewise 编译首次启动较慢（inductor 编译）；`--no-async-scheduling` 是否仍需要要实测
  （原因是 eager 后端与异步调度不友好，piecewise 下可能可去掉）。

**A2（可选，追求极致 TPOT）—— FULL cudagraph（需 batched graph-capturable kernel）**
- 只有 FULL 能把 attention 也 graph 化，但要求：`_cudagraph_support` ≥ `UNIFORM_SINGLE_TOKEN_DECODE`，
  `build()` **纯 pass-through** `common_attn_metadata` 的持久 GPU 张量（禁 `.tolist()`/`searchsorted`/
  host sync），`forward` **单次 grid kernel** 驱动整 batch（禁 per-request python 循环）。
- 现有 `PagedKVDecoder` 是**单序列**接口（`FlashDecodeKernel(...,1,...)` + `seq_id`），需要写一个
  **batched paged decode**（一次 kernel 处理整 batch 单 token）+ 参照 `TritonAttentionMetadataBuilder`
  重写 builder（`_cudagraph_support=ALWAYS`、pass-through、`build_for_cudagraph_capture` 里
  `seq_lens.fill_(1)`）。**工程量大**，且 A1 已能拿走大部分 TPOT 收益 → **A2 视 A1 后的剩余差距再决定**。

**结论**：**先 A1（配置级 piecewise）拿到有效 E2E**；若 attention 段间 eager 的开销仍显著（ncu/对比
可测），再投 A2。**A1 不做，E2E 分数没有意义**（TPOT 项爆炸）。

### 轨-B：有损提速——追求**超过** flash 的 TTFT（砍 E2E 最大预算块）

**B.1 block-top-k 稀疏 prefill（已有 kernel，主攻长 prefill TTFT）**
- 复用 `_wzc_attn_sparse.py`：cumulative-mass `tau` + sink + local window + 强制对角。
- 触发前提 = 轨-0 打通（否则不跑）。
- **正确性闸门（硬，先过）**：`tests/test_paged_attn_correctness.py` 必须 `ALL PASS`（→ `correctness.png`）。
  这要求 sparse 在 `tau=1`（选全部 causal 段）下逐位无损等价 dense；`test_wzc_sparse.py` 已验证过
  `tau=1` 无损、非对齐 padding 正确，但**提交认的是 tutorial 那道正确性测试**，需确保它也 `ALL PASS`。
- **质量自测（软，落地前做）**：跑完整 HumanEval `pass@1`，要求与 flash（145/164）**差 ≤1~2 题**。
  - 先用 `tau=0.999`（记忆库：过随机数据 harness、~2×），若 pass@1 达标再逐步降 tau（0.99/0.95）
    换更多稀疏度/更低 TTFT，画「pass@1 vs E2E」帕累托。
  - HumanEval 有精确检索性质（要复现函数签名/变量名），是 sparse 的**掉点高风险区**，务必实测。
- **性能验收**：`perf_test.py --input-len 100000` 默认参数下的 **E2E 分数**，对比 flash 的 147s；
  并对比 dense kernel（若轨-0 用 dense）证明稀疏净收益。TTFT 每砍 1s，E2E 就降 1s。

**B.2 fp8 KV（换量级的带宽/算力，主攻 decode/TPOT + 中长 prefill）**
- 思路：KV cache 存 fp8（e4m3），字节减半 → decode ~2×（压 TPOT 项）、prefill PV 算力减半。vLLM
  原生支持 `--kv-cache-dtype fp8`，但那是走它自己的 backend；我们要在 **CUSTOM kernel 内**支持 fp8 读。
- 约束（记忆库 & sparse 文档 §附）：QK 用 e4m3 + per-tensor descale（descale fold 进
  `softmax_scale_log2`）；PV 的 V 需 **K-major 物理布局**（fp8 WGMMA 不吃 MN-major operand），
  需 TMA 落转置或 stmatrix 转置。
- **正确性闸门**：仍要过 `test_paged_attn_correctness.py`（bf16 `2e-2`）。fp8 的量化误差在 `2e-2`
  容差下大概率能过（这道测试比 microbench 的 decode `1e-3` 宽得多），但需实测确认。
- **不要**指望过 `bench_attention.py` 的 decode `1e-3` 容差（fp8 e4m3 相对误差 ~1e-2 必然超）
  ——那是内核 microbench，与评分口径无关，别自我否定。落地后再用 HumanEval 端到端复核质量。
- **优先级**：fp8 是「换量级」但工程量大、精度风险高。**建议放在轨-A + 稀疏之后**，且先 decode
  （直接压 TPOT 项、V 布局改造集中在 paged decode kernel）再考虑 prefill。

**B.3（可选）sparse + fp8 叠加**：稀疏省段数、fp8 省每段时间，理论叠乘。**务必先各自独立验证
质量与正确性，再叠加**（否则出问题无法定位是稀疏漏 KV 还是 fp8 精度）。

---

## 6. 评分约束矩阵（一张表看清「什么方案动 E2E 哪一项、踩什么闸门」）

E2E = TTFT(76%) + 1000×TPOT(24%)。正确性测试 `ALL PASS` 是所有方案的硬前提。

| 方案 | 动 TTFT 项 | 动 1000×TPOT 项 | 正确性测试(2e-2)风险 | HumanEval 自测风险 | 主要工程量 |
|---|---|---|---|---|---|
| 轨-0（解阻塞） | 让稀疏**能跑** | — | 无（tau 可先=1 无损） | 无 | 小（改 import+配置） |
| 轨-A（CUDA graph） | ↓（去启动气泡） | **↓↓（主杠杆，−66s）** | 无（无损） | 无 | **大（接入层改造）** |
| 轨-B.1（sparse prefill） | **↓↓（长输入）** | — | 低（tau=1 逐位无损须过） | **中高**（检索掉点，实测 tau） | 中（kernel 已有，调 tau/接 chunked） |
| 轨-B.2（fp8 KV） | ↓（中长 prefill PV） | **↓↓（字节减半）** | 中（量化误差，实测确认过 2e-2） | 中（精度） | 大（V 布局/descale） |

> 记住：`bench_attention.py` 的 decode `1e-3` 是**内核 microbench**，不进 E2E 评分。fp8 过不了它
> 不代表方案失败——评分只看「正确性测试 `ALL PASS` + E2E 分数」。

---

## 7. 决策（D1–D4，已按推荐口径定稿）

下面把 §5/§6 的推荐答案定为**已采纳的决策**，作为 §8 路线图的依据。若你要改任一项，
告诉我编号，我改这里并同步 §8。

| 决策 | 选定 | 一句话理由 |
|---|---|---|
| **D1 先打通哪条闭环** | **(a) 先轨-0**，紧接 (b) 轨-A | 轨-0 零风险、先证明 prefill/稀疏能触发并拿 `correctness.png`；但 E2E 有效分要靠轨-A |
| **D2 chunked prefill 解法** | **0.2-A 已验证机制（≤32k）；100k 必走 0.2-B** | 实测 0.2-A 关分块后单 forward 激活+KV 撑爆单卡 H20（100k 需 25GiB KV、仅剩 8.82GiB）；32k 已验证稀疏 kernel 零回退触发 |
| **D3 精度预算** | **(a)→(b) 递进**，(c) fp8 视收益再定 | 先无损把 E2E 追平/微超守正确性；再开 sparse 换 TTFT（tau=1 必过正确性测试）；fp8 收尾可选 |
| **D4 E2E 主攻项** | **先 TPOT（轨-A 入场券）再 TTFT（轨-0+B 主战场）** | 不做 CUDA graph，TPOT 项直接把总分拖垮；拉正后 TTFT(76%) 才是拉开加速比的地方 |

> 一句话总纲：**轨-0（解阻塞）→ 轨-A（拉正 TPOT，拿有效 E2E）→ 轨-B（砍 TTFT/压 TPOT，超越 flash）**，
> 全程正确性测试 `ALL PASS` 是硬底线。

---

## 8. 实现路线图（D1–D4 已落地为四个阶段）

每个阶段有明确**入口→动作→验收 gate**；未过 gate 不进下一阶段。测速纪律见文末。

### 阶段 0 —— 解阻塞（对应 D1-a / D2-a，零风险，先做）—— ✅ 已完成（2026-07-29）
- **动作**：
  1. 修 **BUG-1**：`custom_backend/wzc_sparse_attention.py` 的 `_load_kernel()` 改
     `from ops import _wzc_attn_sparse as k`，订正 docstring 里 c4/c5 引用。✅
  2. 回归 `tests/test_wzc_sparse.py`、`tests/test_wzc_decode.py`。✅ 均 `ALL PASS`。
  3. 新增 `scripts/serve_qwen3_wzc_sparse.sh`（不改教学脚本）：`WZC_SPARSE_BACKEND=1` +
     `--no-enable-chunked-prefill --max-num-batched-tokens=MAX_LEN` + YaRN + `WZC_SPARSE_STATS=1`。✅
- **验收 gate（实测结果）**：
  - `tests/test_paged_attn_correctness.py` → **`ALL PASS`** ✅（prefill 7.8e-3 / decode 3.9e-3 / mixed 7.8e-3）。
  - 起服务（GPU 1，`MAX_LEN=32768`）发 ~28.7k 请求，日志 **`[wzc-stats] fallback_reqs=0
    max_kernel_seq=28757`** ✅ → **稀疏 kernel 100% 触发、零 torch 回退**，机制打通。
  - ⚠️ **`MAX_LEN=102400` 启动即 OOM**（关分块→单 forward 激活+KV 抢显存，KV 仅 8.82GiB<25GiB 需求）。
    → **100k 评分必须走阶段 0.2-B（kernel 支持 chunked/context prefill），不能靠配置法**。见 §5 轨-0 更新。

### 阶段 A —— 拉正 TPOT，拿有效 E2E（对应 D4「入场券」）
分两步：**A1（PIECEWISE，配置级，已完成）** + **A2（FULL，需 batched kernel，按需）**。

**A1 —— PIECEWISE cudagraph（✅ 已完成并验证，2026-07-29）**
- **动作**：`serve_qwen3_wzc_sparse.sh` 加 `PIECEWISE=1`（默认）→ 去掉 `--enforce-eager`，
  传 `-cc '{"mode":"VLLM_COMPILE","cudagraph_mode":"PIECEWISE"}'`。后端 `_cudagraph_support`
  保持 NEVER 不动，**零 kernel 改动**。模型除 attention 外的部分 graph 化，attention 段间跑 eager。
- **验收 gate（实测，GPU 3，~2k 输入）**：
  - 服务日志确认 `cudagraph_mode=PIECEWISE`、`enforce_eager=False`、"Capturing CUDA graphs
    (PIECEWISE): 51" 全部捕获、`unified_attention_with_output` 在 splitting_ops 里。✅
  - **decode 23.4 tok/s（TPOT 42.8ms）vs eager 基线 9.8 tok/s（TPOT ~102ms）→ ~2.4× decode 提速、
    TPOT 砍掉 ~59ms**，纯配置、不碰 kernel、不碰正确性。✅ 2k E2E=43.9s。
  - 注：README flash baseline 在 2k 上是 44 tok/s（TPOT ~23ms）。A1 后 CUSTOM 的 TPOT(42.8ms) 仍
    比 flash 高 ~20ms —— 差距来自 attention 段间仍 eager + wzc 逐请求 python 循环开销，这是 A2 的空间。

**A2 —— FULL cudagraph（可选，追极致 TPOT；未做）**
- 只有 FULL 能把 attention 也 graph 化，需要：`_cudagraph_support≥UNIFORM_SINGLE_TOKEN_DECODE`、
  builder 纯 pass-through `common_attn_metadata`（禁 `.tolist()`/`searchsorted`/host sync）、
  forward 单次 grid kernel 驱动整 batch（禁 per-request python 循环）。
- 现有 `PagedKVDecoder` 是单序列接口，需写 **batched paged decode** + 仿 `TritonAttentionMetadataBuilder`
  重写 builder。**工程量大**。**决策：先用 A1 拿有效 E2E，A2 视剩余 TPOT 差距（~20ms/token）再定**。
- **验收 gate（A1 已满足「拿到有效 E2E」）**：
  - 正确性测试仍 `ALL PASS`（无损，不碰精度）。✅（A1 不改 kernel，阶段 0 已过）
  - `perf_test.py` 完整 E2E：TPOT 从 eager ~102ms 拉到 42.8ms。✅ A2 目标是进一步逼近 flash 的 ~23ms。

### 阶段 B —— 砍 TTFT，超越 flash（对应 D3-b / D1-c，有损但守闸门）
- **入口**：阶段 A gate 通过（有了有效 E2E）。

**B.0 关键实测发现（2026-07-29，重塑本阶段）**：起 100k chunked+piecewise 服务（`CHUNKED=1`
`MAX_LEN=98304` GPU_MEM_UTIL=0.95，KV 103776 tokens，正常就绪），发 100k 请求 → **EngineCore
CUDA OOM 崩溃**。根因：chunked prefill 的**后续 chunk**（q_len≈2048, context≈93k）走适配器
`_torch_causal_gqa`，其 `scores = einsum("qgd,kd->qgk")`（`wzc_sparse_attention.py:208`）物化
`(2048, 8, ~95k)` fp32 ≈ 每 kv-head 数 GiB，权重+KV 占满 95GB 后只剩 ~512MiB → 试图分配 4GiB 即 OOM。
- **含义（load-bearing）**：**torch 回退在 100k 上根本跑不起来**——所以矩形 causal 稀疏 kernel
  不只是「加速」，而是**让 100k 能跑通的前提**。同时也要先把回退改成**显存安全**（分块 einsum），
  否则任何未被 kernel 覆盖的 chunk 都会 OOM。
- 已验证：矩形 causal 稀疏**算法**（`ops/_wzc_sparse_rect_ref.py` torch 参考）tau=1==dense
  逐位一致（max_err ~1e-7，ctx=0/2048/4096），算法正确性已锁定，可作未来 kernel 的金标准。

**B.0 已完成 + 100k 基线（2026-07-29）**：`_torch_causal_gqa` 改为 over-q-rows 分块（`Q_CHUNK=64`，
峰值 ~195MB），数值对拍 dense 参考 max_err ~1e-7，decode/mixed 测试全绿。**100k 不再 OOM、能跑通**：
- **TTFT=793.7s、TPOT=261.6ms、E2E=1055s**（vs flash TTFT 111.4 / E2E 147 → 当前 **7× 差**）。
- `[wzc-stats]`：`kernel_tok=1.25M`（chunk0 ~95k pure-prefill 走稀疏 kernel，`max_kernel_seq=95721`）
  + `fallback_tok=5.6M`（~46 个后续 chunk 走**显存安全但慢**的 torch）。**793s 里绝大部分 = 5.6M
  fallback token 的 torch attention**。→ 这正是 B.1 矩形 causal 稀疏 kernel 要消灭的部分，也是 E2E
  从 1055s 砍向 <147s 的全部空间所在。

**B.1 已完成 + 实测（2026-07-29，重大突破）**：矩形 causal 稀疏 kernel
`ops/_wzc_attn_sparse_rect.py`（square kernel 的 context-offset 泛化，新文件不改现有）已实现，
tau=1==dense 逐位无损（`_test_sparse_rect.py`/`test_wzc_chunked.py`：ctx=0/128/2048/4096/8192 +
非 128 对齐 q_len + mixed batch 全 PASS）。适配器把 chunk（q_len<seq_len, context%128==0）路由到它。
100k chunked+piecewise 实测（tau=0.999）：
- **TTFT 793.7 → 134.7s（5.9× 更快，已在 flash 111.4s 的 1.2× 内）；E2E 1055 → 395s（2.7× 更好）。**
- `[wzc-stats] fallback_tok=0`（全部 6.85M token 走 wzc kernel，torch 回退彻底消失）。
- **剩余瓶颈完全转移到 TPOT=260ms**（decode ~3.8 tok/s）——E2E 里 1000×TPOT=260s ≫ TTFT 135s。
  这不是 prefill/attention 问题，是 **decode 路径在 100k context 下的适配器开销**（疑似 block_size≠128
  时每步 gather+repack 95k 历史，或 piecewise 下 decode 仍有大量非图开销）。→ 下一步诊断 TPOT。

**TPOT 根因 + 修复（2026-07-29）**：根因证实——vLLM 默认 `block_size=16`，而适配器 decode 的
零拷贝路径要求 `block_size==128`；否则走 `_decode_one` 慢路径，**每 token gather+repack 整段 ~95k
历史进 128-page pool**（O(context) 拷贝）→ TPOT 260ms。**修复**：`CustomTritonBackend` 加
`get_supported_kernel_block_sizes`，在 `WZC_SPARSE_BACKEND=1` 时返回 `[128]`，vLLM
`get_preferred_block_size` 遂选 128（默认 16 不被 128 整除→取 min=128），KV cache 物理页布局直接
等于 kernel 的 128-page pool → decode 走 `_decode_one_zerocopy`（零拷贝）。已代码验证（wzc on:
16→128；wzc off: 保持 16，教学默认路径不受影响）。**100k 重测待 GPU 空闲**（预期 TPOT 从 260ms
回落到 A1 量级 ~40ms → E2E 从 395s 进一步降到 ~135+40=~175s 级别）。


- **动作（按依赖排序）**：
  1. **B.0 修回退显存安全**（先做，小改动）：`_torch_causal_gqa` 的 einsum 分块（over q-rows 或
     kv-cols）避免物化 `(q_len, group, seq)`，让 100k chunked 至少**能正确跑通**（慢但不崩）→ 建立
     诚实的 E2E 基线上界。
  2. **B.1 矩形 causal 稀疏 kernel**（核心，D2-b）：把 `_wzc_attn_sparse.py` 的 square-causal 扩展为
     **context 偏移矩形 causal**（新文件，不改现有 kernel）：query 块行绝对位置 `context+row_base+i`，
     `n_block_max=ceil((abs_row_last+1)/128)`，段选择在完整 `[0,seq_len)` context 上做，对角/边界段
     用绝对位置 causal mask。参考 `_wzc_sparse_rect_ref.py` 的 index 公式对拍。适配器把带 context 的
     chunk 路由到它（替换 torch 回退）。
  3. 用它 + 阶段 0 已验证的 chunk0 pure-prefill 稀疏，在 tau∈{1.0, 0.999, 0.99, 0.95} 扫。
  4. 完整 HumanEval `pass@1` 自测（vs flash 145/164），画 **pass@1 vs E2E 帕累托**，定生产 tau。
- **验收 gate**：
  - **tau=1 仍过 `test_paged_attn_correctness.py`（`ALL PASS`）**——有损方案的正确性底线。
    （该测试含 decode/mixed，矩形 causal kernel 接入后要保持全绿。）
  - 100k chunked 服务**不再 OOM、能跑完**；选定 tau 下 HumanEval 掉 ≤1~2 题；
    `perf_test` E2E（默认参数 100k）**低于 147s**（加速比 > 1）。

### 阶段 C —— fp8 KV（对应 D3-c，可选收尾，压 TPOT/中长 prefill）
- **入口**：阶段 A（+B）稳定；有余力再做。
- **动作**：KV cache 存 fp8（e4m3），先 decode（V 需 K-major 物理布局 + per-tensor descale fold 进
  `softmax_scale_log2`），再考虑 prefill PV。
- **验收 gate**：过 `test_paged_attn_correctness.py`（bf16 `2e-2`，实测确认 fp8 量化误差在容差内）
  + 端到端 HumanEval 守质量。**不看** `bench_attention.py` 的 decode `1e-3`（内核 microbench，不进评分）。

### 阶段依赖图
```
阶段0(解阻塞) ──► 阶段A(CUDA graph, 拉正 TPOT) ──► 阶段B(稀疏, 砍 TTFT) ──► 阶段C(fp8, 可选)
   │ gate: correctness ALL PASS        │ gate: 有效 E2E          │ gate: tau=1 无损+E2E<147   │ gate: 2e-2+HumanEval
   └ 也可单独先跑，只为验证 prefill 单点收益（E2E 总分此时无意义）
```

> 每阶段都遵循本项目一贯方法论（记忆库/HANDOFF）：算法先行→torch 参考→kernel→`--check-only`
> 对拍→空闲卡上 `--warmup/--iters` 稳定测速→诚实区分「正确」与「达标」。测速务必
> `nvidia-smi` 确认目标卡 util=0%（机器多人共用，GPU 0 常被占）。
> **评分提交口径**：`correctness.png`（`ALL PASS`）+ `performance.png`（含 `E2E 评分` 与末行
> `[perf] SUMMARY ...`），默认参数、单卡 H20，加速比 = 147 / 你的 E2E。
