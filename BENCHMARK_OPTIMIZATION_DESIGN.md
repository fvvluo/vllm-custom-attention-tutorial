# vLLM 自定义 Attention Backend —— Benchmark 评分优化设计文档

> 目标：优化 `README.md` 所述 benchmark 的评分。本文先把「评分标准到底测什么、当前
> 接入实现卡在哪、有哪些可动的杠杆」讲清楚，再给出 prefill / paged-attn 的架构方案
> （含无损与有损两轨），最后列出**需要你拍板的决策点**。
>
> 面向读者：本项目作者本人（GPU kernel 工程师）。硬件/内核背景见
> `/dockerdata/wangzicheng/h20_attn_tech.md` 与记忆库，本文不重复。
>
> 状态：**设计稿，未改任何代码**。等你在 §7 做决策后再进入实现。

---

## 0. TL;DR（先看结论）

1. **README 里其实有两套评分标准，方向相反**：
   - **质量分**：Part 1.4 的 HumanEval `pass@1`（flash_attn 基线 **145/164 = 88.41%**，
     教学 Triton 146/164）。这是**正确性/精度**闸门，有损方案（sparse / fp8）绝不能把它打崩。
   - **性能分**：Part 4 的 `perf_test.py`，测 **TTFT（≈长输入 prefill 耗时）** 和
     **decode 吞吐 (tok/s)**。这是**速度**闸门，教学 kernel 在 2k 输入上就比 flash 慢 ~9×。
   - 另有一套**内核级** microbench：`attention-test/bench_attention.py`（vs `flash_attn.cute`
     基线，128K 形状，看 TFLOPS/GB·s + 严格容差）。它不是 README 的评分，但它是你验证
     kernel 正确性+性能的主力，且 **decode 容差 1e-3 极严**（fp8 大概率过不了）。

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

3. **decode 吞吐（9.8 vs flash 44 tok/s）的瓶颈不在 attention kernel**：`--enforce-eager`
   让整个 32B 模型逐 kernel 启动，attention 只占单步预算极小一块（见
   `scripts/bench_wzc_decode_adapter.py` 的实测方法）。**要提 decode 分，得让后端可被
   CUDA graph 捕获并去掉 `--enforce-eager`，而不是继续抠 attention。**

4. **三条优化轨（详见 §5）**，建议优先级：
   - **轨-0（解阻塞，必做）**：修 BUG-1；用「关 chunked prefill / 调大 batched tokens」或
     「让 kernel 支持 context 分块 prefill」两选一，让稀疏 prefill 真正触发。
   - **轨-A（无损提速）**：让后端可 CUDA-graph 捕获（去 `--enforce-eager`）——这是把 TTFT 和
     decode tps 都拉到接近 flash 的关键；attention 用现成的 dense/decode 无损 kernel。
   - **轨-B（有损提速，追求超过 flash）**：block-top-k 稀疏 prefill + （可选）fp8 KV。
     必须用 HumanEval `pass@1` 守住质量，用 `perf_test` 证明 TTFT 收益。

---

## 1. 两套评分标准的精确拆解

### 1.1 README 评分（本次优化的**真正目标**）

| 评分项 | 脚本 | 测什么 | 现状（单卡 H20 实测，README 记录） | 闸门性质 |
|---|---|---|---|---|
| **HumanEval pass@1** | `humaneval_generate.py`+`humaneval_evaluate.py` | 164 题贪心补全通过率 | flash_attn **88.41%**；教学 Triton **89.02%** | **质量下限**：有损方案必须≈flash |
| **TTFT** | `perf_test.py --input-len 100000` | 首 token 延迟≈100k prefill 耗时 | flash ~110s@95k；教学 kernel 跑不动 100k | **性能**：越低越好 |
| **decode 吞吐** | `perf_test.py`（同上，看 decode_tps） | 首 token 后逐 token 速度 | flash 27.8 tok/s@95k；2k 输入 flash 44 / CUSTOM 9.8 | **性能**：越高越好 |

关键约束（决定方案可行性）：

- **质量校验会真的跑代码**（HumanEval 沙箱执行）。sparse 漏掉 retrieval 相关 KV、fp8 精度不足
  都可能让 `pass@1` 掉几题；README 明说「若后端把 KV 写错，分数会大幅崩塌」——所以有损方案的
  验收标准是 **pass@1 与 flash 差 ≤1~2 题（≈1%）**，不能只看「能出通顺文本」。
- **perf_test 默认加唯一前缀绕过 prefix cache**，测的是真实 prefill。这对稀疏 prefill 有利
  （稀疏正是省长 prefill）。
- **TTFT 在长输入下由 attention 主导**，但 **decode tps 在 eager 模式下由全模型调度主导**
  （§0.3）。两者的优化杠杆不同，别混。

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

## 4. 架构理解：prefill vs paged-attn（与 benchmark 的对应）

- **Prefill（compute-bound）**：FLOPs∝S²，唯一目标喂满 Tensor Core。对应 **TTFT** 评分。
  - 无损天花板：dense kernel 143 TFLOPS ≈ 96.7% MFU，已到极限。要**超过** flash 只能**减少
    FLOPs**：block-top-k 稀疏（跳过低分 KV 段，省 QK+PV）或 fp8（每 FLOP 更快）。
  - 稀疏对**长 prefill** 收益最大（段数∝S，稀疏比例随 S 升），正好命中 100k 评分场景。
- **Paged decode（memory-bound）**：逐 token 读全部 KV，瓶颈是 HBM 带宽。对应 **decode tps** 评分。
  - 无损天花板：0.162ms ≈ 96.7% TMA 天花板，已到极限。**稀疏在 decode 上收益有限**
    （仍要读被选 KV，且 memory-bound）；**唯一换量级的杠杆是 fp8/int8 KV**（字节减半→~2×）。
  - **但 decode tps 在 eager 全模型下不由 attention 决定**（§0.3）→ 先解 CUDA graph。

---

## 5. 三条优化轨（方案设计）

### 轨-0：解阻塞（必做，低风险，先做）

**0.1 修 BUG-1（import 改名）**
- `wzc_sparse_attention.py` 的 `_load_kernel()` 改 `from ops import _wzc_attn_sparse as k`；
  docstring 里 c4/c5 引用一并订正。改完先跑 `tests/test_wzc_sparse.py` / `test_wzc_decode.py` 回归。

**0.2 让稀疏 prefill 真正触发（BUG-2，二选一）**

- **方案 0.2-A（快，改配置，推荐先做）**：serve 时**关 chunked prefill + 调大 batched tokens**，
  使 100k prompt 作为**一次 pure-prefill**（`q_len==seq_len`）进入 `forward` → 现有 square kernel
  （+尾部 padding）直接吃下。改 `serve_qwen3_custom.sh` 加：
  ```
  --no-enable-chunked-prefill --max-num-batched-tokens 102400
  ```
  代价：单次 forward 处理整段 100k，激活/中间显存更大（需与 KV cache 抢显存，可能要降
  `--max-model-len` 或提 `--gpu-memory-utilization`）。**优点：零 kernel 改动即可让稀疏 kernel 在
  评分路径上跑起来**，是验证「稀疏是否真能降 TTFT」的最快闭环。
- **方案 0.2-B（稳，改 kernel，生产正道）**：让 prefill 内核/适配器支持 **context 偏移的矩形
  causal**（chunked prefill：`q_len<seq_len`，query 行绝对位置 = `context + i`）。这是 vLLM 长
  上下文的标准工作方式，不必调大 batched tokens。工作量：把 square kernel 的对角 mask 与
  `n_block_max` 从「基于 q_blk 行」改为「基于 `context+q_blk 行`」，并让稀疏段选择在 context 上
  做。**收益更通用**，但要动 CuTe kernel（有 719/barrier 等风险，见记忆库 pitfalls）。

> 建议：先 0.2-A 打通闭环、拿到 TTFT 数字；确认稀疏有效后再投 0.2-B 做生产化。

### 轨-A：无损提速——让后端可 CUDA graph 捕获（提 decode tps + 降固定开销）

**动机**：`bench_wzc_decode_adapter.py` 的方法论已经点明：9.8 tok/s 的锅在 `--enforce-eager`
（32B 模型逐 kernel 启动，attention 只占单步极小比例）。flash 的 44 tok/s 靠 CUDA graph replay
把上百次 launch 合成一次。

**设计**：
- 让 `CustomTritonBackend` 支持 CUDA graph：`_cudagraph_support` 从 `NEVER` 提升；`forward` 里
  去掉依赖运行期 `.tolist()`/python 循环的路径（graph 捕获要求 shape/控制流稳定）。
- decode 走 **batched paged decode**（一次 kernel 处理整个 batch 的单 token），而非 per-request
  python 循环——现有 `PagedKVDecoder` 是单请求接口，需要一个 batched wrapper 或 persistent kernel
  的多请求分派。
- serve 去掉 `--enforce-eager`（及 `--no-async-scheduling`）。

**风险/成本**：这是**接入层的较大改造**（graph 捕获对 metadata builder、buffer 复用有强约束）。
但它是把 README 性能分（尤其 decode tps）拉到 flash 量级的**唯一正道**，无损、不碰精度闸门。

### 轨-B：有损提速——追求**超过** flash 的 TTFT

**B.1 block-top-k 稀疏 prefill（已有 kernel，主攻长 prefill TTFT）**
- 复用 `_wzc_attn_sparse.py`：cumulative-mass `tau` + sink + local window + 强制对角。
- 触发前提 = 轨-0 打通（否则不跑）。
- **质量验收（关键）**：跑完整 HumanEval `pass@1`，要求与 flash（145/164）**差 ≤1~2 题**。
  - 先用 `tau=0.999`（记忆库：过随机数据 harness、~2×），若 pass@1 达标再逐步降 tau（0.99/0.95）
    换更多稀疏度/更低 TTFT，画「pass@1 vs TTFT」帕累托。
  - HumanEval 有精确检索性质（要复现函数签名/变量名），是 sparse 的**掉点高风险区**，务必实测。
- **性能验收**：`perf_test.py --input-len 100000` 的 TTFT，对比 flash 的 ~110s；并对比 dense
  kernel（若轨-0 用 dense）证明稀疏净收益。

**B.2 fp8 KV（换量级的带宽/算力，主攻 decode + 中长 prefill）**
- 思路：KV cache 存 fp8（e4m3），字节减半 → decode ~2×、prefill PV 算力减半。vLLM 原生支持
  `--kv-cache-dtype fp8`，但那是走它自己的 backend；我们要在 **CUSTOM kernel 内**支持 fp8 读。
- 约束（记忆库 & sparse 文档 §附）：QK 用 e4m3 + per-tensor descale（descale fold 进
  `softmax_scale_log2`）；PV 的 V 需 **K-major 物理布局**（fp8 WGMMA 不吃 MN-major operand），
  需 TMA 落转置或 stmatrix 转置。
- **质量验收**：只在 **vLLM 端到端 HumanEval** 上验（pass@1 差 ≤1~2 题）。**不要**指望过
  `bench_attention.py` 的 decode `1e-3` 容差（fp8 必然超）——这是两个 benchmark，别自我否定。
- **优先级**：fp8 是「换量级」但工程量大、精度风险高。**建议放在 sparse 之后**，且先 decode
  （收益最直接、V 布局改造集中在 paged decode kernel）再考虑 prefill。

**B.3（可选）sparse + fp8 叠加**：稀疏省段数、fp8 省每段时间，理论叠乘。**务必先各自独立验证
质量与正确性，再叠加**（否则出问题无法定位是稀疏漏 KV 还是 fp8 精度）。

---

## 6. 评分约束矩阵（一张表看清「什么方案会踩什么闸门」）

| 方案 | 降 TTFT | 提 decode tps | HumanEval pass@1 风险 | bench_attention 容差 | 主要工程量 |
|---|---|---|---|---|---|
| 轨-0（解阻塞） | 让稀疏**能跑** | — | 无（tau 可先=1 无损） | prefill 2e-2 OK | 小（改 import+配置） |
| 轨-A（CUDA graph） | ↓（去启动气泡） | **↑↑（主杠杆）** | 无（无损） | 不涉及 | **大（接入层改造）** |
| 轨-B.1（sparse prefill） | **↓↓（长输入）** | — | **中高**（检索掉点，需实测 tau） | prefill 2e-2 需过 | 中（kernel 已有，调 tau/接 chunked） |
| 轨-B.2（fp8 KV decode） | ↓（中长） | **↑↑（字节减半）** | 中（精度） | **decode 1e-3 必挂**（只走端到端） | 大（V 布局/descale） |

---

## 7. 需要你决策的点（请拍板，我据此进入实现）

**D1 — 先打通哪条闭环？**（决定第一步动手对象）
- (a) 只做轨-0（修 import + 关 chunked prefill），先让**无损/稀疏 prefill 在 100k 评分路径上真正跑起来**，
  拿到第一组 TTFT/HumanEval 数字。← 推荐，最快见效、风险最低。
- (b) 直接上轨-A（CUDA graph 改造），优先把 decode tps 拉到 flash 量级。← 收益大但工程重。
- (c) 直接上轨-B.1（稀疏调 tau + HumanEval 质量画像）。← 依赖轨-0 先打通。

**D2 — chunked prefill 怎么处理？**（BUG-2 的解法）
- (a) 配置法：serve 关 chunked prefill + 调大 `--max-num-batched-tokens`（快，先验证稀疏收益）。
- (b) kernel 法：扩展 prefill kernel 支持 context 矩形 causal（慢，生产正道）。
- （可先 a 后 b。）

**D3 — 有损到什么程度？**（精度预算）
- (a) 先只做**无损**（dense prefill + paged decode + CUDA graph），把 flash 的分追平/微超，不碰精度。
- (b) 允许 **sparse**（tau 自适应），接受 HumanEval 掉 ≤1~2 题换 TTFT。
- (c) 进一步允许 **fp8 KV**（decode 换量级），只在端到端 HumanEval 上守质量。

**D4 — 主要冲哪个评分？**（TTFT 还是 decode tps 还是 pass@1 领先）
- 三者杠杆不同（TTFT→prefill kernel/稀疏；decode tps→CUDA graph/fp8；pass@1→守住即可）。
  告诉我优先级，我把实现顺序对齐。

---

## 8. 建议的实现顺序（若你认可，等 §7 决策后细化为任务）

1. **轨-0**：修 BUG-1 → 回归两个 tests → 改 serve 脚本关 chunked prefill → 用
   `perf_test.py --input-len 100000` 拿到「dense/tau=1 无损」的 TTFT，与 flash 对比（建立基线）。
2. **轨-B.1 质量画像**：完整 HumanEval `pass@1` 在 tau∈{1.0, 0.999, 0.99, 0.95} 上扫，画
   pass@1 vs TTFT 帕累托，定生产 tau。
3. **轨-A**：CUDA graph 化后端（batched paged decode + 稳定 metadata），去 `--enforce-eager`，
   `perf_test` 复测 decode tps。
4. **轨-B.2（可选）**：fp8 KV decode（V K-major + descale），端到端 HumanEval 守质量。

> 每步都遵循本项目一贯方法论（记忆库/HANDOFF）：算法先行→torch 参考→kernel→`--check-only`
> 对拍→空闲卡上 `--warmup/--iters` 稳定测速→诚实区分「正确」与「达标」。测速务必
> `nvidia-smi` 确认目标卡 util=0%（机器多人共用，GPU 0 常被占）。
