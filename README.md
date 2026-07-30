# vLLM 教程：跑通模型 · 自定义 Attention Backend · 分页注意力正确性检测

本教程带你在**单卡 GPU** 上完成三件事：

1. **Part 1**：用指定 commit 的 vLLM 跑通 Qwen3-32B，起一个 OpenAI 兼容服务并验证。
2. **Part 2**：给 vLLM 添加一个**自定义 attention backend**（提供一个简易 Triton 实现作为示例），只要满足接口，你就能把自己的 attention kernel 接进 vLLM。
3. **Part 3**：用这个 Triton attention 作为 **baseline**，方便地检查你自己 kernel 的**正确性**（支持分页 KV cache 输入）。

> 面向对象：拿到同款镜像的同学，照着本 README 从零复现。所有命令都是自包含的（绝对路径 + 环境变量 + 预期输出）。

> **📊 评分怎么算（先看这个）**：接入你自己的 attention kernel 后，用**统一分数** `E2E = TTFT + 1000 × TPOT`（生成 1000 token 的端到端秒数，**越低越好**）来衡量。本教程 flash_attn 的 **baseline = `147s`**（单张 H20，~95k 输入实测），你的**加速比 = `147s / 你的 E2E 用时`**。起好服务后跑一条 `scripts/perf_test.py` 即可直接读出分数，详见 [**Part 4.2「评测你自己 kernel 的 E2E 分数」**](#42-评测你自己-kernel-的-e2e-分数一条命令)。

> **📮 最终需要提交两张截图（缺一不可）**：
> 1. **`correctness.png`** —— 跑 [Part 3 正确性测试](#part-3分页注意力正确性检测) `tests/test_paged_attn_correctness.py` 的输出截图，须显示 **`ALL PASS`**；
> 2. **`performance.png`** —— 跑 [Part 4.2 性能测试](#42-评测你自己-kernel-的-e2e-分数一条命令) `scripts/perf_test.py`（默认参数：`--input-len 100000 --output-len 64`）的输出截图，须显示 `E2E 评分` 与最后一行 `[perf] SUMMARY ...`。
>
> **正确性是性能的前提**：`correctness.png` 必须先 `ALL PASS`，`performance.png` 的分数才有效——否则近似/丢 KV 的 kernel（如 sparse attention）可以只刷速度而不保证结果正确。两张图一起看，速度分才算数。

---

## 0. 环境与前置说明

本教程在如下环境验证通过（你的镜像应当一致）：

| 组件 | 版本 / 值 |
| --- | --- |
| GPU | NVIDIA H20（单卡 ~97GB 显存即可） |
| Python | 3.12（`/usr/bin/python`） |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1 |
| CUDA (nvcc) | 13.0 |
| vLLM（已装 wheel） | `0.18.1rc1.dev3933+ga49d37c6b`，对应 commit **`a49d37c6b`** |
| 模型 | `/dockerdata/models/Qwen3-32B`（**镜像内固定路径，两台机器一致，直接保留即可**；如你的模型在别处，用 `MODEL=/your/path` 覆盖） |

> **先把本仓库 clone 到本地任意目录**，后续所有命令都在仓库根目录执行（示例里的 `cd <你的仓库路径>` 换成你实际 clone 的位置）。vLLM 源码**不在本仓库里**，会由 `scripts/setup_vllm_source.sh` 克隆到仓库的**兄弟目录** `../vllm_src`，最终目录结构为：
>
> ```
> 你clone的位置/
> ├── <本仓库>/     # vllm-custom-attention-tutorial
> └── vllm_src/     # setup 脚本克隆的 vLLM 源码（不纳入本仓库）
> ```

关键点：**镜像里已经用 wheel 装好了 vLLM，且它的 commit 正好就是本教程要求的 `a49d37c6b`**，编译好的 CUDA 扩展（`.so`）都在。我们不重新编译 CUDA，而是：

- `git clone` 一份 vLLM 源码并 `checkout a49d37c6b`；
- 把已安装包里的 `.so` 全部**软链接**进源码树；
- 通过 `PYTHONPATH` 让 Python 优先用源码树里的 `.py`。

这样你就能自由修改/新增 vLLM 的 Python 代码（比如加自定义 backend），而无需漫长的 CUDA 编译。

> 为什么必须软链 `.so`？因为 vLLM 用 `import vllm._C_stable_libtorch` 这种**包内子模块**方式加载编译扩展，Python 只会在 `sys.path` 里**第一个** `vllm` 目录下找这些 `.so`。源码树本身没有 `.so`，不软链过去就会 `ModuleNotFoundError`。

---

## Part 1：跑通 Qwen3-32B

### 1.1 准备 vLLM 源码（一次性）

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
bash scripts/setup_vllm_source.sh
```

这个脚本会：
- clone vLLM 到仓库的**兄弟目录** `../vllm_src` 并 checkout `a49d37c6b`；
- 把已安装包的 16 个 `.so` + `_version.py` 软链接到源码树；
- 校验 `import vllm` 来自源码树、且 `vllm._C_stable_libtorch` 可用。

**预期输出结尾**：

```
  vllm.__file__ = .../vllm_src/vllm/__init__.py
  vllm._C_stable_libtorch: OK
  torch: 2.13.0+cu130 cuda: 13.0
  vllm.__version__: 0.18.1rc1.dev3933+ga49d37c6b
SETUP OK
```

之后所有命令都要让源码树在 `PYTHONPATH` 最前面：

```bash
export PYTHONPATH=../vllm_src:$PYTHONPATH
```

（下面的启动脚本已自动帮你设置，无需手动 export。）

### 1.2 单卡启动服务（flash attention 后端）

在**第一个终端**里启动：

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
GPU=0 PORT=8000 bash scripts/serve_qwen3_flashattn.sh
```

脚本核心命令（供理解）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../vllm_src \
python -m vllm.entrypoints.openai.api_server \
    --model /dockerdata/models/Qwen3-32B \
    --served-model-name qwen3-32b \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --attention-backend flash_attn
```

说明：
- **单卡**由 `CUDA_VISIBLE_DEVICES=0` 保证（`tensor_parallel_size` 默认 1）。
- `--attention-backend flash_attn` 显式指定 attn 后端（本 commit 用该 CLI 参数选后端，日志会打印 `Using AttentionBackendEnum.FLASH_ATTN backend.` / `Using FlashAttention version 3`）。
- Qwen3-32B 权重约 61GB，单张 H20（97GB）放得下；`--max-model-len 8192` 只是演示够用，可按需调大。
- 首次启动会加载 17 个权重分片 + torch.compile + 抓 CUDA graph，需要几分钟；看到日志出现 `Application startup complete` / 监听 `:8000` 即就绪。

### 1.3 验证服务与回答正确性

在**第二个终端**里跑冒烟测试（会自动等待服务就绪）：

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
PYTHONPATH=../vllm_src python scripts/smoke_test.py --port 8000 --model qwen3-32b
```

冒烟测试会发一道**答案确定**的算术题（`17 + 25`），并校验回答里是否出现正确答案 `42`——只“非空”不算通过，必须**算对**才算通过。这样能真正验证后端计算是否正确（下文 Part 2 会解释为什么这点很重要）。

**预期**：以 `[smoke] PASS` 结尾。示例（已实测）：

```
[smoke] 服务已就绪: http://127.0.0.1:8000/v1/models
[smoke] 发送请求，prompt='What is 17 + 25? Reply with only the number, nothing else.'
============================================================
模型回答：
42
============================================================
[smoke] PASS：服务正常，且回答包含正确答案 '42'
```

> 冒烟测试用 `chat_template_kwargs.enable_thinking=false` 关闭 Qwen3 的思考链，让回答简短、确定、快速。若想看带思考链的完整回答，可自己发一条不带该参数的请求（回答会包含 `<think>...</think>`，属正常现象）。

验证完可在启动终端按 `Ctrl-C` 停止服务，释放显存。

### 1.4 用 HumanEval 评测本服务（pass@1，贪心）

用 **HumanEval**（164 道 Python 函数补全题）通过 OpenAI 兼容接口评测本服务，
计算 `pass@1`（每题贪心解码只生成 1 个补全，要求它通过全部单测）。

整个流程分两步，脚本都在 `scripts/` 下，**只依赖 `openai` 客户端 + Python 标准库**：

- `scripts/humaneval_generate.py`：对每道题调用 `/v1/chat/completions`（`temperature=0`）
  生成补全，清洗成纯函数体，写到 `logs/humaneval_samples.jsonl`。
- `scripts/humaneval_evaluate.py`：把 `prompt + 补全 + 官方单测` 拼成可执行程序，
  在**独立子进程**里带超时执行（**沙箱**），统计 `pass@1`。

> HumanEval 数据集（`HumanEval.jsonl.gz`，164 题）会在首次运行 generate 时从官方
> `openai/human-eval` 仓库自动下载并缓存到 `data/`。离线环境可手动把该文件放到
> `data/HumanEval.jsonl.gz`。

**第 0 步：装 openai 客户端（一次性）**

```bash
pip install openai
```

**第 1 步：起好服务**（1.2 的 flash_attn 或 Part 2 的 CUSTOM 后端均可），在**另一个终端**生成补全：

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
# 先快速冒烟（前 20 题），确认链路通
PYTHONPATH=../vllm_src python scripts/humaneval_generate.py \
    --port 8000 --model qwen3-32b --limit 20

# 跑完整 164 题（并发 4 条请求加速）
PYTHONPATH=../vllm_src python scripts/humaneval_generate.py \
    --port 8013 --model qwen3-32b --concurrency 4 \
    --output logs/humaneval_samples.jsonl
```

**第 2 步：在沙箱里评测，得到 pass@1**：

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
python scripts/humaneval_evaluate.py \
    --samples logs/humaneval_samples.jsonl \
    --timeout 10 --workers 8 \
    --report logs/humaneval_report.jsonl
```

**输出结尾**（逐题打印 PASS/FAIL 后给出总分）：

```
==================================================
pass@1 = 145/164 = 0.8841  (88.41%)
```

**两后端实测对比**（单张 H20，完整 164 题，贪心 `pass@1`，本仓库实测）：

| 后端 | pass@1 | 说明 |
| --- | --- | --- |
| `flash_attn` | **145/164 = 88.41%** | baseline |
| `CUSTOM`（教学 Triton kernel） | **146/164 = 89.02%** | 与 baseline 相差 1 题 |

> **为什么这个对比重要**：两个后端跑出的分数**基本一致**（差 1 题 ≈ 0.6%，属贪心解码在个别
> 边界题上的正常抖动），这就证明了自定义 `CUSTOM` attention backend 的计算是**正确**的——
> 若后端把 KV cache 写错位置，分数会大幅崩塌（而不是只差 1 题）。这也是 Part 2 里那个"必须
> 校验答案正确、而非仅非空"的思想在**数据集规模**上的延伸验证。
>
> **预期分数区间**：Qwen3-32B 贪心 `pass@1` 通常落在 **~85%–95%**；具体分数会因思考链开关、
> 补全清洗、`max_tokens` 等细节略有浮动。你在自己机器上复现时，只要落在该区间、且两后端
> 分数接近，即为正常。
>
> **安全警告**：评测会**真实执行模型生成的代码**。本脚本已做进程级隔离（独立子进程 +
> SIGALRM 超时 + `RLIMIT_CPU/RLIMIT_AS` 资源上限），但这**不是**强隔离沙箱。大规模或
> 不可信场景请在**容器 / gVisor / nsjail** 等隔离环境中运行。

---

## Part 2：添加自定义 Attention Backend

目标：给 vLLM 接入一个**自定义 attention backend**，它内部用一个**简易 Triton attention** 作为示例实现。做到两点：

1. 用这个自定义后端能**正常启动模型**并给出**正确**回答；
2. 里面的 Triton attention 是个**可替换的示例**——你只要让自己的 kernel 满足同样的接口，就能替换它、接进 vLLM。

> ## ✅ 半天就能接入你已有的 kernel（已实测验证）
>
> **如果你已经写好了 attention kernel（例如 [`attention-test`](https://github.com/fvvluo/attention-test) 里的实现），接入 vLLM 只需改 1 个文件的 1 个函数。**
>
> 我们已经**端到端验证过**：把 `attention-test` 的参考 kernel（签名 `attention(q,k,v,causal,sm_scale)`、形状 `(batch,heads,seq,dim)`）按 [**2.5 的适配器模板**](#25-把你已有的-kernel-接进来半天冲刺版--照着做)填进去，正确性测试**直接三项全过**：
>
> ```
> [prefill] max_abs_err=2.09e-02  -> PASS
> [decode]  max_abs_err=1.11e-02  -> PASS
> [mixed]   max_abs_err=2.22e-02  -> PASS
> ================================================
> ALL PASS
> ```
>
> 也就是说 **2.5 给的模板是照抄就能跑通的**（不是伪代码）。时间紧的话，**可以直接跳到 [Part 2.5](#25-把你已有的-kernel-接进来半天冲刺版--照着做)** 照着做；2.1~2.4 是背景原理，先接入、后细看也行。

### 2.1 代码结构

自定义后端放在 `custom_backend/` 包里：

```
custom_backend/
├── __init__.py                 # 导出后端类与 paged_attention_triton（不用改）
├── triton_attention.py         # 【★你唯一要改的文件★】简易 Triton 分页注意力 kernel
├── custom_triton_backend.py    # v1 backend 三件套：Backend / MetadataBuilder / Impl（不用改）
└── plugin.py                   # vllm.general_plugins 入口：把类注册到 CUSTOM 后端（不用改）
```

> **接入你自己的算子，通常只改 `triton_attention.py` 一个文件**（半天冲刺照做版见 [2.5](#25-把你已有的-kernel-接进来半天冲刺版--照着做)）。

vLLM v1 的 attention backend 由“三件套”组成，都在 `custom_triton_backend.py` 里：

| 类 | 职责 |
| --- | --- |
| `CustomTritonBackend` | 声明后端能力、KV cache 形状，关联下面两个类 |
| `CustomTritonMetadataBuilder` | 每步前向把 vLLM 的通用元数据转成本后端所需的元数据 |
| `CustomTritonImpl` | `forward()`：写 KV cache + 调用 Triton attention |

为让教程尽量简单，本后端做了两个选择：
- `forward_includes_kv_cache_update = True`：把“写 KV cache”和“算 attention”都放进 `forward()`，一个方法看完整流程；
- 不参与 CUDA graph（`_cudagraph_support` 默认 `NEVER`），所以启动服务时用 `--enforce-eager`，路径最简单、最好调试。

### 2.2 注册机制（如何让 `--attention-backend CUSTOM` 生效）

vLLM 在**所有进程**（前端 / EngineCore / worker）启动时都会加载 `vllm.general_plugins` 组下的插件并调用其入口函数。我们在 `pyproject.toml` 里声明了这个入口：

```toml
[project.entry-points."vllm.general_plugins"]
custom_triton = "custom_backend.plugin:register"
```

`plugin.register()` 里调用 `register_backend(AttentionBackendEnum.CUSTOM, "custom_backend.custom_triton_backend.CustomTritonBackend")`，把 `CUSTOM` 这个枚举指向我们的后端类。之后 `--attention-backend CUSTOM` 就会选中它。

**安装（一次性，注册 entry point）**：

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
pip install -e .
```

> 用 `pip install -e .`（可编辑安装）即可，之后改 `custom_backend/` 里的代码**无需重装**。

### 2.3 用自定义后端启动服务

在**第一个终端**启动（脚本已设好 `PYTHONPATH` 和 `--enforce-eager --attention-backend CUSTOM`）：

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
GPU=0 PORT=8000 bash scripts/serve_qwen3_custom.sh
```

脚本核心命令（供理解）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../vllm_src \
python -m vllm.entrypoints.openai.api_server \
    --model /dockerdata/models/Qwen3-32B \
    --served-model-name qwen3-32b \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --enforce-eager \
    --attention-backend CUSTOM
```

就绪的关键日志（说明后端确实被选中）：

```
Using AttentionBackendEnum.CUSTOM backend.
...
Application startup complete.
```

> **注意性能**：这个教学 kernel 是“每个 query token 一个 Triton program”的朴素实现，只求正确、不求快。在 eager 模式下逐 token 生成会比较慢，因此下面的验证用**关闭思考链**的短请求。

### 2.4 验证正确性

在**第二个终端**跑同一个冒烟测试：

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
PYTHONPATH=../vllm_src python scripts/smoke_test.py --port 8000 --model qwen3-32b
```

**预期**（已实测，与 flash attn 后端结果一致）：

```
[smoke] 服务已就绪: http://127.0.0.1:8000/v1/models
[smoke] 发送请求，prompt='What is 17 + 25? Reply with only the number, nothing else.'
============================================================
模型回答：
42
============================================================
[smoke] PASS：服务正常，且回答包含正确答案 '42'
```

> **为什么要校验“答案正确”而不只是“非空”？** attention backend 即使把 KV cache 写错了位置，也能返回**非空但错误**的文本（一堆看似通顺、实则乱码的字）。只有用一道**答案已知**的题去校验，才能真正确认后端计算正确。这也是本教程在实现过程中踩过的坑：KV cache 有 **NHD/HND** 两种物理布局，写和读必须用**同一套步长(stride)寻址**才对得上（见 `custom_triton_backend.py` 里 `forward()` 的中文注释）。

### 2.5 把你已有的 kernel 接进来（半天冲刺版 · 照着做）

> ## 👉 直接告诉你要改哪里
>
> **改 1 个文件、1 个函数、其它全部不动：**
>
> | | |
> |---|---|
> | **改哪个文件** | `custom_backend/triton_attention.py` |
> | **改哪个函数** | 只改 `paged_attention_triton(...)` 的**函数体**（示例里的 `triton_attention.py:144-173`） |
> | **怎么改** | 函数体清空，换成 3 行逻辑：**① 把入参整理成你 kernel 的输入 → ② 调你的 kernel → ③ 结果写进 `output` 并 `return output`** |
> | **绝不能改** | 函数名 `paged_attention_triton`、它的参数列表（顺序/名字）|
> | **不用碰** | `custom_triton_backend.py`、`plugin.py`、`__init__.py`、`pyproject.toml`，以及文件名本身 |
> | **示例里的 `_paged_attn_kernel`** | 那是**旧的教学 kernel**，直接删掉或忽略，用不上 |
> | **你们 `attention-test` 的 kernel** | 签名 `attention(q,k,v,causal,sm_scale)`、形状 `(batch,heads,seq,dim)`，和 vLLM 给的形状不同——[**第 2 步**](#第-2-步套用这个适配器模板已按-attention-test-的-kernel-签名写好)有**已写好、可直接抄**的转换模板 |
>
> 改完跑 **第 3 步** 的三条命令，全绿即接入成功。下面第 1、2 步是这张表的展开说明 + 可抄模板。

#### 第 1 步：认清这个函数（唯一的接入点）

`paged_attention_triton(...)` 就是 vLLM 每次算 attention 时会调用的入口。vLLM 会把这些**已经准备好的张量**传给你（形状/含义如下），你只要用它们算出 attention、写进 `output` 即可：

```python
def paged_attention_triton(
    query,           # [num_tokens, num_heads, head_size]  本次要算的 query（prefill+decode 已拼在一起）
    key_cache,       # [num_blocks, num_kv_heads, block_size, head_size]  分页 K cache（当前 K 已写好）
    value_cache,     # [num_blocks, num_kv_heads, block_size, head_size]  分页 V cache（当前 V 已写好）
    output,          # [num_tokens, num_heads, head_size]  ★结果原地写进这里★
    query_start_loc, # [num_seqs+1] int32  第 i 条请求的 query 落在 query[start_i : start_{i+1}]
    seq_lens,        # [num_seqs]   int32  第 i 条请求的总长度（历史 context + 本次 query）
    token_seq_idx,   # [num_tokens] int32  每个 token 属于第几条请求（已替你算好）
    block_table,     # [num_seqs, max_num_blocks] int32  逻辑块号 -> 物理块号
    scale,           # float  softmax 缩放系数 1/sqrt(head_size)
) -> torch.Tensor:   # 返回 output
    ...
```

要点：
- **KV cache 已经写好了**（写 cache 由上游 `CustomTritonImpl.forward` 完成，不用你管），你只负责**读 cache 算 attention**。
- 结果必须**原地写进 `output`**（形状 `[num_tokens, num_heads, head_size]`），最后 `return output`。
- 必须是 **causal**（token 只能看自己及之前的位置）+ **GQA**（`num_heads` 是 `num_kv_heads` 的整数倍，多个 Q 头共享一个 KV 头）。
- 第 `req` 条请求第 `j` 个位置的 KV 在：`pb = block_table[req, j // block_size]`，槽位 `j % block_size`，即 `key_cache[pb, kv_head, j % block_size, :]`。

#### 第 2 步：套用这个适配器模板（已按 `attention-test` 的 kernel 签名写好）

**为什么需要适配**：你们在 [`attention-test`](https://github.com/fvvluo/attention-test) 里写的 kernel 签名是——

```python
attention(q, k, v, causal=True, sm_scale=None) -> output
#  q:      (batch, q_heads,  seq_len, head_dim)
#  k, v:   (batch, kv_heads, seq_len, head_dim)   # GQA: q_heads 是 kv_heads 整数倍
#  output: 同 q, (batch, q_heads, seq_len, head_dim)
```

而 vLLM 传给 `paged_attention_triton` 的是**另一套形状**：query 没有 batch 维、是变长序列拼接的 `[num_tokens, num_heads, head_size]`；KV 是**分页**的 `[num_blocks, num_kv_heads, block_size, head_size]`。**接入 = 在这个函数里把两套形状对接起来**。把下面这段**直接抄进 `triton_attention.py` 替换函数体**（把 `YOUR_ATTENTION` 换成你 import 进来的那个 `attention`）：

```python
# 顶部先 import 你自己的 kernel，例如：
# from ops.your_flash_attention import attention as YOUR_ATTENTION

def paged_attention_triton(query, key_cache, value_cache, output,
                           query_start_loc, seq_lens, token_seq_idx,
                           block_table, scale):
    num_seqs = seq_lens.shape[0]
    block_size = key_cache.shape[2]
    for i in range(num_seqs):
        s, e = int(query_start_loc[i]), int(query_start_loc[i + 1])   # 本请求 query 区间
        L = int(seq_lens[i])                                          # 该请求总长度(含历史)

        # 1) 用 block_table 把分页 KV gather 成连续序列 [L, num_kv_heads, head_size]
        blocks = block_table[i]
        rows = torch.arange(L, device=key_cache.device)
        pb = blocks[rows // block_size]                               # 每个位置的物理块号
        slot = rows % block_size
        k_i = key_cache[pb, :, slot, :]                               # [L, num_kv_heads, hs]
        v_i = value_cache[pb, :, slot, :]

        # 2) 转成你 kernel 要的 (batch=1, heads, seq, dim)
        q_i = query[s:e].transpose(0, 1).unsqueeze(0)                 # [1, num_heads,  q_len, hs]
        k_i = k_i.transpose(0, 1).unsqueeze(0)                        # [1, num_kv_heads, L,   hs]
        v_i = v_i.transpose(0, 1).unsqueeze(0)

        # 3) 调你的 kernel（causal + GQA 已由 kernel 内部处理，scale 用 vLLM 给的）
        out_i = YOUR_ATTENTION(q_i, k_i, v_i, causal=True, sm_scale=scale)
        #        out_i: [1, num_heads, q_len, hs]

        # 4) 变回 [q_len, num_heads, hs] 写回 output（原地）
        output[s:e] = out_i.squeeze(0).transpose(0, 1)
    return output
```

> **对齐要点（照抄即可，不用自己推）**：
> - 你的 kernel 用 `(kv_len - q_len + i)` 做 causal 对齐（见 `_example_flash_attention.py`），正好等于 vLLM 里 `context_len + i` 的绝对位置——所以 **decode（q_len=1、L=完整长度）也能直接跑通**，无需额外处理。
> - `sm_scale` 直接传 vLLM 给的 `scale`（就是 `1/sqrt(head_size)`），别再让 kernel 用默认 `None`，否则可能双重缩放。
> - GQA 的 head 展开（`repeat_interleave`）在**你 kernel 内部**做，这里不用管。

> ✅ **本模板已实测**：用 `attention-test` 的参考 kernel 套用上面这段，`tests/test_paged_attn_correctness.py` 三项（prefill/decode/mixed）全部 PASS。照抄即可，不用怀疑对错。

#### 第 3 步：跑通三连（每步都有预期输出，照着比对）

```bash
cd <你的仓库路径>

# ① 先离线验正确性（最快，不用起 32B 服务）——必须 ALL PASS
PYTHONPATH=../vllm_src:. python tests/test_paged_attn_correctness.py

# ② 起服务（第一个终端）
GPU=0 PORT=8000 bash scripts/serve_qwen3_custom.sh

# ③ 冒烟测试（第二个终端）——必须 PASS 且答案含 42
PYTHONPATH=../vllm_src python scripts/smoke_test.py --port 8000 --model qwen3-32b
```

三步全绿，你的 kernel 就接通了。之后按 Part 4 测 E2E 分数、和 baseline `147s` 比加速比。

#### 🔒 铁律（省得踩坑返工）
- **不要改** `paged_attention_triton` 的**函数名和参数列表**（`__init__.py` / `custom_triton_backend.py` 按名字和关键字调它）。函数体随便写。
- **不要动** `custom_triton_backend.py` / `plugin.py` / `__init__.py` / `pyproject.toml`。
- 文件名**保持 `triton_attention.py`**（改名要同步改 2 处 import，没必要）。
- 结果一定写进 `output` 并 `return output`；别新建张量返回却不写 `output`。

> **进阶（一般用不到）**：若你的 kernel 签名实在没法套进这个函数，也可以改 `CustomTritonImpl.forward()`（`custom_triton_backend.py`）里调用它的那几行。但对半天冲刺来说，**只改 `triton_attention.py` 函数体**是最快路径。

完整接口约定（签名 + 语义 + 分页寻址规则）另见下方 **Part 3.1**。

---

## Part 3：分页注意力正确性检测

Part 2 里的 Triton attention 直接**接收分页 KV cache（paged KV cache）作为输入**。我们把它当作 **baseline**：你实现自己的 kernel 后，用同一份测试就能方便地校验正确性——无需启动整个 32B 服务。

### 3.1 接口约定（你的 kernel 要满足的签名）

```python
def paged_attention_triton(
    query: torch.Tensor,        # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size]
    output: torch.Tensor,       # [num_tokens, num_heads, head_size]（原地写入）
    query_start_loc: torch.Tensor,  # [num_seqs + 1] int32：每条请求 query 的起始偏移
    seq_lens: torch.Tensor,         # [num_seqs]     int32：每条请求总长度(context+query)
    token_seq_idx: torch.Tensor,    # [num_tokens]   int32：每个 token 属于哪条请求
    block_table: torch.Tensor,      # [num_seqs, max_num_blocks] int32：逻辑块->物理块
    scale: float,
) -> torch.Tensor:                  # 返回 output
```

语义要求：
- **causal**：每个 query token 只能看到不超过自身绝对位置的 KV；
- **GQA**：`num_heads` 是 `num_kv_heads` 的整数倍，多个 Q 头共享一个 KV 头；
- **分页寻址**：第 `req` 条请求第 `j` 个（全局）位置的 KV，物理位置是
  `block = block_table[req, j // block_size]`、槽位 `j % block_size`，即
  `key_cache[block, kv_head, j % block_size, :]`；
- 全部按张量**步长(stride)**寻址，不假设特定物理内存布局。

### 3.2 运行正确性测试

测试 `tests/test_paged_attn_correctness.py` 会构造分页 KV cache，把 `paged_attention_triton` 的输出与一份**朴素 PyTorch 参考实现**逐元素比对，覆盖 **prefill / decode / 混合** 三种场景。

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
PYTHONPATH=../vllm_src:. python tests/test_paged_attn_correctness.py
```

**预期输出**（bf16 容差 `rtol=atol=2e-2`，已实测）：

```
[prefill] max_abs_err=7.8111e-03  -> PASS
[decode] max_abs_err=3.8853e-03  -> PASS
[mixed] max_abs_err=7.7505e-03  -> PASS
==================================================
ALL PASS
```

### 3.3 用它检查你自己的 kernel

1. 把 `custom_backend/triton_attention.py` 里的 `paged_attention_triton` 换成你自己的实现（保持签名与语义）；
2. 重新跑 3.2 的测试：
   - 若 `ALL PASS`，说明你的 kernel 在分页 KV cache 上计算正确；
   - 若某个 case `FAIL`，看 `max_abs_err`：数量级很大（如 >1）通常是**寻址/布局**错了（block/slot/head 索引或 stride 用错）；只是略超容差则多半是**数值精度**问题（累加顺序、是否用 fp32 做 online softmax）。
3. 正确性通过后，再按 Part 2 用 `--attention-backend CUSTOM` 起服务，用 2.4 的冒烟测试做端到端确认。

---

## Part 4：性能测试（长上下文 100k 输入）

正确性之外，attention backend 的另一半价值是**性能**——尤其在**长上下文**下。本部分给出一个
可复现的性能测试方法，用 **~100k token 的输入**压测 prefill，并对比 `flash_attn` 与 `CUSTOM`
两个后端。

脚本 `scripts/perf_test.py` 用流式接口测量，并汇总出下面几个指标：

| 指标 | 含义 |
| --- | --- |
| **TTFT**（Time To First Token，首 token 延迟） | 从发请求到收到第 1 个输出 token 的时间。长输入下**主要就是 prefill（处理这 100k 输入）的耗时**，最能体现 attention 后端在长序列上的效率。 |
| **decode 吞吐**（tokens/s） | 首 token 之后逐 token 生成的速度。 |
| **TPOT**（Time Per Output Token） | decode 阶段平均每个 token 的耗时，= 1 / decode 吞吐。 |
| **E2E 评分** | `TTFT + 1000 × TPOT`，即"生成 1000 个 token 的端到端秒数"。把 prefill 与 decode 汇总成一个**可比较的分数（越低越好）**——用它给不同 attention kernel 打分。 |

### 4.0 关于上下文长度：本检查点需开 YaRN 才能到 128k（重要）

Qwen3 系列**标称** 128k 上下文，但**要看具体检查点的 config**。本教程的
`/dockerdata/models/Qwen3-32B` 里 `config.json` 是：

```
max_position_embeddings = 40960     # 原生只到 ~40k
rope_scaling            = None
```

也就是说**原生只支持 ~40k**。要跑 100k 输入，需用 **YaRN** rope 缩放把上下文扩展到 128k+
（Qwen3 官方推荐做法）。本教程通过 `--hf-overrides` 注入 YaRN 配置（`serve_qwen3_flashattn.sh`
已支持用 `HF_OVERRIDES` 环境变量传入）：

```json
{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},
 "max_position_embeddings":163840}
```

`factor=4.0` 把 40960 扩到 163840（>128k），足以覆盖 100k 输入。

> **前置**：先 `pip install openai transformers`（`transformers` 用于把输入**精确**编码到
> 目标 token 数；缺失时脚本退化为按字符估算并提示）。

### 4.1 flash_attn 后端：完整 100k 性能

在**第一个终端**起服务（开 YaRN、`--max-model-len` 调大；100k 的 KV cache 占显存较多，
用一张**空闲卡**并适当调高 `--gpu-memory-utilization`）：

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
HF_OVERRIDES='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":40960},"max_position_embeddings":163840}' \
MAX_LEN=102400 GPU_MEM_UTIL=0.94 GPU=4 PORT=8004 \
bash scripts/serve_qwen3_flashattn.sh
```

> 单张 H20（97GB）装完 32B 权重（~61GB）后，KV cache 余量有限：`gpu-memory-utilization=0.92`
> 时只够 ~92k token，`--max-model-len 110000` 会报 KV cache 不足。这里用 `0.94` + `102400`
> 实测能放下 100k 输入 + 少量输出：KV cache ~106k token（>102400，够用），prefill 峰值也不触发 OOM。
> （`0.97` 也能跑通，但 prefill 峰值时显存只剩 ~440MB、出现过一次非致命 OOM 重试贴着天花板跑；
> 推荐用 `0.94` 留出余量更稳。）若仍报显存不足，降 `MAX_LEN` 或换更空闲的卡。

在**第二个终端**跑性能测试：

```bash
cd <你的仓库路径>   # 你 clone 本仓库的位置
python scripts/perf_test.py --port 8004 --input-len 100000 --output-len 64
```

> `perf_test.py` **默认给每次请求加一段唯一前缀**，绕过 vLLM 的 prefix caching，
> 这样测到的 TTFT 才是**真实的长输入 prefill 耗时**。如果不加唯一前缀（`--allow-prefix-cache`），
> 第二次起相同 prompt 会命中缓存、跳过 prefill，TTFT 会假性降到亚秒级（缓存收益本身也值得一看）。

**实测输出（本教程 baseline，单张 H20 / `GPU_MEM_UTIL=0.94`；数值随卡型/负载/vLLM 版本浮动）**：

```
[perf] 服务: http://127.0.0.1:8004/v1  模型: qwen3-32b
[perf] 目标输入长度: 100000 tokens  生成上限: 64 tokens
[perf] 实际输入长度(精确): ~95653 tokens
[perf] 预热 1/1 ...
[perf] 第 1/3 次: TTFT=111.390s  decode=28.2 tok/s  total=112.138s  out=21 tok
[perf] 第 2/3 次: TTFT=111.411s  decode=27.4 tok/s  total=113.236s  out=50 tok
[perf] 第 3/3 次: TTFT=111.411s  decode=28.1 tok/s  total=112.269s  out=24 tok
================================================================
输入长度        : ~95653 tokens (精确)
生成长度        : 24 tokens (上限 64)
TTFT (中位数)   : 111.411 s   <- prefill 长输入的耗时
TPOT            : 35.57 ms/token   <- decode 每 token 耗时 (=1/吞吐)
decode 吞吐     : 28.1 tokens/s   <- 首 token 之后的生成速度
端到端总时延    : 112.269 s   <- 本次实际请求 (~24 tok) 的总耗时
E2E 评分        : 146.981 s   <- TTFT + 1000×TPOT (生成 1000 tok 的端到端时间，越低越好)
================================================================
[perf] SUMMARY input=95653 out=24 ttft_s=111.411 tpot_ms=35.57 decode_tps=28.1 total_s=112.269 e2e_score_s=146.981
```

> **为什么是 ~95653 而不是 100000？** `--input-len 100000` 是**目标**长度；脚本用 Qwen3 tokenizer
> 把重复文本编码→截断到 10 万 token→再 `decode` 回文本，而 decode 后**重新编码**时 token 边界会重新
> 合并（非无损），实际落到 ~95653。脚本诚实地打印这个**重编码后的真实值**并用它计分。
>
> **这个长度对所有人是统一的、可复现的**：filler 文本在脚本里写死、tokenizer 都是同一个
> `/dockerdata/models/Qwen3-32B`、截断/编解码都是确定性操作——所以只要大家都用**默认的
> `--input-len 100000` 跑同一个脚本**，实际输入就都是 ~95653，不会出现"他 95k、他 98k、他 90k"。
> 唯一会变的是有人手动改了 `--input-len`（那样就不可比了，见 4.2 的提交要求）。

> 📌 **本教程 baseline（flash_attn，单张 H20，~95k 输入实测）：E2E ≈ `147s`**（= TTFT 111.4s + 1000×TPOT 35.57ms）。上面即为实测结果。
>
> **你的加速比 = `147s / 你的 E2E 用时`**（>1 即比 baseline 快）。换上自己的 kernel 后跑同样命令，用输出的 `e2e_score_s` 代入即可。

> 补充：单张 H20 上 ~95k token 的真实 prefill（无缓存）就是上面的 TTFT ~111s、decode ~28 tok/s。
> 若加 `--allow-prefix-cache` 复跑，第 2 次起 TTFT 会降到 **~0.2s**（命中前缀缓存，跳过 prefill）。

### 4.2 评测你自己 kernel 的 E2E 分数（一条命令）

换上自己的 attention kernel 后，最省事的评测方式：**起好服务，跑一条 `perf_test.py`，直接读最后一行的 `e2e_score_s`**。

```bash
# 服务端起在某个端口后（flash_attn 或你的 CUSTOM 后端都行）：
python scripts/perf_test.py --port 8004 --input-len 100000 --output-len 64
```

脚本会输出三个你关心的数（下面是 flash_attn baseline 的值）：

```
TTFT (中位数)   : 111.411 s
TPOT            : 35.57 ms/token
E2E 评分        : 146.981 s   <- TTFT + 1000×TPOT
```

**评分口径统一为 `E2E = TTFT + 1000 × TPOT`**，即"生成 1000 个 token 的端到端时间"，**数值越低越好**。它同时惩罚 prefill 慢（TTFT 高）和 decode 慢（TPOT 高），所以一个数就能公平比较不同 kernel。手算也行：

```
TPOT(s)   = 1 / decode_tps                # 例：1 / 28.1 ≈ 0.03557 s = 35.57 ms
E2E(s)    = TTFT + 1000 × TPOT            # 例：111.411 + 1000 × 0.03557 ≈ 146.98 s
```

> **成绩可比的前提（重要）**：必须用**默认参数**跑 —— `--input-len 100000`（实际 ~95653 token，人人一致）、
> `--output-len 64`、不加 `--allow-prefix-cache`。改了输入长度或开了前缀缓存，分数就**不可与 baseline 或他人比较**。
> 提交成绩时请连同脚本最后一行 `[perf] SUMMARY ...`（含 `input= / ttft_s / tpot_ms / decode_tps / e2e_score_s`）
> 与 **GPU 型号**一起给出；不同卡型之间也不可直接比（baseline 是单张 H20）。

### 4.3 CUSTOM 后端：只在小长度上对比（朴素 kernel 跑不动 100k）

**为什么 CUSTOM 不跑 100k**：Part 2 的教学 kernel 是 `grid=(num_tokens × num_heads)` 的朴素
实现——每个 (token, head) 一个 Triton program，program 内用一层 `for` **串行**扫过整条 KV
序列；再加上服务用 `--enforce-eager`（不抓 CUDA graph）。100k prefill 意味着
`100000 × 64 ≈ 640 万`个 program，每个还要串行读多达 100k 个 KV 位置——实测**极慢、不可实用化**。
这正是"**为什么需要一个真正高效的 kernel**"的教学点。

所以对 CUSTOM 我们只在**小长度**上测，用来直观感受它比 `flash_attn` 慢多少：

```bash
# 第一个终端：CUSTOM 后端，小上下文即可（无需 YaRN）
cd <你的仓库路径>   # 你 clone 本仓库的位置
GPU=5 PORT=8005 bash scripts/serve_qwen3_custom.sh

# 第二个终端：用小 --input-len 测（如 2048）
cd <你的仓库路径>   # 你 clone 本仓库的位置
python scripts/perf_test.py --port 8005 --input-len 2048 --output-len 32
```

**预期输出**（单张 H20 实测，仅 ~2k 输入）：

```
[perf] 实际输入长度(精确): ~1960 tokens
[perf] 第 1/3 次: TTFT=9.113s  decode=9.8 tok/s  total=12.263s  out=32 tok
...
================================================================
输入长度        : ~1960 tokens (精确)
TTFT (中位数)   : 9.117 s
decode 吞吐     : 9.8 tokens/s
================================================================
[perf] SUMMARY input=1960 out=32 ttft_s=9.117 decode_tps=9.8 total_s=12.268
```

### 4.4 对比与结论

同样 ~2k 输入下两个后端的实测对比（单张 H20，供量级参考）：

| 后端 | 输入长度 | TTFT | decode 吞吐 |
| --- | --- | --- | --- |
| `flash_attn` | ~1960 tokens | **0.97 s** | **44.2 tok/s** |
| `CUSTOM`（教学 kernel） | ~1960 tokens | **9.12 s** | **9.8 tok/s** |

即使只在 2k 这种小输入上，教学版 `CUSTOM` 的 prefill 也比 `flash_attn` 慢约 **9 倍**、
decode 慢约 **4~5 倍**；把输入放大到 100k，这个差距会进一步急剧拉大（所以 `CUSTOM` 不跑 100k）。
作为对照，`flash_attn` 在 **~95k** 真实输入上的 TTFT 也才 ~110s、decode ~28 tok/s。

结论与延伸：
- 教学版 `CUSTOM` 只保证**正确**、不追求快；性能差距正是把
  `custom_backend/triton_attention.py` 换成**高效 kernel**（沿 KV 分块并行、向量化、支持 CUDA graph
  而非逐 token 逐 program 串行扫）的动机（见 Part 2.5 / Part 3 的接口约定）。
- 换上你自己的高效 kernel 后，工作流是：先按 **Part 3** 用正确性测试确认无误，
  再用本节 `perf_test.py` 对比它和 `flash_attn` 的性能差距。

> **说明**：性能数值高度依赖 GPU 型号、显存、并发负载与 vLLM 版本，仅作**量级**参考，
> 不代表 Qwen3-32B 或 vLLM 的官方性能指标。
