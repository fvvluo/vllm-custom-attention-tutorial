# vLLM 教程：跑通模型 · 自定义 Attention Backend · 分页注意力正确性检测

本教程带你在**单卡 GPU** 上完成三件事：

1. **Part 1**：用指定 commit 的 vLLM 跑通 Qwen3-32B，起一个 OpenAI 兼容服务并验证。
2. **Part 2**：给 vLLM 添加一个**自定义 attention backend**（提供一个简易 Triton 实现作为示例），只要满足接口，你就能把自己的 attention kernel 接进 vLLM。
3. **Part 3**：用这个 Triton attention 作为 **baseline**，方便地检查你自己 kernel 的**正确性**（支持分页 KV cache 输入）。

> 面向对象：拿到同款镜像的同学，照着本 README 从零复现。所有命令都是自包含的（绝对路径 + 环境变量 + 预期输出）。

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
| 模型 | `/dockerdata/models/Qwen3-32B` |

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
cd /dockerdata/landojiang/vllm_tutorial
bash scripts/setup_vllm_source.sh
```

这个脚本会：
- clone vLLM 到 `/dockerdata/landojiang/vllm_src` 并 checkout `a49d37c6b`；
- 把已安装包的 16 个 `.so` + `_version.py` 软链接到源码树；
- 校验 `import vllm` 来自源码树、且 `vllm._C_stable_libtorch` 可用。

**预期输出结尾**：

```
  vllm.__file__ = /dockerdata/landojiang/vllm_src/vllm/__init__.py
  vllm._C_stable_libtorch: OK
  torch: 2.13.0+cu130 cuda: 13.0
  vllm.__version__: 0.18.1rc1.dev3933+ga49d37c6b
SETUP OK
```

之后所有命令都要让源码树在 `PYTHONPATH` 最前面：

```bash
export PYTHONPATH=/dockerdata/landojiang/vllm_src:$PYTHONPATH
```

（下面的启动脚本已自动帮你设置，无需手动 export。）

### 1.2 单卡启动服务（flash attention 后端）

在**第一个终端**里启动：

```bash
cd /dockerdata/landojiang/vllm_tutorial
GPU=0 PORT=8000 bash scripts/serve_qwen3_flashattn.sh
```

脚本核心命令（供理解）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dockerdata/landojiang/vllm_src \
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
cd /dockerdata/landojiang/vllm_tutorial
PYTHONPATH=/dockerdata/landojiang/vllm_src python scripts/smoke_test.py --port 8000 --model qwen3-32b
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

### 1.4 TODO（留给助教）：接入 HumanEval 测试

<!-- =========================== TODO: HUMAN EVAL =========================== -->
> **TODO（助教补充）**：在此处补充用 **HumanEval** 数据集评测本服务的方式。
>
> 建议思路（待助教完善为可复现步骤）：
> 1. 起好本服务（1.2 节）后，用 EvalPlus 或官方 `human-eval` 库通过 OpenAI 兼容接口
>    （`http://127.0.0.1:8000/v1`，模型名 `qwen3-32b`）批量生成代码补全；
> 2. 在**沙箱/容器**内执行生成的代码，计算 `pass@1`（贪心）等指标；
> 3. 把生成脚本、评测命令、预期分数区间补充到本节，并在 `scripts/` 下提供对应脚本。
>
> 注意：评测会真实执行模型生成的代码，务必在隔离环境中进行。
<!-- ========================================================================= -->

---

## Part 2：添加自定义 Attention Backend

目标：给 vLLM 接入一个**自定义 attention backend**，它内部用一个**简易 Triton attention** 作为示例实现。做到两点：

1. 用这个自定义后端能**正常启动模型**并给出**正确**回答；
2. 里面的 Triton attention 是个**可替换的示例**——你只要让自己的 kernel 满足同样的接口，就能替换它、接进 vLLM。

### 2.1 代码结构

自定义后端放在 `custom_backend/` 包里：

```
custom_backend/
├── __init__.py                 # 导出后端类与 paged_attention_triton
├── triton_attention.py         # 【你要替换的部分】简易 Triton 分页注意力 kernel
├── custom_triton_backend.py    # v1 backend 三件套：Backend / MetadataBuilder / Impl
└── plugin.py                   # vllm.general_plugins 入口：把类注册到 CUSTOM 后端
```

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
cd /dockerdata/landojiang/vllm_tutorial
pip install -e .
```

> 用 `pip install -e .`（可编辑安装）即可，之后改 `custom_backend/` 里的代码**无需重装**。

### 2.3 用自定义后端启动服务

在**第一个终端**启动（脚本已设好 `PYTHONPATH` 和 `--enforce-eager --attention-backend CUSTOM`）：

```bash
cd /dockerdata/landojiang/vllm_tutorial
GPU=0 PORT=8000 bash scripts/serve_qwen3_custom.sh
```

脚本核心命令（供理解）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/dockerdata/landojiang/vllm_src \
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
cd /dockerdata/landojiang/vllm_tutorial
PYTHONPATH=/dockerdata/landojiang/vllm_src python scripts/smoke_test.py --port 8000 --model qwen3-32b
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

### 2.5 如何替换成你自己的 kernel

你**只需要**改 `custom_backend/triton_attention.py` 里的 `paged_attention_triton(...)`（保持函数签名与语义不变），或在 `CustomTritonImpl.forward` 里改成调用你自己的函数。接口约定见 Part 3。

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
cd /dockerdata/landojiang/vllm_tutorial
PYTHONPATH=/dockerdata/landojiang/vllm_src:. python tests/test_paged_attn_correctness.py
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
