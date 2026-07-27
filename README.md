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

**预期**：打印一段对“注意力机制”的合理中文解释，并以 `[smoke] PASS` 结尾。示例（已实测）：

```
[smoke] 服务已就绪: http://127.0.0.1:8000/v1/models
[smoke] 发送请求，prompt='用一句话解释什么是注意力机制（attention）。'
============================================================
模型回答：
<think> ... 注意力机制是一种让模型在处理信息时动态关注关键部分 ... </think>
============================================================
[smoke] PASS：服务正常且返回了非空回答
```

> Qwen3 默认开启 thinking 模式，回答里带 `<think>...</think>` 属正常现象。

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

## Part 2：添加自定义 Attention Backend（待补充）

（实现完成并实测通过后填写。）

## Part 3：分页注意力正确性检测（待补充）

（实现完成并实测通过后填写。）
