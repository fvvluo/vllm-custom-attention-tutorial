#!/usr/bin/env python
"""
humaneval_generate.py —— 用本地 vLLM OpenAI 兼容服务，对 HumanEval 做贪心代码补全。
=============================================================================

它做什么：
  1) 拿到 HumanEval 数据集（164 道 Python 函数补全题）。数据集来自官方
     openai/human-eval 仓库的 HumanEval.jsonl.gz，首次运行会下载并缓存到
     本目录下的 data/HumanEval.jsonl.gz（离线环境可手动放置该文件）。
  2) 对每道题，把题目里的函数签名+docstring（字段 prompt）发给
     /v1/chat/completions，用 temperature=0（贪心）让模型补全函数体；
  3) 把模型输出清洗成"纯 Python 补全"（去掉 ```python 代码块围栏、去掉模型
     重复打印的 import/函数签名），写到 samples.jsonl，供 humaneval_evaluate.py
     在沙箱里执行、算 pass@1。

为什么用贪心（temperature=0）：
  pass@1 的标准做法是每题只采样 1 个补全并要求它通过全部单测。贪心解码可复现、
  无随机性，最适合教学里"跑一次就能对比分数"的场景。

用法：
    # 先按 README 1.2 起好服务（flash_attn 或 CUSTOM 后端都行）
    python scripts/humaneval_generate.py \
        --port 8000 --model qwen3-32b \
        --output logs/humaneval_samples.jsonl

    # 快速冒烟：只跑前 20 题
    python scripts/humaneval_generate.py --limit 20

输出文件格式（每行一个 JSON）：
    {"task_id": "HumanEval/0", "prompt": "...", "completion": "    ...函数体..."}
"""
import argparse
import concurrent.futures
import gzip
import json
import os
import re
import sys
import time
import urllib.request

# 官方 HumanEval 数据集（164 题）。首次运行下载并缓存到本地。
HUMANEVAL_URL = (
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "HumanEval.jsonl.gz")


def load_problems(data_path: str) -> list:
    """加载 HumanEval 题库；本地没有就从官方仓库下载并缓存。"""
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        print(f"[gen] 本地无数据集，正在下载: {HUMANEVAL_URL}")
        urllib.request.urlretrieve(HUMANEVAL_URL, data_path)
        print(f"[gen] 已缓存到: {data_path}")
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    print(f"[gen] 载入 {len(problems)} 道题: {data_path}")
    return problems


def build_prompt(problem: dict) -> str:
    """把 HumanEval 的函数签名+docstring 包装成一条明确的补全指令。

    要点：要求模型**只**输出补全后的完整函数（含签名），不要解释、不要测试，
    这样 sanitize_completion 才能稳定地把它切成纯代码。
    """
    return (
        "Complete the following Python function. "
        "Return ONLY the complete function implementation in a single ```python code "
        "block, including the given signature. Do not add explanations, examples, "
        "tests, or any text outside the code block.\n\n"
        "```python\n"
        f"{problem['prompt']}\n"
        "```"
    )


_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code_block(text: str) -> str:
    """取出第一个 ```python ... ``` 代码块；没有围栏就返回原文。"""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    # 没有围栏：去掉可能残留的开头 ``` 行
    return text.strip().removeprefix("```python").removeprefix("```").strip("\n")


def sanitize_completion(raw: str, problem: dict) -> str:
    """把模型输出转成"接在 problem['prompt'] 之后"的函数体补全（completion）。

    HumanEval 的评测方式是执行  prompt + completion + test。prompt 里已经包含了
    函数签名，所以 completion 必须是**函数体**（缩进的若干行），不能再重复签名。

    这里的策略：
      1) 抽出代码块；
      2) 找到目标函数 `def <entry_point>(` 所在行，丢弃它之前的所有内容
         （模型常把 import/别的辅助函数写在前面——但 prompt 已给过 import，
          我们只保留目标函数体，避免重复定义带来的副作用）；
      3) 丢弃这行 `def ...:` 本身，保留其后的缩进函数体作为 completion。
    """
    code = _extract_code_block(raw)
    entry = problem["entry_point"]
    lines = code.splitlines()

    # 定位目标函数定义行
    def_idx = None
    def_re = re.compile(rf"^\s*def\s+{re.escape(entry)}\s*\(")
    for i, ln in enumerate(lines):
        if def_re.match(ln):
            def_idx = i
            break

    if def_idx is None:
        # 模型没重写签名，直接把整块当函数体（保证有缩进）
        body_lines = code.splitlines()
    else:
        # 取签名行之后、直到函数结束（下一个顶格非空行）为止
        body_lines = []
        for ln in lines[def_idx + 1:]:
            # 顶格且非空 => 已经离开函数体（可能是别的 def / 测试代码），停止
            if ln.strip() and not ln.startswith((" ", "\t")):
                break
            body_lines.append(ln)

    # 若模型把 docstring 也重复写了一遍，去掉它——因为 problem['prompt'] 末尾已含
    # 同一个 docstring，拼接后会重复。评测拼的是 prompt + completion，completion 只需
    # 是真正的实现语句。
    body_lines = _strip_leading_docstring(body_lines)

    body = "\n".join(body_lines).rstrip("\n")
    if not body.strip():
        # 兜底：至少给个 pass，评测会判 FAIL，但不会让执行器崩
        body = "    pass"
    return body + "\n"


def _strip_leading_docstring(body_lines: list) -> list:
    """跳过函数体开头的（缩进）docstring，返回剩余实现行。

    只处理紧跟在开头的一段 \"\"\" 或 ''' 三引号字符串（可能单行也可能多行）；
    没有 docstring 则原样返回。
    """
    # 找到第一处非空行
    i = 0
    while i < len(body_lines) and not body_lines[i].strip():
        i += 1
    if i >= len(body_lines):
        return body_lines
    stripped = body_lines[i].lstrip()
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            rest = stripped[len(quote):]
            # 单行 docstring：同一行内闭合
            if rest.rstrip().endswith(quote) and len(rest.rstrip()) >= len(quote):
                return body_lines[i + 1:]
            # 多行 docstring：向后找闭合行
            for j in range(i + 1, len(body_lines)):
                if quote in body_lines[j]:
                    return body_lines[j + 1:]
            # 没找到闭合，保守起见原样返回
            return body_lines
    return body_lines


def generate_one(client, model, problem, max_tokens, timeout_s):
    """对单道题请求一次贪心补全，返回 sample dict。"""
    prompt = build_prompt(problem)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout_s,
        # 关闭 Qwen3 思考链，输出更短更快更稳定
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = resp.choices[0].message.content or ""
    completion = sanitize_completion(raw, problem)
    return {
        "task_id": problem["task_id"],
        "prompt": problem["prompt"],
        "completion": completion,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="HumanEval 贪心补全生成器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--data", default=DEFAULT_DATA, help="HumanEval.jsonl.gz 路径")
    ap.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "logs", "humaneval_samples.jsonl"),
        help="生成结果输出路径",
    )
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=600, help="单题请求超时（秒）")
    ap.add_argument(
        "--concurrency", type=int, default=4,
        help="并发请求数（服务端可并行处理多条请求，能显著加速）",
    )
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("[gen] 需要 openai 客户端：pip install openai", file=sys.stderr)
        return 1

    problems = load_problems(args.data)
    if args.limit > 0:
        problems = problems[: args.limit]
        print(f"[gen] --limit={args.limit}，只跑前 {len(problems)} 题")

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    print(f"[gen] 服务地址: {base_url}  模型: {args.model}  并发: {args.concurrency}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    results = {}
    t0 = time.time()
    done = 0
    total = len(problems)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(
                generate_one, client, args.model, p, args.max_tokens, args.timeout
            ): p
            for p in problems
        }
        for fut in concurrent.futures.as_completed(futs):
            p = futs[fut]
            try:
                results[p["task_id"]] = fut.result()
            except Exception as e:  # 单题失败不影响整体；记一个空补全，评测判 FAIL
                print(f"[gen] {p['task_id']} 请求失败: {e!r}", file=sys.stderr)
                results[p["task_id"]] = {
                    "task_id": p["task_id"],
                    "prompt": p["prompt"],
                    "completion": "    pass\n",
                }
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"[gen] 进度 {done}/{total}  用时 {elapsed:.0f}s")

    # 按原始题序写出，便于 diff / 复现
    with open(args.output, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(results[p["task_id"]], ensure_ascii=False) + "\n")

    print(f"[gen] 完成：{len(problems)} 条补全已写入 {args.output}")
    print(f"[gen] 下一步：python scripts/humaneval_evaluate.py --samples {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
