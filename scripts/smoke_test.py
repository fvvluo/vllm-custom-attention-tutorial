#!/usr/bin/env python
"""
smoke_test.py —— 验证 vLLM OpenAI 兼容服务是否正常、且回答“正确”。

用法：
    python scripts/smoke_test.py [--port 8000] [--model qwen3-32b]

会等待服务的 /v1/models 就绪，然后发一条**答案已知**的算术题，
并校验回答里是否包含正确答案。

为什么这样设计（重要）：
    只检查“回答非空”并不足以发现后端 bug —— 一个 KV cache 写错位置的
    attention backend 也能返回“非空但是乱码”的文本。所以这里用一道
    确定答案的题（17 + 25 = 42）来做**正确性**校验：只有真正算对了才算 PASS。
    这样学生在另一台机器上跑本脚本，就能一眼确认后端是否真的接对了。

    另外用 chat_template_kwargs.enable_thinking=false 关闭 Qwen3 的思考链，
    让回答简短、快速、确定，避免 eager + 教学版 Triton kernel 下长思考耗时过久。
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
import json


def wait_ready(base_url: str, timeout_s: int = 1800) -> None:
    """轮询 /v1/models，直到服务就绪或超时。"""
    url = f"{base_url}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[smoke] 服务已就绪: {url}")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    raise TimeoutError(f"等待服务就绪超时（{timeout_s}s）: {url}")


def chat(base_url: str, model: str, prompt: str, timeout_s: int = 300) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        # 关闭 Qwen3 思考链：回答简短、确定、快。
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="qwen3-32b")
    # 一道答案确定的题：正确答案里必须出现 "42"。
    ap.add_argument(
        "--prompt",
        default="What is 17 + 25? Reply with only the number, nothing else.",
    )
    ap.add_argument(
        "--expect",
        default="42",
        help="回答中必须包含的子串（正确性校验）；传空字符串则只检查非空。",
    )
    ap.add_argument("--timeout", type=int, default=300, help="单次请求超时（秒）")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    wait_ready(base_url)

    print(f"[smoke] 发送请求，prompt={args.prompt!r}")
    answer = chat(base_url, args.model, args.prompt, timeout_s=args.timeout)
    print("=" * 60)
    print("模型回答：")
    print(answer)
    print("=" * 60)

    if not answer or not answer.strip():
        print("[smoke] FAIL：回答为空")
        return 1
    if args.expect and args.expect not in answer:
        print(
            f"[smoke] FAIL：回答中未包含期望答案 {args.expect!r}。"
            f"这通常意味着 attention backend 计算不正确（例如 KV cache 写/读位置错位），"
            f"服务能启动但输出是错的。"
        )
        return 1
    print(f"[smoke] PASS：服务正常，且回答包含正确答案 {args.expect!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
