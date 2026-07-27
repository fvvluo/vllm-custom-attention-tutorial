#!/usr/bin/env python
"""
smoke_test.py —— 验证 vLLM OpenAI 兼容服务是否正常、回答是否合理。

用法：
    python scripts/smoke_test.py [--port 8000] [--model qwen3-32b]

会等待服务的 /v1/models 就绪，然后发一条 chat 请求并打印回答。
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


def chat(base_url: str, model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--prompt", default="用一句话解释什么是注意力机制（attention）。")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    wait_ready(base_url)

    print(f"[smoke] 发送请求，prompt={args.prompt!r}")
    answer = chat(base_url, args.model, args.prompt)
    print("=" * 60)
    print("模型回答：")
    print(answer)
    print("=" * 60)

    if not answer or not answer.strip():
        print("[smoke] FAIL：回答为空")
        return 1
    print("[smoke] PASS：服务正常且返回了非空回答")
    return 0


if __name__ == "__main__":
    sys.exit(main())
