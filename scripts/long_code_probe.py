#!/usr/bin/env python
"""
long_code_probe.py — send a LONG real-code prompt to a vLLM server and report
first-token latency + a coherence check on the continuation.

Purpose: exercise the wzc sparse backend on long code sequences. Pair with
WZC_SPARSE_STATS=1 on the server to confirm the sparse kernel actually fires
(the server log prints [wzc-stats] with kernel_reqs / max_kernel_seq).

Builds a long prompt from real Python source (asks the model to continue a
large module), padded to ~--input-len tokens using the model tokenizer, then a
unique marker so prefix caching can't skip the prefill.
"""
import argparse
import time
import uuid

from openai import OpenAI


CODE_SEED = '''
import math
from dataclasses import dataclass


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def dot(self, o: "Vec3") -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def norm(self) -> float:
        return math.sqrt(self.dot(self))


def quicksort(a):
    if len(a) <= 1:
        return a
    p = a[len(a) // 2]
    lo = [x for x in a if x < p]
    eq = [x for x in a if x == p]
    hi = [x for x in a if x > p]
    return quicksort(lo) + eq + quicksort(hi)


def binary_search(a, target):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == target:
            return mid
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''


def build_prompt(tokenizer, target_tokens):
    marker = f"# session {uuid.uuid4().hex}\n"
    header = ("You are reading a large Python module. After it ends, continue "
              "by writing a new function `def summarize(items):` that returns a "
              "dict with count/min/max/mean of a numeric list. Reply with ONLY a "
              "```python code block.\n\n")
    body = marker + header + "```python\n"
    # repeat the seed until we approach the target token budget
    if tokenizer is not None:
        base = len(tokenizer(body)["input_ids"])
        seed_tok = len(tokenizer(CODE_SEED)["input_ids"])
        reps = max(1, (target_tokens - base - 50) // max(1, seed_tok))
        body += CODE_SEED * reps + "\n```\n"
        n = len(tokenizer(body)["input_ids"])
    else:
        reps = max(1, target_tokens // 400)
        body += CODE_SEED * reps + "\n```\n"
        n = None
    return body, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8006)
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=96)
    ap.add_argument("--model-path", default="/dockerdata/models/Qwen3-32B")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] tokenizer unavailable ({e!r}); estimating by chars")
        tok = None

    prompt, n = build_prompt(tok, args.input_len)
    print(f"[probe] target input ~{args.input_len} tok; actual ~{n} tok"
          if n else f"[probe] target ~{args.input_len} tok (char-estimated)")

    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="EMPTY")

    t0 = time.time()
    ttft = None
    chunks = []
    stream = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=args.output_len,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for ev in stream:
        delta = ev.choices[0].delta.content if ev.choices else None
        if delta:
            if ttft is None:
                ttft = time.time() - t0
            chunks.append(delta)
    total = time.time() - t0
    text = "".join(chunks)
    out_tok = len(tok(text)["input_ids"]) if tok else len(text.split())
    decode_tps = (out_tok - 1) / (total - ttft) if ttft and total > ttft else 0.0

    print("=" * 60)
    print(f"TTFT           : {ttft:.3f} s   <- long-prompt prefill latency")
    print(f"decode tput    : {decode_tps:.1f} tok/s")
    print(f"total          : {total:.3f} s   out~{out_tok} tok")
    print("=" * 60)
    # coherence: did it produce a python code block mentioning summarize?
    ok = ("```" in text) and ("def summarize" in text or "summarize" in text)
    print("[probe] continuation preview:")
    print(text[:400])
    print("=" * 60)
    print(f"[probe] coherence {'PASS' if ok else 'CHECK'} "
          f"(code block + summarize present={ok})")


if __name__ == "__main__":
    main()
