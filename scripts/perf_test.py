#!/usr/bin/env python
"""
perf_test.py —— vLLM OpenAI 兼容服务的长上下文性能测试。
=============================================================================

它测什么：
  给服务发一条**指定长度**的输入（默认 100k token，贴近 Qwen3 的 128k 上下文上限），
  用流式(stream=True)接口测两个关键指标：
    - TTFT (Time To First Token，首 token 延迟)：主要反映 **prefill**（处理长输入）耗时；
    - decode 吞吐 (tokens/s)：首 token 之后的**逐 token 生成**速度。
  另外给出端到端总时延、实际生成 token 数。

为什么用 100k 输入：
  Qwen3-32B 上下文 128k。用 ~100k 的输入能压到接近上限，充分暴露 attention 后端在
  **长序列 prefill** 上的性能差异——这正是不同 attention kernel（flash_attn vs 你自己的
  kernel）拉开差距的地方。

怎么精确控制输入长度：
  用 Qwen3-32B 的 tokenizer 把一段重复文本编码/截断到**精确**的目标 token 数，
  作为一条 user message 发出。若 transformers/tokenizer 不可用，则退化为按字符比例
  拼接（不精确，会提示你用 --input-len 校准）。

用法：
    # 先起服务（见 README Part 4）。flash_attn 跑完整 100k：
    #   MAX_LEN=110000 GPU=1 bash scripts/serve_qwen3_flashattn.sh
    python scripts/perf_test.py --input-len 100000 --output-len 64

    # CUSTOM 后端只在小长度上对比（朴素 kernel 无法实用化跑 100k）：
    python scripts/perf_test.py --input-len 2048 --output-len 32

依赖：openai 客户端；（可选）transformers 用于精确控长。
"""
import argparse
import statistics
import sys
import time

MODEL_PATH = "/dockerdata/models/Qwen3-32B"

# 一段中性的重复语料，用于把输入撑到目标长度（内容无关紧要，只为占满上下文）。
_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Large language models process long contexts by attending over many tokens. "
)


def build_long_prompt(target_tokens: int):
    """构造 token 数≈target_tokens 的输入。返回 (prompt_str, actual_tokens, exact)。

    主路径用 Qwen3 tokenizer 精确控长；tokenizer 不可用时退化按字符比例（不精确）。
    """
    # 结尾追加一个明确的简短指令，避免模型对着长文本长篇大论——我们只关心速度。
    instruction = "\n\nSummarize the above in one short sentence."
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        instr_ids = tok.encode(instruction, add_special_tokens=False)
        budget = max(1, target_tokens - len(instr_ids))
        filler_ids = tok.encode(_FILLER, add_special_tokens=False)
        if not filler_ids:
            filler_ids = tok.encode("token ", add_special_tokens=False)
        # 重复 filler 到超过 budget，再精确截断到 budget
        reps = budget // len(filler_ids) + 1
        body_ids = (filler_ids * reps)[:budget]
        text = tok.decode(body_ids) + instruction
        # 重新编码核实真实长度（decode/encode 可能有 ±few token 偏差）
        actual = len(tok.encode(text, add_special_tokens=False))
        return text, actual, True
    except Exception as e:
        print(f"[perf] tokenizer 不可用（{e!r}），退化为按字符估算长度（不精确）",
              file=sys.stderr)
        # 粗略经验：英文约 4 字符/token
        approx_chars = target_tokens * 4
        reps = approx_chars // len(_FILLER) + 1
        text = (_FILLER * reps)[:approx_chars] + instruction
        return text, target_tokens, False


def measure_once(client, model, prompt, output_len, timeout_s, unique_prefix=""):
    """发一条流式请求，返回 (ttft_s, decode_tps, total_s, n_out_tokens)。

    - ttft：从发出请求到收到第一个内容 chunk 的时间；
    - decode_tps：(生成 token 数 - 1) / (最后一个 token 时刻 - 第一个 token 时刻)；
      只统计首 token 之后的稳定生成阶段，避免把 prefill 时间算进吞吐。
    - unique_prefix：给每次请求加一段**唯一**前缀，避免 vLLM 的 prefix caching 命中
      导致 prefill 被跳过（那样 TTFT 会假性偏低，测不出真实的长输入 prefill 耗时）。
    """
    content = (unique_prefix + prompt) if unique_prefix else prompt
    t0 = time.perf_counter()
    first_t = None
    last_t = None
    n_chunks = 0
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=output_len,
        timeout=timeout_s,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if not piece:
            continue
        now = time.perf_counter()
        if first_t is None:
            first_t = now
        last_t = now
        n_chunks += 1

    total_s = time.perf_counter() - t0
    if first_t is None:
        # 没收到任何内容 token
        return float("nan"), 0.0, total_s, 0
    ttft = first_t - t0
    # 每个流式 chunk 近似 1 个 token；用 chunk 数作为生成 token 数的估计
    n_out = n_chunks
    gen_span = (last_t - first_t) if last_t and last_t > first_t else 0.0
    decode_tps = (n_out - 1) / gen_span if (n_out > 1 and gen_span > 0) else 0.0
    return ttft, decode_tps, total_s, n_out


def main() -> int:
    ap = argparse.ArgumentParser(description="vLLM 长上下文性能测试（TTFT + decode 吞吐）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--input-len", type=int, default=100000,
                    help="输入 token 目标长度（默认 100000，贴近 Qwen3 128k 上限）")
    ap.add_argument("--output-len", type=int, default=64,
                    help="生成 token 数上限（默认 64，聚焦 TTFT + 稳定 decode 采样）")
    ap.add_argument("--warmup", type=int, default=1, help="预热次数（不计入统计）")
    ap.add_argument("--repeat", type=int, default=3, help="正式测量次数（取中位数）")
    ap.add_argument("--timeout", type=int, default=1800, help="单次请求超时（秒）")
    ap.add_argument("--allow-prefix-cache", action="store_true",
                    help="默认每次请求加唯一前缀以绕过 prefix caching（测真实 prefill）；"
                         "加本开关则不加唯一前缀，故意测命中缓存后的 TTFT")
    args = ap.parse_args()

    import uuid

    def _uniq():
        # 默认给每次请求一段唯一前缀，避免 vLLM prefix caching 命中导致 prefill 被跳过、
        # TTFT 假性偏低。加 --allow-prefix-cache 则返回空串（不绕过缓存）。
        return "" if args.allow_prefix_cache else f"[req-{uuid.uuid4().hex}] "

    try:
        from openai import OpenAI
    except ImportError:
        print("[perf] 需要 openai 客户端：pip install openai", file=sys.stderr)
        return 1

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    print(f"[perf] 服务: {base_url}  模型: {args.model}")
    print(f"[perf] 目标输入长度: {args.input_len} tokens  生成上限: {args.output_len} tokens")
    prompt, actual_len, exact = build_long_prompt(args.input_len)
    tag = "精确" if exact else "估算"
    print(f"[perf] 实际输入长度({tag}): ~{actual_len} tokens")

    # 预热（首次含 tokenizer/compile/cache 冷启动，不计入）
    for i in range(args.warmup):
        print(f"[perf] 预热 {i + 1}/{args.warmup} ...")
        try:
            measure_once(client, args.model, prompt, args.output_len, args.timeout,
                         unique_prefix=_uniq())
        except Exception as e:
            print(f"[perf] 预热请求失败: {e!r}", file=sys.stderr)
            return 1

    ttfts, tpss, totals, nouts = [], [], [], []
    for i in range(args.repeat):
        ttft, tps, total_s, n_out = measure_once(
            client, args.model, prompt, args.output_len, args.timeout,
            unique_prefix=_uniq()
        )
        ttfts.append(ttft)
        tpss.append(tps)
        totals.append(total_s)
        nouts.append(n_out)
        print(f"[perf] 第 {i + 1}/{args.repeat} 次: "
              f"TTFT={ttft:.3f}s  decode={tps:.1f} tok/s  "
              f"total={total_s:.3f}s  out={n_out} tok")

    med_ttft = statistics.median(ttfts)
    med_tps = statistics.median(tpss)
    med_total = statistics.median(totals)
    med_out = int(statistics.median(nouts))

    print("=" * 64)
    print(f"输入长度        : ~{actual_len} tokens ({tag})")
    print(f"生成长度        : {med_out} tokens (上限 {args.output_len})")
    print(f"TTFT (中位数)   : {med_ttft:.3f} s   <- 主要是 prefill 长输入的耗时")
    print(f"decode 吞吐     : {med_tps:.1f} tokens/s   <- 首 token 之后的生成速度")
    print(f"端到端总时延    : {med_total:.3f} s")
    print("=" * 64)
    # 一行机器可读汇总，便于把两个后端结果对比
    print(f"[perf] SUMMARY input={actual_len} out={med_out} "
          f"ttft_s={med_ttft:.3f} decode_tps={med_tps:.1f} total_s={med_total:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
