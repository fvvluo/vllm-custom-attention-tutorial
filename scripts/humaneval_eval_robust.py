#!/usr/bin/env python
"""Robust + deterministic HumanEval pass@1 driver.

为什么需要它（相比官方 humaneval_evaluate.py）：
  1) 官方脚本用**共享** ProcessPoolExecutor：某题触发 CPU rlimit 被 SIGKILL 会让池里
     一个 worker 猝死 → BrokenProcessPool → 整轮评测崩溃中断。
  2) 本脚本改用 humaneval_evaluate.run_one：**每题 fork 一个全新子进程**，子进程被杀不
     影响后续题。且**串行执行**——run_one 依赖 SIGALRM 超时，SIGALRM 只在主线程有效，
     用线程/进程池并发驱动会让子进程 alarm 偶发失效、join 超时 → 非确定性假 FAIL
     （实测同份补全并发跑分数在 123~143 间抖动）。串行下结果与“直接 exec”ground truth
     一致、可复现。164 题各 <1s，串行总耗时可接受。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import humaneval_evaluate as ev  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--data", default=ev.DEFAULT_DATA)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--cpu-limit", type=int, default=10)
    ap.add_argument("--mem-limit-mb", type=int, default=4096)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    problems = ev.load_problems(args.data)
    with open(args.samples, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"[eval] 载入 {len(samples)} 条补全，题库 {len(problems)} 题")

    mem_bytes = args.mem_limit_mb * 1024 * 1024 if args.mem_limit_mb > 0 else 0

    # 串行（一次只 fork 一个子进程）评测。**必须串行**：run_one 用 fork + SIGALRM，
    # 而 SIGALRM 只在主线程有效、fork-from-multithreaded 又有已知隐患——若用线程池并发
    # 驱动 run_one，子进程的 alarm 偶发失效 → join 超时 → **非确定性假 FAIL**（实测同一份
    # 补全并发跑分数在 123~143 间抖动）。串行下每题结果与“直接 exec”的 ground truth 一致、
    # 可复现。164 题各 <1s，串行总耗时可接受。
    results = {}
    for i, s in enumerate(samples, 1):
        tid = s["task_id"]
        ok = ev.run_one(
            problems[tid], s["completion"],
            args.timeout, args.cpu_limit, mem_bytes,
        )
        results[tid] = ok
        print(f"[eval] {i:>3}/{len(samples)}  {tid:<16} {'PASS' if ok else 'FAIL'}")

    n_pass = sum(1 for ok in results.values() if ok)
    n_total = len(results)
    pass_at_1 = n_pass / n_total if n_total else 0.0

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            for s in samples:
                tid = s["task_id"]
                f.write(json.dumps({"task_id": tid, "passed": results[tid]}) + "\n")
        print(f"[eval] 逐题结果已写入 {args.report}")

    print("=" * 50)
    print(f"pass@1 = {n_pass}/{n_total} = {pass_at_1:.4f}  ({pass_at_1 * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
