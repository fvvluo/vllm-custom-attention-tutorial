#!/usr/bin/env python
"""
humaneval_evaluate.py —— 在沙箱里执行 HumanEval 补全，计算 pass@1（贪心）。
=============================================================================

它做什么：
  1) 读取 humaneval_generate.py 生成的 samples.jsonl（每行 {task_id, prompt, completion}）；
  2) 对每道题，拼出可执行程序：
         problem["prompt"] + sample["completion"] + problem["test"] + f"check({entry_point})"
     （prompt 里含函数签名，completion 是函数体，test 里含 check() 单测）；
  3) 把这段程序丢进**独立子进程**执行，带超时；子进程正常退出(0) => 该题通过；
  4) pass@1 = 通过题数 / 总题数（贪心解码下每题仅 1 个补全）。

安全（务必看）：
  模型生成的代码是**不可信**的，评测会真实执行它。本脚本用以下方式隔离：
    - 每题在**单独子进程**里跑，超时(默认 10s)即 SIGKILL，防死循环/挂起；
    - 子进程内通过 resource 限制 CPU 时间与地址空间（Linux 上生效），
      降低 fork 炸弹 / 内存爆炸的影响；
    - OOM 兜底：worker 内捕获 MemoryError（rlimit 触发）仅判该题失败；SIGXCPU
      转成超时判负，避免 CPU 超限直接杀死 worker；若 worker 仍被硬杀（如
      内核 OOM killer 的 SIGKILL，BrokenProcessPool），未完成的题逐题重试
      精确定位肇事题判 FAIL——一题内存爆炸/自毁不会拖垮整轮评测；
    - 关闭子进程对父进程内存的影响（独立解释器）。
  但这**不是**强隔离沙箱。生产/大规模评测请在容器或 gVisor/nsjail 等隔离环境中运行。
  这也是 README 里反复强调"务必在隔离环境中执行"的原因。

用法：
    python scripts/humaneval_evaluate.py \
        --samples logs/humaneval_samples.jsonl

    # 也可指定数据集路径（默认与 generate 相同的缓存）
    python scripts/humaneval_evaluate.py \
        --samples logs/humaneval_samples.jsonl \
        --data data/HumanEval.jsonl.gz \
        --timeout 10 --workers 8
"""
import argparse
import concurrent.futures
import gzip
import json
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_DATA = os.path.join(REPO_ROOT, "data", "HumanEval.jsonl.gz")


def load_problems(data_path: str) -> dict:
    if not os.path.exists(data_path):
        print(
            f"[eval] 找不到数据集: {data_path}\n"
            f"       请先运行 humaneval_generate.py（会自动下载缓存），"
            f"或手动放置 HumanEval.jsonl.gz。",
            file=sys.stderr,
        )
        sys.exit(1)
    problems = {}
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                problems[p["task_id"]] = p
    return problems


class _Timeout(Exception):
    pass


def _run_in_this_process(program: str, timeout_s: int,
                         cpu_limit: int, mem_limit_bytes: int) -> bool:
    """在【当前进程】里执行拼好的程序，带 SIGALRM 超时 + 资源上限。

    本函数被设计为在 ProcessPoolExecutor 的 worker 子进程里调用：每个 worker 是
    独立进程，因此这里的 rlimit / SIGALRM / exec 都不会影响主进程或其它任务。
    """
    # 资源上限（Linux 生效），给失控/恶意代码兜底
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
        if mem_limit_bytes > 0:
            resource.setrlimit(
                resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes)
            )
    except Exception:
        pass

    # 屏蔽 exit/quit，避免候选代码直接把 worker 干掉
    import builtins

    for name in ("exit", "quit"):
        setattr(builtins, name, lambda *a, **k: None)

    def _on_alarm(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _on_alarm)
    # SIGXCPU 兜底：RLIMIT_CPU 软限触发时默认动作是直接杀死进程（会拖垮
    # ProcessPoolExecutor）。转成 _Timeout 让 worker 干净地判 FAIL 存活下来；
    # 只有 handler 失效时的硬限（或 OOM killer 的 SIGKILL）才会真正杀死 worker。
    old_xcpu = None
    if hasattr(signal, "SIGXCPU"):
        old_xcpu = signal.signal(signal.SIGXCPU, _on_alarm)
    signal.alarm(timeout_s)
    try:
        env = {"__name__": "__humaneval__"}
        exec(compile(program, "<candidate>", "exec"), env)
        return True
    except MemoryError:
        # OOM 兜底①：RLIMIT_AS 触发后 Python 抛 MemoryError。worker 进程本身
        # 存活，仅判该题失败；gc 一下避免残留大对象污染同 worker 的后续任务。
        import gc

        gc.collect()
        print("[eval] worker 捕获 MemoryError（内存超限），该题判 FAIL",
              file=sys.stderr)
        return False
    except BaseException:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        if old_xcpu is not None:
            signal.signal(signal.SIGXCPU, old_xcpu)


def _worker(args) -> tuple:
    """ProcessPoolExecutor 的任务入口（顶层函数，可被 pickle）。"""
    task_id, program, timeout_s, cpu_limit, mem_bytes = args
    try:
        ok = _run_in_this_process(program, timeout_s, cpu_limit, mem_bytes)
    except BaseException:
        ok = False
    return task_id, ok


def build_program(problem: dict, completion: str) -> str:
    """拼出可执行程序：prompt(含签名) + completion(函数体) + test + check(entry)。"""
    return (
        problem["prompt"]
        + completion
        + "\n"
        + problem["test"]
        + f"\ncheck({problem['entry_point']})\n"
    )


def run_one(problem: dict, completion: str, timeout_s: int,
            cpu_limit: int, mem_limit_bytes: int) -> bool:
    """在独立 fork 子进程里执行单题并判定通过与否（供单测/交互式调用）。

    用裸 fork（而非 ProcessPoolExecutor）以便无论本模块如何被 import 都能工作：
    fork 会直接继承已定义的 _run_in_this_process，不需要按模块名重新 import。
    """
    import multiprocessing as mp

    program = build_program(problem, completion)
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    def _child():
        ok = False
        try:
            ok = _run_in_this_process(program, timeout_s, cpu_limit, mem_limit_bytes)
        except BaseException:
            ok = False
        child_conn.send(ok)

    proc = ctx.Process(target=_child)
    proc.start()
    proc.join(timeout_s + cpu_limit + 5)  # 兜底 join 超时，防极端挂起
    if proc.is_alive():
        proc.kill()
        proc.join()
        return False
    return bool(parent_conn.recv()) if parent_conn.poll() else False


def main() -> int:
    ap = argparse.ArgumentParser(description="HumanEval 沙箱评测，算 pass@1")
    ap.add_argument(
        "--samples",
        default=os.path.join(REPO_ROOT, "logs", "humaneval_samples.jsonl"),
        help="humaneval_generate.py 生成的补全文件",
    )
    ap.add_argument("--data", default=DEFAULT_DATA, help="HumanEval.jsonl.gz 路径")
    ap.add_argument("--timeout", type=int, default=10, help="单题执行超时（秒）")
    ap.add_argument(
        "--cpu-limit", type=int, default=15,
        help="子进程 CPU 秒数上限；应大于 --timeout，让 SIGALRM 先触发、"
             "worker 得以存活判 FAIL 而不是被 SIGXCPU 直接杀死",
    )
    ap.add_argument(
        "--mem-limit-mb", type=int, default=4096,
        help="子进程地址空间上限（MB），0=不限制",
    )
    ap.add_argument(
        "--workers", type=int, default=8,
        help="并行评测的题目数（每题仍是独立子进程执行）",
    )
    ap.add_argument(
        "--report", default="",
        help="可选：把逐题结果写入该 JSONL 文件",
    )
    args = ap.parse_args()

    if not os.path.exists(args.samples):
        print(f"[eval] 找不到补全文件: {args.samples}\n"
              f"       请先运行 humaneval_generate.py。", file=sys.stderr)
        return 1

    problems = load_problems(args.data)
    with open(args.samples, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"[eval] 载入 {len(samples)} 条补全，题库 {len(problems)} 题")

    mem_bytes = args.mem_limit_mb * 1024 * 1024 if args.mem_limit_mb > 0 else 0

    # 每个任务在独立 worker 进程里执行候选代码：进程级隔离 + 可安全并行。
    # （不能用线程池——在多线程进程里 fork/exec 不可信代码既不安全，结果也可能错乱。）
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    payloads = [
        (
            s["task_id"],
            build_program(problems[s["task_id"]], s["completion"]),
            args.timeout,
            args.cpu_limit,
            mem_bytes,
        )
        for s in samples
    ]

    results: dict = {}  # task_id -> passed
    from concurrent.futures.process import BrokenProcessPool

    def _run_pool(idxs) -> set:
        """用进程池跑 idxs 里的题，返回完成的下标集合。

        worker 被硬杀（OOM killer / 资源超限 / 候选代码自毁）时 Pool 抛
        BrokenProcessPool——已出结果的题保留，未完成的由调用方决定如何重试。
        """
        done = set()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx
        ) as ex:
            fut2idx = {ex.submit(_worker, payloads[i]): i for i in idxs}
            try:
                for fut in concurrent.futures.as_completed(fut2idx):
                    i = fut2idx[fut]
                    tid, passed = fut.result()
                    results[tid] = passed
                    done.add(i)
                    mark = "PASS" if passed else "FAIL"
                    print(f"[eval] {len(results):>3}/{len(samples)}  "
                          f"{tid:<16} {mark}")
            except BrokenProcessPool:
                pass  # 由外层恢复逻辑处理
        return done

    # OOM 兜底②：先并行跑；若有题把 worker 跑死，对未完成者**逐题重试**——
    # 单题池里崩溃能精确定位肇事题，直接判 FAIL 继续，不误伤同池其它题，
    # 一道“内存炸弹/自毁”题不会拖垮整轮评测。
    pending = list(range(len(payloads)))
    done = _run_pool(pending)
    leftover = [i for i in pending if i not in done]
    if leftover:
        print(f"[eval] 评测 worker 被杀（疑似 OOM killer / 资源超限），"
              f"逐题重试 {len(leftover)} 题以定位肇事题", file=sys.stderr)
    for i in leftover:
        tid = payloads[i][0]
        done = _run_pool([i])
        if i not in done:
            results[tid] = False
            print(f"[eval] {len(results):>3}/{len(samples)}  {tid:<16} "
                  f"FAIL (worker 被杀，疑似 OOM/资源超限)")

    n_pass = sum(1 for ok in results.values() if ok)
    n_total = len(results)
    pass_at_1 = n_pass / n_total if n_total else 0.0

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            for s in samples:
                tid = s["task_id"]
                f.write(json.dumps(
                    {"task_id": tid, "passed": bool(results.get(tid, False))}
                ) + "\n")
        print(f"[eval] 逐题结果已写入 {args.report}")

    print("=" * 50)
    print(f"pass@1 = {n_pass}/{n_total} = {pass_at_1:.4f}  ({pass_at_1 * 100:.2f}%)")
    if n_total < len(problems):
        print(f"（注意：本次只评测了 {n_total}/{len(problems)} 题，非完整 164 题）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
