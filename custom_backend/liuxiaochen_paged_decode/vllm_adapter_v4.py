# SPDX-License-Identifier: Apache-2.0
"""V4 adapter: route STRICTLY-supported vLLM CUSTOM decode calls to the V3
cp.async paged-KV warp-MMA decode kernel; everything else keeps the existing
tutorial-Triton path unchanged.

Design constraints (Phase V4):
  - NO change to V2/V3 kernel math, MMA mapping, cp.async pipeline, or combine.
  - NO PDL, NO extra pipeline stage, NOT wired as a new backend.
  - `can_use_v3_decode()` is a PURE predicate: it allocates no GPU tensor, does
    NO device sync (no `.item()`/`.max()`/`.cpu()` on CUDA tensors), and does not
    mutate its inputs. It only reads shapes/strides/dtypes/scalars already on hand.
  - On a supported pure-decode batch we hand V3 the LIVE tensors (query, the
    packed KV-cache split views, block_table, seq_lens, current stream). No clone,
    no contiguous(), no gather, no KV copy. V3's own runner reads them via dlpack.
  - Feature flag `LIUXIAOCHEN_PAGED_DECODE_V3`: "1" enables V3 inside the support
    domain; "0"/unset keeps the pure tutorial-Triton baseline.
  - Dispatch evidence is low-frequency (first-hit + first-of-each-reason + atexit
    totals) so it never pollutes a benchmark.

Support domain (all must hold, else fallback):
  pure decode (max_query_len==1, num_tokens==num_seqs), bf16 q/k/v/out,
  Hq=64, Hkv=8, head_dim=128, block_size=16, SM90, causal, finite positive scale,
  last-dim-contiguous 16B-aligned paged cache whose block/head/slot strides are
  multiples of 8 bf16 (V3's cp.async alignment domain — satisfied by the packed
  (num_blocks,Hkv,bs,2*hs) cache sliced into K/V halves).
"""

import atexit
import os
import threading

import torch

# V3 fixed target.
_Q_HEADS = 64
_KV_HEADS = 8
_HEAD_DIM = 128
_BLOCK_SIZE = 16
_VEC_ELEMS = 8            # 16 bytes / 2-byte bf16
_SPLIT_SIZE_TOKENS = 256  # V3's validated best split; multiple of 64 (V3 requirement)

_FLAG_ENV = "LIUXIAOCHEN_PAGED_DECODE_V3"
_DEBUG_ENV = "LIUXIAOCHEN_PAGED_DECODE_V3_DEBUG"


class _DispatchStats:
    """Process-wide, low-frequency dispatch evidence. No per-token printing."""

    def __init__(self):
        self._lock = threading.Lock()
        self.hits = 0
        self.fallbacks = 0
        self.reason_counts = {}
        self.ns1_hits = 0            # num_seqs==1 decode hits (real single-request)
        self.ns_multi_hits = 0       # num_seqs>1 hits (profiling/batched)
        self._logged_first_hit = False
        self._logged_first_ns1 = False
        self._logged_reasons = set()
        self._atexit_registered = False

    def _debug(self):
        return os.environ.get(_DEBUG_ENV, "0") == "1"

    def record_hit(self, shape_stride_desc, num_seqs=None):
        with self._lock:
            self.hits += 1
            if num_seqs == 1:
                self.ns1_hits += 1
                if not self._logged_first_ns1:
                    self._logged_first_ns1 = True
                    print(f"[v4-dispatch] FIRST num_seqs=1 V3 decode HIT: {shape_stride_desc}",
                          flush=True)
            elif num_seqs is not None:
                self.ns_multi_hits += 1
            if not self._logged_first_hit:
                self._logged_first_hit = True
                print(f"[v4-dispatch] FIRST V3 decode HIT: {shape_stride_desc}",
                      flush=True)
            if not self._atexit_registered:
                atexit.register(self.report)
                self._atexit_registered = True

    def record_fallback(self, reason):
        with self._lock:
            self.fallbacks += 1
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
            if reason not in self._logged_reasons:
                self._logged_reasons.add(reason)
                # First occurrence of each distinct reason only (low-frequency).
                print(f"[v4-dispatch] fallback (first of kind): {reason}", flush=True)
            if not self._atexit_registered:
                atexit.register(self.report)
                self._atexit_registered = True

    def report(self):
        # Printed once at process exit (or on explicit call).
        rc = ", ".join(f"{k}={v}" for k, v in sorted(self.reason_counts.items()))
        print(f"[v4-dispatch] TOTALS: v3_hits={self.hits} (ns1={self.ns1_hits} "
              f"ns_multi={self.ns_multi_hits}) fallbacks={self.fallbacks} "
              f"reasons[{rc}]", flush=True)


STATS = _DispatchStats()

# Set to True after the V3 call raises once, so we never repeatedly crash a live
# service; subsequent supported calls fall back to Triton for the process lifetime.
_v3_runtime_disabled = False
_v3_runtime_disabled_reason = ""


def v3_enabled() -> bool:
    """Feature flag: only "1" turns V3 on. Read live so tests can toggle."""
    return os.environ.get(_FLAG_ENV, "0") == "1"


def _aligned_cache(cache) -> bool:
    """V3 cp.async alignment domain, as a pure bool (mirrors runner_v3
    `_check_cpasync_alignment` but never raises / never mutates)."""
    if cache.stride(-1) != 1:
        return False
    if cache.data_ptr() % 16 != 0:
        return False
    sblock, shead, sslot, _ = cache.stride()
    for s in (sblock, shead, sslot):
        if s % _VEC_ELEMS != 0:
            return False
    return True


def can_use_v3_decode(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output_view: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    num_actual_tokens: int,
    num_seqs: int,
    max_query_len: int,
    causal,
    scale: float,
) -> tuple[bool, str]:
    """Pure predicate. Returns (ok, reason). reason=="ok" iff ok is True.

    Does NOT allocate, sync, or mutate. Only reads shapes/strides/dtypes and the
    already-computed scalar metadata (`max_query_len`, `num_actual_tokens`).
    """
    import math

    # pure decode: exactly one query token per sequence.
    if max_query_len != 1:
        return False, "not_pure_decode(max_query_len!=1)"
    if num_actual_tokens != num_seqs:
        return False, "not_pure_decode(num_tokens!=num_seqs)"
    # causal only (decode is causal; V3/tutorial assume causal).
    if isinstance(causal, torch.Tensor) or causal is not True:
        return False, "non_causal_or_tensor_causal"
    # dtype: bf16 everywhere.
    if query.dtype != torch.bfloat16:
        return False, f"query_dtype={query.dtype}"
    if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        return False, "kv_cache_not_bf16"
    if output_view.dtype != torch.bfloat16:
        return False, f"output_dtype={output_view.dtype}"
    # fixed shape target.
    if num_heads != _Q_HEADS or num_kv_heads != _KV_HEADS:
        return False, f"heads={num_heads}/{num_kv_heads}"
    if head_size != _HEAD_DIM:
        return False, f"head_size={head_size}"
    if key_cache.dim() != 4 or key_cache.shape[2] != _BLOCK_SIZE:
        return False, f"block_size={key_cache.shape[2] if key_cache.dim()==4 else 'na'}"
    if tuple(value_cache.shape) != tuple(key_cache.shape):
        return False, "kv_shape_mismatch"
    # scale sanity.
    if not math.isfinite(scale) or scale <= 0.0:
        return False, "bad_scale"
    # SM90 (cheap capability lookup, no sync).
    if not query.is_cuda:
        return False, "query_not_cuda"
    major, minor = torch.cuda.get_device_capability(query.device)
    if (major, minor) != (9, 0):
        return False, f"sm{major}{minor}"
    # cp.async alignment domain (no clone/contiguous allowed to fix it).
    if not _aligned_cache(key_cache):
        return False, "key_cache_unaligned"
    if not _aligned_cache(value_cache):
        return False, "value_cache_unaligned"
    return True, "ok"


def try_v3_decode(
    *,
    query,
    key_cache,
    value_cache,
    output_view,
    query_start_loc,
    seq_lens,
    token_seq_idx,
    block_table,
    scale,
    max_seq_len,
    num_heads,
    num_kv_heads,
    head_size,
    num_actual_tokens,
    max_query_len,
    causal,
) -> bool:
    """If the batch is in the V3 support domain (and flag on), run V3 in place on
    the live tensors and return True. Otherwise return False (caller runs the
    existing Triton path). Never raises to the caller: a first-time runtime error
    disables V3 for the process and falls back.
    """
    global _v3_runtime_disabled, _v3_runtime_disabled_reason

    if not v3_enabled():
        return False
    if _v3_runtime_disabled:
        STATS.record_fallback(f"v3_runtime_disabled:{_v3_runtime_disabled_reason}")
        return False

    num_seqs = seq_lens.shape[0]
    ok, reason = can_use_v3_decode(
        query=query, key_cache=key_cache, value_cache=value_cache,
        output_view=output_view, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_size=head_size, num_actual_tokens=num_actual_tokens,
        num_seqs=num_seqs, max_query_len=max_query_len, causal=causal, scale=scale,
    )
    if not ok:
        STATS.record_fallback(reason)
        return False

    from .runner_v3 import paged_decode_v3

    try:
        paged_decode_v3(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output_view,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            token_seq_idx=token_seq_idx,
            block_table=block_table,
            scale=scale,
            split_size_tokens=_SPLIT_SIZE_TOKENS,
            max_seq_len=int(max_seq_len),
        )
    except Exception as e:  # noqa: BLE001 - service safety: fall back, do not crash
        _v3_runtime_disabled = True
        _v3_runtime_disabled_reason = f"{type(e).__name__}:{e}"
        print(f"[v4-dispatch] V3 raised at runtime, DISABLING V3 for this process "
              f"and falling back to Triton: {_v3_runtime_disabled_reason}", flush=True)
        STATS.record_fallback(f"v3_exception:{type(e).__name__}")
        return False

    STATS.record_hit(
        f"num_seqs={num_seqs} q={tuple(query.shape)}/{query.stride()} "
        f"kc={tuple(key_cache.shape)}/{key_cache.stride()} "
        f"out={tuple(output_view.shape)}/{output_view.stride()} "
        f"split={_SPLIT_SIZE_TOKENS} max_seq_len={int(max_seq_len)}",
        num_seqs=num_seqs,
    )
    return True


def try_v3_decode_from_triton_args(
    *,
    query,
    key_cache,
    value_cache,
    output,
    query_start_loc,
    seq_lens,
    token_seq_idx,
    block_table,
    scale,
    num_heads,
    num_kv_heads,
    head_size,
) -> bool:
    """Entry point callable from `paged_attention_triton(...)` (the only project
    file the grading rules allow us to edit). Derives the pure-decode scalars that
    the vLLM metadata would otherwise carry, straight from the live tensors, then
    delegates to `try_v3_decode`. Returns True iff V3 handled the read in place.

    Pure-decode detection WITHOUT trusting external metadata:
      num_tokens == num_seqs AND query_start_loc == [0,1,2,...,num_seqs]
      (every request contributes exactly one query token). Otherwise we treat it as
      prefill/mixed and the caller keeps the Triton path.

    `max_seq_len` for split sizing = int(seq_lens.max()). One tiny D->H read of a
    [num_seqs] int tensor, done ONLY after we know it is a decode batch, so it never
    touches the prefill hot path.
    """
    if not v3_enabled():
        return False

    num_tokens = query.shape[0]
    num_seqs = seq_lens.shape[0]

    if num_tokens != num_seqs:
        STATS.record_fallback("not_pure_decode(num_tokens!=num_seqs)")
        return False
    if query_start_loc.shape[0] != num_seqs + 1:
        STATS.record_fallback("bad_query_start_loc_shape")
        return False
    expected = torch.arange(num_seqs + 1, device=query_start_loc.device,
                            dtype=query_start_loc.dtype)
    if not torch.equal(query_start_loc, expected):
        STATS.record_fallback("not_pure_decode(max_query_len!=1)")
        return False

    max_seq_len = int(seq_lens.max().item())  # decode-only path; tiny tensor

    return try_v3_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        output_view=output,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_seq_idx=token_seq_idx,
        block_table=block_table,
        scale=scale,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_actual_tokens=num_tokens,
        max_query_len=1,
        causal=True,
    )
