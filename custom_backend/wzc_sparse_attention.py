# SPDX-License-Identifier: Apache-2.0
"""
Adapter: wzc block-level top-k SPARSE prefill kernel -> vLLM paged-attention API.
=================================================================================

Exposes ``paged_attention_wzc(...)`` with the SAME signature/semantics as the
tutorial's ``paged_attention_triton`` (see custom_backend/triton_attention.py and
README Part 3), so it drops into ``CustomTritonImpl.forward`` unchanged.

Background / why an adapter is needed
-------------------------------------
The wzc sparse kernel (``ops/_wzc_attn_sparse.py`` in the attention-test
repo) is a **dense-layout, square-causal PREFILL** kernel:
    run(Q, K, V, causal=True, sm_scale, tau, ...)
        Q : (B, H, S, D)   K/V : (B, HK, S, D)   causal, q_len == kv_len,
        requires S % 128 == 0 and head_dim == 128, bf16/fp16.
vLLM instead hands attention a **paged, variable-length, flattened** batch:
    query      : [num_tokens, num_heads, hd]
    key/value  : paged cache [num_blocks, num_kv_heads, block_size, hd]
    + query_start_loc / seq_lens / block_table / token_seq_idx
and mixes prefill (q_len == seq_len) with decode (q_len == 1, seq_len > q_len).

This adapter bridges the two, PER REQUEST:
  * gather that request's contiguous K/V (positions [0, seq_len)) out of the
    paged cache using ``block_table`` (stride-correct, matching the tutorial);
  * if the request is a **pure prefill** the sparse kernel can serve
    (q_len == seq_len, seq_len % 128 == 0, hd == 128) -> call the wzc kernel;
  * otherwise (decode, chunked-prefill with context, ragged length) -> a small
    exact torch causal-GQA fallback for that request.

Block-level sparsity is a *prefill* optimization; decode is memory-bound and a
different kernel (see the wzc decode kernels), so routing decode to the fallback
here is intentional and correct.

Uses the production sparse kernel ``ops/_wzc_attn_sparse.py`` (dual
consumer-warpgroup ping-pong + parallel selection + WGMMA tensor-core scoring),
with fixed hyper-parameters tau=0.999 / local_window=2 / sink_blocks=1. Set
``WZC_SPARSE_STATS=1`` to log per-forward kernel-vs-fallback routing counters.
"""

import os
import sys

import torch

# --- locate the attention-test ops package that holds the wzc kernels ---
_ATTN_TEST_DIR = os.environ.get(
    "WZC_ATTENTION_TEST_DIR", "/dockerdata/wangzicheng/attention-test"
)
if _ATTN_TEST_DIR not in sys.path:
    sys.path.insert(0, _ATTN_TEST_DIR)

# Sparsity hyper-parameters. tau is read from WZC_SPARSE_TAU (default 0.999):
#   - tau=1.0  -> LOSSLESS: selects ALL causal segments, bit-identical to dense.
#                Use for HumanEval / correctness-sensitive runs (short prompts, so
#                the pad-to-128 pure-prefill path costs ~nothing extra).
#   - tau<1.0  -> lossy block-top-k sparsity; lower tau = more skipped KV segments
#                = faster long-context prefill, at some quality risk.
_TAU = float(os.environ.get("WZC_SPARSE_TAU", "0.999"))
_LOCAL = 2
_SINK = 1
_FORCE_DENSE = False  # debug toggle: True -> always torch fallback

_BLOCK = 128  # the wzc prefill kernel tiles KV in BLOCK_N=128 and needs S%128==0
_STATS = os.environ.get("WZC_SPARSE_STATS", "0") == "1"

# Routing counters (WZC_SPARSE_STATS=1 prints a line each forward). Lets us
# PROVE the sparse kernel actually fires vs silently falling back to torch.
_stat = {
    "forwards": 0,
    "kernel_reqs": 0, "kernel_tokens": 0,   # requests/tokens served by wzc kernel
    "fallback_reqs": 0, "fallback_tokens": 0,
    "max_kernel_seq": 0,
}


def _load_kernel():
    """Import the production wzc sparse-prefill kernel (dual-WG + WGMMA scoring)."""
    from ops import _wzc_attn_sparse as k
    return k


def _load_kernel_rect():
    """Import the RECTANGULAR-causal sparse-prefill kernel (chunked prefill:
    q_len < kv_len, context-offset causal). See ops/_wzc_attn_sparse_rect.py."""
    from ops import _wzc_attn_sparse_rect as k
    return k


def _load_paged_decoder():
    """Import the wzc paged decode kernel (FlashDecoding split-KV, page_size=128)."""
    from ops import _wzc_paged_attn_decode as d
    return d


_KERNEL = None
_KERNEL_RECT = None
_DECODER = None
_PAGE = 128          # the paged decode kernel's fixed page_size == BLOCK_N
_DECODE_GROUP = 8    # kernel requires q_heads == kv_heads * 8 (GQA group)
# Per-(device, num_pages, kv_heads) reusable kernel-layout KV pool + block_table,
# so the timed decode loop reallocates nothing.
_POOL_CACHE = {}


def _gather_kv(key_cache, value_cache, block_table, seq_idx, seq_len):
    """Gather one request's contiguous K/V from the paged cache.

    Returns K, V of shape (seq_len, num_kv_heads, head_size), stride-addressed
    exactly like the tutorial: position j -> block_table[seq_idx, j//bs], slot
    j%bs. Uses a vectorized gather (no python loop over positions).
    """
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    dev = key_cache.device

    pos = torch.arange(seq_len, device=dev)
    logical_block = pos // block_size
    slot = pos % block_size
    pb = block_table[seq_idx, logical_block]           # (seq_len,) physical block
    # key_cache[pb, :, slot, :] -> (seq_len, num_kv_heads, head_size)
    K = key_cache[pb][:, :, 0, :] if block_size == 1 else key_cache[pb, :, slot, :]
    V = value_cache[pb][:, :, 0, :] if block_size == 1 else value_cache[pb, :, slot, :]
    return K.contiguous(), V.contiguous()


def _decode_one(q1, K, V, scale):
    """Route one DECODE request (q_len==1) to the wzc paged decode kernel.

    q1: (num_heads, head_size) the single query token.  K/V: (seq_len, kv_heads,
    hd) gathered contiguous history. Repacks K/V into the kernel's paged pool
    layout (num_pages, kv_heads, page_size=128, hd) with a trivial block_table
    [0,1,2,...] and seq_len, then calls PagedKVDecoder.decode -> (num_heads, hd).

    Any seq_len works: the tail page is zero-padded; the kernel's last-page mask
    (S side) + the zeroed V pool (V side) make the partial page correct.
    """
    global _DECODER
    if _DECODER is None:
        _DECODER = _load_paged_decoder()
    import torch
    num_heads = q1.shape[0]
    seq_len_i, kv_heads, hd = K.shape
    n_pages = (seq_len_i + _PAGE - 1) // _PAGE
    dev = q1.device

    # Reusable kernel-layout pool. Tail page stays ZERO (required: masked P=0 * V
    # must be 0, not garbage/NaN). Grow the cached pool if a longer seq appears.
    pk = (dev.index, kv_heads)
    pool = _POOL_CACHE.get(pk)
    if pool is None or pool[0].shape[0] < n_pages:
        kc = torch.zeros(n_pages, kv_heads, _PAGE, hd, dtype=torch.bfloat16, device=dev)
        vc = torch.zeros_like(kc)
        bt = torch.arange(n_pages, dtype=torch.int32, device=dev).view(1, n_pages)
        sl = torch.zeros(1, dtype=torch.int32, device=dev)
        pool = (kc, vc, bt, sl)
        _POOL_CACHE[pk] = pool
    kc, vc, bt, sl = pool

    # Pad history to n_pages*128, then reshape (s_pad, HK, hd) ->
    # (n_pages, 128, HK, hd) -> (n_pages, HK, 128, hd) into the pool.
    s_pad = n_pages * _PAGE
    Kp = torch.zeros(s_pad, kv_heads, hd, dtype=torch.bfloat16, device=dev)
    Vp = torch.zeros_like(Kp)
    Kp[:seq_len_i] = K.to(torch.bfloat16)
    Vp[:seq_len_i] = V.to(torch.bfloat16)
    kc[:n_pages].copy_(Kp.view(n_pages, _PAGE, kv_heads, hd).permute(0, 2, 1, 3))
    vc[:n_pages].copy_(Vp.view(n_pages, _PAGE, kv_heads, hd).permute(0, 2, 1, 3))
    sl[0] = seq_len_i

    o = _DECODER.PagedKVDecoder.decode(
        q1.to(torch.bfloat16), kc, vc, bt, sl, 0, sm_scale=scale)
    return o.view(num_heads, hd)


def _decode_one_zerocopy(q1, key_cache, value_cache, block_table, req, seq_len, scale):
    """FAST decode path when the vLLM block_size already == the kernel page_size
    (128): the paged cache [num_blocks, kv_heads, 128, hd] IS the kernel's pool
    layout, so we pass it + the real block_table row through with NO gather/repack
    (zero-copy). Restores the kernel's native ~160us at 131072.

    q1: (num_heads, hd) single token. Returns (num_heads, hd).
    """
    global _DECODER
    if _DECODER is None:
        _DECODER = _load_paged_decoder()
    import torch
    num_heads = q1.shape[0]
    hd = q1.shape[1]
    # seq_len row for this request (int32, on device). Cache per (device,).
    pk = ("sl", q1.device.index)
    sl = _POOL_CACHE.get(pk)
    if sl is None:
        sl = torch.empty(1, dtype=torch.int32, device=q1.device)
        _POOL_CACHE[pk] = sl
    sl[0] = seq_len
    o = _DECODER.PagedKVDecoder.decode(
        q1.to(torch.bfloat16), key_cache, value_cache,
        block_table[req:req + 1], sl, 0, sm_scale=scale)
    return o.view(num_heads, hd)


def _torch_causal_gqa(q, k, v, scale):
    """Exact fallback for one request. q:(q_len,H,D) k/v:(seq_len,HK,D) fp.
    causal with context: query i (0-based within query) attends [0, context+i].

    Memory-safe: chunked over q-rows so we never materialize the full
    (q_len, group, seq_len) score tensor. At 100k chunked prefill a chunk can be
    q_len~2048 attending seq_len~95k; the naive einsum tried to alloc (2048,8,95k)
    fp32 ~= several GiB PER kv-head and OOM'd the engine (only ~512MiB free after
    weights+KV). Processing Q_CHUNK rows at a time caps peak at (Q_CHUNK,group,seq).
    """
    q_len, num_heads, hd = q.shape
    seq_len, num_kv_heads, _ = k.shape
    context = seq_len - q_len
    group = num_heads // num_kv_heads
    kf = k.float()
    vf = v.float()
    out = torch.empty((q_len, num_heads, hd), dtype=torch.float32, device=q.device)
    col = torch.arange(seq_len, device=q.device)[None, :]        # (1,seq_len)
    # Row-chunk size: keep peak scores tensor small regardless of q_len/seq_len.
    # Peak ~= Q_CHUNK*group*seq_len*4 bytes; at seq_len~95k, group=8 that is
    # 64*8*95k*4 ~= 195MB (fits the tight ~512MB-free budget at 100k). The engine
    # OOM'd at Q_CHUNK==q_len(2048) which needed several GiB per kv-head.
    Q_CHUNK = 64
    for r0 in range(0, q_len, Q_CHUNK):
        r1 = min(r0 + Q_CHUNK, q_len)
        qf = q[r0:r1].float() * scale                            # (rc,H,D)
        row = torch.arange(r0, r1, device=q.device)[:, None]     # (rc,1)
        mask = col > (context + row)                             # (rc,seq) True=disallow
        for kvh in range(num_kv_heads):
            qh = qf[:, kvh * group:(kvh + 1) * group, :]         # (rc,group,D)
            scores = torch.einsum("qgd,kd->qgk", qh, kf[:, kvh, :])  # (rc,group,seq)
            scores = scores.masked_fill(mask[:, None, :], float("-inf"))
            w = torch.softmax(scores, dim=-1)
            out[r0:r1, kvh * group:(kvh + 1) * group, :] = torch.einsum(
                "qgk,kd->qgd", w, vf[:, kvh, :])
    return out.to(q.dtype)


def paged_attention_wzc(
    query: torch.Tensor,        # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size]
    output: torch.Tensor,       # [num_tokens, num_heads, head_size]  (in-place)
    query_start_loc: torch.Tensor,  # [num_seqs + 1] int32
    seq_lens: torch.Tensor,         # [num_seqs] int32
    token_seq_idx: torch.Tensor,    # [num_tokens] int32 (unused; per-request loop)
    block_table: torch.Tensor,      # [num_seqs, max_num_blocks] int32
    scale: float,
) -> torch.Tensor:
    """Paged causal GQA attention; routes pure-prefill requests to the wzc
    block-level top-k sparse prefill kernel, everything else to torch."""
    global _KERNEL
    num_heads = query.shape[1]
    head_size = query.shape[2]
    qsl = query_start_loc.tolist()
    slens = seq_lens.tolist()
    num_seqs = len(slens)

    use_sparse = (not _FORCE_DENSE) and head_size == 128
    if use_sparse and _KERNEL is None:
        _KERNEL = _load_kernel()
    # Re-read tau per call so it can be changed at runtime (tests set
    # WZC_SPARSE_TAU per case; a serve run sets it once in the environment).
    tau = float(os.environ.get("WZC_SPARSE_TAU", _TAU))

    for req in range(num_seqs):
        q0, q1 = qsl[req], qsl[req + 1]
        q_len = q1 - q0
        if q_len == 0:
            continue
        seq_len = slens[req]
        q = query[q0:q1]                                    # (q_len,H,D)
        block_size = key_cache.shape[2]

        # DECODE (q_len==1, has context) -> wzc paged decode kernel. Needs GQA
        # group == 8 and head_size 128. The ZERO-COPY fast path requires the vLLM
        # paged cache to ALREADY be the kernel's pool layout: block_size==128 AND
        # the per-page K tile physically contiguous with head_dim leading. vLLM's
        # CUSTOM cache packs K|V in the last dim (…, block_size, 2*hs), so the
        # `key_cache` half-slice has a STRIDED slot dim (stride 2*hs, not hs) ->
        # the kernel's `_kv_to_cute` (mark_compact_shape_dynamic) rejects it
        # ("stride_order not consistent"). So only take zero-copy when key_cache is
        # actually contiguous in the kernel's layout; otherwise fall through to
        # _decode_one (gather+repack into a proper contiguous 128-page pool).
        decode_ok = (use_sparse and q_len == 1 and seq_len > 1
                     and num_heads == key_cache.shape[1] * _DECODE_GROUP)
        if decode_ok and block_size == _PAGE and key_cache.is_contiguous():
            output[q0:q1] = _decode_one_zerocopy(
                q[0], key_cache, value_cache, block_table, req, seq_len, scale
            ).unsqueeze(0).to(output.dtype)
            _stat["kernel_reqs"] += 1
            _stat["kernel_tokens"] += q_len
            if seq_len > _stat["max_kernel_seq"]:
                _stat["max_kernel_seq"] = seq_len
            continue

        # All other paths need the gathered contiguous history.
        K, V = _gather_kv(key_cache, value_cache, block_table, req, seq_len)

        is_pure_prefill = (q_len == seq_len)
        # The wzc kernel is square-causal and requires S % 128 == 0. Real prompt
        # lengths are almost never 128-aligned, so we PAD the sequence up to the
        # next 128 multiple with zeros at the END. This is correct for causal
        # attention: padded key rows sit at absolute positions >= seq_len, and a
        # real query row i (< seq_len) only attends keys <= i, so it never sees
        # the padding; the extra padded query rows compute discardable output
        # (we read back only the first q_len rows). => any pure-prefill request
        # can now use the kernel.
        kernel_ok = use_sparse and is_pure_prefill
        # DECODE with a non-128 vLLM block_size: gather+repack into the kernel's
        # 128-page pool (slower; the block_size==128 zero-copy path was already
        # handled above with a `continue`).
        decode_ok = (use_sparse and q_len == 1 and seq_len > 1
                     and num_heads == K.shape[1] * _DECODE_GROUP)

        # CHUNKED prefill (q_len>1, has context): q_len query tokens at absolute
        # positions [context, context+q_len), context = seq_len - q_len. This is
        # the bulk of long-context prefill under vLLM's default chunked scheduling
        # -> route to the RECTANGULAR-causal sparse kernel (not the slow torch
        # fallback). The kernel needs context % 128 == 0 (vLLM chunk boundaries
        # satisfy this when max_num_batched_tokens is a 128-multiple, e.g. 2048).
        context = seq_len - q_len
        chunk_ok = (use_sparse and q_len > 1 and context > 0
                    and context % _BLOCK == 0)

        if kernel_ok:
            s_pad = ((seq_len + _BLOCK - 1) // _BLOCK) * _BLOCK
            pad = s_pad - seq_len
            # (q_len,H,D)->(1,H,S,D); (seq_len,HK,D)->(1,HK,S,D), zero-padded.
            Qk = q.transpose(0, 1).unsqueeze(0).contiguous()
            Kk = K.transpose(0, 1).unsqueeze(0).contiguous()
            Vk = V.transpose(0, 1).unsqueeze(0).contiguous()
            if pad:
                Qk = torch.nn.functional.pad(Qk, (0, 0, 0, pad))
                Kk = torch.nn.functional.pad(Kk, (0, 0, 0, pad))
                Vk = torch.nn.functional.pad(Vk, (0, 0, 0, pad))
            Ok = _KERNEL.run(Qk, Kk, Vk, causal=True, sm_scale=scale,
                             tau=tau, local_window=_LOCAL, sink_blocks=_SINK)
            # (1,H,S,D) -> (q_len,H,D), dropping padded rows.
            output[q0:q1] = Ok[0, :, :q_len].transpose(0, 1).to(output.dtype)
            _stat["kernel_reqs"] += 1
            _stat["kernel_tokens"] += q_len
            if seq_len > _stat["max_kernel_seq"]:
                _stat["max_kernel_seq"] = seq_len
        elif chunk_ok:
            # Rectangular causal: pad q_len up to 128; kv_pad = context + q_pad
            # (>= real seq_len, and 128-aligned since context is). Zero-pad K/V at
            # the END: padded key rows sit at absolute positions >= seq_len, and a
            # real query row (< seq_len) only attends keys <= its abs pos, so it
            # never sees padding; padded query rows produce discardable output
            # (we read back only the first q_len rows).
            global _KERNEL_RECT
            if _KERNEL_RECT is None:
                _KERNEL_RECT = _load_kernel_rect()
            q_pad = ((q_len + _BLOCK - 1) // _BLOCK) * _BLOCK
            kv_pad = context + q_pad                       # 128-aligned
            padq = q_pad - q_len
            padkv = kv_pad - seq_len
            Qk = q.transpose(0, 1).unsqueeze(0).contiguous()       # (1,H,q_len,D)
            Kk = K.transpose(0, 1).unsqueeze(0).contiguous()       # (1,HK,seq,D)
            Vk = V.transpose(0, 1).unsqueeze(0).contiguous()
            if padq:
                Qk = torch.nn.functional.pad(Qk, (0, 0, 0, padq))
            if padkv:
                Kk = torch.nn.functional.pad(Kk, (0, 0, 0, padkv))
                Vk = torch.nn.functional.pad(Vk, (0, 0, 0, padkv))
            Ok = _KERNEL_RECT.run(Qk, Kk, Vk, causal=True, sm_scale=scale,
                                  tau=tau, local_window=_LOCAL, sink_blocks=_SINK)
            output[q0:q1] = Ok[0, :, :q_len].transpose(0, 1).to(output.dtype)
            _stat["kernel_reqs"] += 1
            _stat["kernel_tokens"] += q_len
            if seq_len > _stat["max_kernel_seq"]:
                _stat["max_kernel_seq"] = seq_len
        elif decode_ok:
            # (1,H,D) query token -> paged decode kernel over gathered history.
            output[q0:q1] = _decode_one(q[0], K, V, scale).unsqueeze(0).to(output.dtype)
            _stat["kernel_reqs"] += 1
            _stat["kernel_tokens"] += q_len
            if seq_len > _stat["max_kernel_seq"]:
                _stat["max_kernel_seq"] = seq_len
        else:
            output[q0:q1] = _torch_causal_gqa(q, K, V, scale)
            _stat["fallback_reqs"] += 1
            _stat["fallback_tokens"] += q_len

    if _STATS:
        _stat["forwards"] += 1
        print(f"[wzc-stats] fwd={_stat['forwards']} "
              f"kernel_reqs={_stat['kernel_reqs']} kernel_tok={_stat['kernel_tokens']} "
              f"fallback_reqs={_stat['fallback_reqs']} fallback_tok={_stat['fallback_tokens']} "
              f"max_kernel_seq={_stat['max_kernel_seq']} "
              f"(this_fwd: seqs={num_seqs} seq_lens={slens[:4]}{'...' if num_seqs>4 else ''})",
              flush=True)

    return output
