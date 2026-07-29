#!/usr/bin/env python3
"""V3 runner: direct paged-KV warp-MMA decode with cp.async 2-stage double buffer.

Public entry `paged_decode_v3(...)` mirrors paged_decode_v2 exactly (same input
support domain, scale/split semantics, workspace shape, packed output mapping) and
reuses the V2 combine kernel unchanged. The ONLY difference from V2 is the Stage-1
K/V load path (cp.async 2-stage prefetch instead of synchronous per-element copy).

Extra validation (Section 六): cp.async needs each token-row's 128-elem span to be
16-byte aligned and last-dim contiguous. We verify data_ptr alignment, stride(-1)==1,
and that block/head/slot strides keep every row start 16-byte aligned. On violation
we raise a clear ValueError (NO silent clone/contiguous/sync-fallback/triton-fallback).

Lazy compile + cached workspace; launched with the current tensors each call (no
tensor-bound closure). identity: "liuxiaochen-paged-decode-v3-mma-cpasync-2stage".
"""

import math

import torch

if not hasattr(torch, "float4_e2m1fn_x2"):
    torch.float4_e2m1fn_x2 = torch.uint8

_IDENTITY = "liuxiaochen-paged-decode-v3-mma-cpasync-2stage"
_CPASYNC_STAGES = 2
_SMEM_LAYOUT_ID = "swizzle(3,3,3)-2K2V-blkcopy(4,8)x(1,8)"
_compile_cache = {}
_workspace_cache = {}

_Q_HEADS = 64
_KV_HEADS = 8
_HEAD_DIM = 128
_BLOCK_SIZE = 16
_N_BLOCK = 64
_VEC_ELEMS = 8  # 16-byte / 2-byte bf16


def _ceil_div(a, b):
    return (a + b - 1) // b


def _workspace(num_seqs, num_heads, num_splits_max, head_dim, device):
    key = (device.index, num_seqs, num_heads, num_splits_max, head_dim)
    buf = _workspace_cache.get(key)
    if buf is None:
        partial = torch.empty(num_seqs, num_heads, num_splits_max, head_dim,
                              device=device, dtype=torch.float32)
        lse = torch.full((num_seqs, num_heads, num_splits_max), float("-inf"),
                         device=device, dtype=torch.float32)
        buf = (partial, lse)
        _workspace_cache[key] = buf
    return buf


def workspace_bytes(num_seqs, num_heads, num_splits_max, head_dim):
    p = num_seqs * num_heads * num_splits_max * head_dim * 4
    l = num_seqs * num_heads * num_splits_max * 4
    return p + l


def _check_cpasync_alignment(name, cache):
    """cp.async requires 16-byte aligned, contiguous 128-elem token rows."""
    if cache.stride(-1) != 1:
        raise ValueError(f"V3 cp.async requires {name}.stride(-1)==1 (contiguous D); "
                         f"got stride={cache.stride()}")
    if cache.data_ptr() % 16 != 0:
        raise ValueError(f"V3 cp.async requires {name}.data_ptr() 16-byte aligned; "
                         f"got {cache.data_ptr()} (mod16={cache.data_ptr()%16})")
    # Every token-row start must be 16-byte aligned => each stride that indexes
    # rows (block, head, slot) must be a multiple of 8 bf16 elements (16 bytes).
    sblock, shead, sslot, _ = cache.stride()
    for nm, s in (("block", sblock), ("head", shead), ("slot", sslot)):
        if s % _VEC_ELEMS != 0:
            raise ValueError(
                f"V3 cp.async requires {name} {nm}-stride ({s}) to be a multiple of "
                f"{_VEC_ELEMS} bf16 (16B) so every token row start is 16-byte aligned")


def paged_decode_v3(
    query, key_cache, value_cache, output,
    query_start_loc, seq_lens, token_seq_idx, block_table,
    scale, split_size_tokens, max_seq_len,
):
    # ---- validation (decode-only fixed target; mirrors V2) ----
    for nm, t in (("query", query), ("key_cache", key_cache), ("value_cache", value_cache),
                  ("output", output), ("query_start_loc", query_start_loc),
                  ("seq_lens", seq_lens), ("block_table", block_table)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{nm} must be a torch.Tensor")
        if not t.is_cuda:
            raise ValueError(f"{nm} must be on CUDA")
    major, minor = torch.cuda.get_device_capability(query.device)
    if (major, minor) != (9, 0):
        raise ValueError(f"V3 requires SM90, got SM{major}{minor}")
    if query.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        raise TypeError("V3 requires bf16 query/key_cache/value_cache")
    if output.dtype != torch.bfloat16:
        raise TypeError("V3 requires bf16 output")

    num_tokens, num_heads, head_dim = query.shape
    num_blocks, num_kv_heads, block_size, head_dim_c = key_cache.shape
    num_seqs = seq_lens.shape[0]
    if num_heads != _Q_HEADS or num_kv_heads != _KV_HEADS:
        raise ValueError(f"V3 fixed target Hq={_Q_HEADS}, Hkv={_KV_HEADS}; got {num_heads}/{num_kv_heads}")
    if head_dim != _HEAD_DIM or head_dim_c != _HEAD_DIM:
        raise ValueError(f"V3 requires head_dim={_HEAD_DIM}")
    if block_size != _BLOCK_SIZE:
        raise ValueError(f"V3 requires block_size={_BLOCK_SIZE}")
    if tuple(value_cache.shape) != tuple(key_cache.shape):
        raise ValueError("key_cache and value_cache must share shape")
    if num_tokens != num_seqs:
        raise ValueError(f"V3 decode-only: num_tokens({num_tokens}) must == num_seqs({num_seqs})")
    qsl = query_start_loc
    if qsl.shape[0] != num_seqs + 1:
        raise ValueError("query_start_loc must have num_seqs+1 entries")

    # cp.async alignment domain (Section 六): raise, never silently fix.
    _check_cpasync_alignment("key_cache", key_cache)
    _check_cpasync_alignment("value_cache", value_cache)

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a finite positive number")
    if split_size_tokens % _N_BLOCK != 0:
        raise ValueError(f"split_size_tokens must be a multiple of {_N_BLOCK}")

    num_splits_max = _ceil_div(int(max_seq_len), int(split_size_tokens))
    if num_splits_max < 1:
        num_splits_max = 1

    device = query.device
    partial, lse = _workspace(num_seqs, num_heads, num_splits_max, head_dim, device)
    lse.fill_(float("-inf"))  # reset neutral each call

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from .paged_decode_stage1_v3 import LiuXiaochenPagedDecodeStage1V3
    from .paged_decode_combine_v2 import LiuXiaochenPagedDecodeCombineV2

    q_c = from_dlpack(query, assumed_align=16)
    kc = from_dlpack(key_cache, assumed_align=16)
    vc = from_dlpack(value_cache, assumed_align=16)
    o_c = from_dlpack(output, assumed_align=16)
    p_c = from_dlpack(partial, assumed_align=16)
    l_c = from_dlpack(lse, assumed_align=16)
    qsl_c = from_dlpack(query_start_loc.to(torch.int32), assumed_align=4)
    sl_c = from_dlpack(seq_lens.to(torch.int32), assumed_align=4)
    bt_c = from_dlpack(block_table.to(torch.int32), assumed_align=4)

    torch_stream = torch.cuda.current_stream(device)
    stream = cuda.CUstream(int(torch_stream.cuda_stream))
    scale_log2 = cutlass.Float32(scale * 1.4426950408889634)

    key = (_IDENTITY, device.index, (major, minor), str(query.dtype),
           num_tokens, num_seqs, num_heads, num_kv_heads, head_dim, block_size,
           num_blocks, block_table.shape[1],
           query.stride(), key_cache.stride(), value_cache.stride(), output.stride(),
           block_table.stride(), int(split_size_tokens), num_splits_max,
           float(scale), _CPASYNC_STAGES, _SMEM_LAYOUT_ID)
    compiled = _compile_cache.get(key)
    if compiled is None:
        s1 = LiuXiaochenPagedDecodeStage1V3(
            num_seqs=num_seqs, num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, block_size=block_size, num_splits_max=num_splits_max,
            split_size_tokens=int(split_size_tokens), n_block=_N_BLOCK,
        )
        cmb = LiuXiaochenPagedDecodeCombineV2(
            num_seqs=num_seqs, num_heads=num_heads, head_dim=head_dim,
            num_splits_max=num_splits_max,
        )
        stage1 = cute.compile(s1, q_c, kc, vc, p_c, l_c, qsl_c, sl_c, bt_c, scale_log2, stream)
        combine = cute.compile(cmb, p_c, l_c, o_c, qsl_c, stream)
        compiled = (stage1, combine)
        _compile_cache[key] = compiled

    stage1, combine = compiled
    stage1(q_c, kc, vc, p_c, l_c, qsl_c, sl_c, bt_c, scale_log2, stream)
    combine(p_c, l_c, o_c, qsl_c, stream)
    return output


def build_v3_runners(query, key_cache, value_cache, output,
                     query_start_loc, seq_lens, token_seq_idx, block_table,
                     scale, split_size_tokens, max_seq_len):
    """Benchmark accessor (NOT a kernel change): compile V3 Stage-1 + V2 combine
    and return separate launch closures + workspace-reset closure, so a
    microbenchmark can time Stage-1-only / combine-only separately using
    pre-allocated, pre-reset workspace. Kernel math identical to paged_decode_v3().
    Returns (run_stage1, run_combine, reset_lse, num_splits_max, workspace_bytes_val).
    """
    import math as _math
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from .paged_decode_stage1_v3 import LiuXiaochenPagedDecodeStage1V3
    from .paged_decode_combine_v2 import LiuXiaochenPagedDecodeCombineV2

    _check_cpasync_alignment("key_cache", key_cache)
    _check_cpasync_alignment("value_cache", value_cache)

    num_tokens, num_heads, head_dim = query.shape
    num_blocks, num_kv_heads, block_size, _ = key_cache.shape
    num_seqs = seq_lens.shape[0]
    if scale is None:
        scale = 1.0 / _math.sqrt(head_dim)
    scale = float(scale)
    num_splits_max = max(1, _ceil_div(int(max_seq_len), int(split_size_tokens)))
    device = query.device
    partial, lse = _workspace(num_seqs, num_heads, num_splits_max, head_dim, device)

    q_c = from_dlpack(query, assumed_align=16)
    kc = from_dlpack(key_cache, assumed_align=16)
    vc = from_dlpack(value_cache, assumed_align=16)
    o_c = from_dlpack(output, assumed_align=16)
    p_c = from_dlpack(partial, assumed_align=16)
    l_c = from_dlpack(lse, assumed_align=16)
    qsl_c = from_dlpack(query_start_loc.to(torch.int32), assumed_align=4)
    sl_c = from_dlpack(seq_lens.to(torch.int32), assumed_align=4)
    bt_c = from_dlpack(block_table.to(torch.int32), assumed_align=4)

    torch_stream = torch.cuda.current_stream(device)
    stream = cuda.CUstream(int(torch_stream.cuda_stream))
    scale_log2 = cutlass.Float32(scale * 1.4426950408889634)

    s1 = LiuXiaochenPagedDecodeStage1V3(
        num_seqs=num_seqs, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, block_size=block_size, num_splits_max=num_splits_max,
        split_size_tokens=int(split_size_tokens), n_block=_N_BLOCK,
    )
    cmb = LiuXiaochenPagedDecodeCombineV2(
        num_seqs=num_seqs, num_heads=num_heads, head_dim=head_dim,
        num_splits_max=num_splits_max,
    )
    stage1 = cute.compile(s1, q_c, kc, vc, p_c, l_c, qsl_c, sl_c, bt_c, scale_log2, stream)
    combine = cute.compile(cmb, p_c, l_c, o_c, qsl_c, stream)

    def reset_lse():
        lse.fill_(float("-inf"))

    def run_stage1():
        stage1(q_c, kc, vc, p_c, l_c, qsl_c, sl_c, bt_c, scale_log2, stream)

    def run_combine():
        combine(p_c, l_c, o_c, qsl_c, stream)

    ws = workspace_bytes(num_seqs, num_heads, num_splits_max, head_dim)
    return run_stage1, run_combine, reset_lse, num_splits_max, ws
