"""
CuTe WGMMA prefill 接入 vLLM（支持 chunked prefill）。

分页 KV -> gather 成连续 [kvlen, nkv, D] -> BHSD -> CuTe。
- q_len < kv_len（chunk 带 context 前缀）用 WINDOW_MASK_INFERENCE（offset=kv-q）。
- q_len == kv_len（首个 chunk / 无 context）用 WINDOW_MASK + window_right=0。
- 编译按 (q_len, kv_len) 形状缓存复用。
"""
import math
import sys

import torch
import triton
import triton.language as tl

_CUTE = None
_COMPILE_CACHE = {}


def _get_cute():
    global _CUTE
    if _CUTE is None:
        # prefill_kernel.py / fmha_helpers.py 在本文件同目录，加入 sys.path 后直接 import。
        import os as _os
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from prefill_kernel import FlashAttention5GQACuteDSL
        import fmha_helpers as fmha_utils
        _CUTE = (cuda, cutlass, cute, from_dlpack, FlashAttention5GQACuteDSL, fmha_utils)
    return _CUTE


# ---- gather 分页 KV -> 连续 [kvlen, nkv, D] ----
@triton.jit
def _gather_kv_kernel(
    kc_ptr, vc_ptr, block_table_ptr,
    k_out_ptr, v_out_ptr,
    seq_len, KV_BLOCK: tl.constexpr, nkv: tl.constexpr, D: tl.constexpr,
    kc_sb, kc_sh, kc_ss, vc_sb, vc_sh, vc_ss,
    ko_st, ko_sh, vo_st, vo_sh, bt_stride,
    BLOCK_T: tl.constexpr,
):
    # grid=(cdiv(seq_len,BLOCK_T), nkv)
    t0 = tl.program_id(0) * BLOCK_T
    h = tl.program_id(1)
    offs_t = t0 + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    valid = offs_t < seq_len
    logical = offs_t // KV_BLOCK
    slot = offs_t % KV_BLOCK
    pb = tl.load(block_table_ptr + logical, mask=valid, other=0)
    k_src = kc_ptr + pb[:, None] * kc_sb + h * kc_sh + slot[:, None] * kc_ss + offs_d[None, :]
    v_src = vc_ptr + pb[:, None] * vc_sb + h * vc_sh + slot[:, None] * vc_ss + offs_d[None, :]
    k = tl.load(k_src, mask=valid[:, None], other=0.0)
    v = tl.load(v_src, mask=valid[:, None], other=0.0)
    k_dst = k_out_ptr + offs_t[:, None] * ko_st + h * ko_sh + offs_d[None, :]
    v_dst = v_out_ptr + offs_t[:, None] * vo_st + h * vo_sh + offs_d[None, :]
    tl.store(k_dst, k, mask=valid[:, None])
    tl.store(v_dst, v, mask=valid[:, None])


def _gather_kv(key_cache, value_cache, block_table_row, seq_len):
    """分页 -> 连续 [seq_len, nkv, D]。block_table_row: [max_blocks] int32（单请求）。"""
    num_blocks, nkv, kv_block, D = key_cache.shape
    k_out = torch.empty(seq_len, nkv, D, device=key_cache.device, dtype=key_cache.dtype)
    v_out = torch.empty(seq_len, nkv, D, device=key_cache.device, dtype=key_cache.dtype)
    BLOCK_T = 64
    grid = ((seq_len + BLOCK_T - 1) // BLOCK_T, nkv)
    _gather_kv_kernel[grid](
        key_cache, value_cache, block_table_row, k_out, v_out,
        seq_len, KV_BLOCK=kv_block, nkv=nkv, D=D,
        kc_sb=key_cache.stride(0), kc_sh=key_cache.stride(1), kc_ss=key_cache.stride(2),
        vc_sb=value_cache.stride(0), vc_sh=value_cache.stride(1), vc_ss=value_cache.stride(2),
        ko_st=k_out.stride(0), ko_sh=k_out.stride(1),
        vo_st=v_out.stride(0), vo_sh=v_out.stride(1),
        bt_stride=block_table_row.stride(0), BLOCK_T=BLOCK_T,
    )
    return k_out, v_out


def _make_views(q, k, v):
    """q:(1,Hq,qlen,D) k/v:(1,Hkv,kvlen,D) -> (S,D,Hr,Hkv,B) 逻辑布局。"""
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads, kv_len = k.shape[1], k.shape[2]
    heads_per_kv = q_heads // kv_heads
    q_5d = q.reshape(batch, kv_heads, heads_per_kv, q_len, head_dim)
    q_sdrrb = q_5d.permute(3, 4, 2, 1, 0)
    k_sdrrb = k.permute(2, 3, 1, 0).unsqueeze(2)
    v_sdrrb = v.permute(2, 3, 1, 0).unsqueeze(2)
    out_bhsd = torch.empty_like(q)
    out_5d = out_bhsd.reshape(batch, kv_heads, heads_per_kv, q_len, head_dim)
    out_sdrrb = out_5d.permute(3, 4, 2, 1, 0)
    lse_scalar = torch.empty(1, device=q.device, dtype=torch.float32)
    lse_sdrrb = lse_scalar.as_strided(
        (q_len, 1, heads_per_kv, kv_heads, batch), (0, 0, 0, 0, 0))
    return q_sdrrb, k_sdrrb, v_sdrrb, out_sdrrb, lse_sdrrb, out_bhsd


def cute_prefill_paged(query, key_cache, value_cache, output,
                       query_start_loc, seq_lens, block_table, scale):
    """vLLM CUSTOM 后端 prefill 路径（单请求 batch=1）。
    query: [num_tokens, Hq, D]; output: [num_tokens, Hq, D]（原地写）。
    KV cache 已含本 chunk 的新 KV（forward 里先 reshape_and_cache 写过）。
    """
    cuda, cutlass, cute, from_dlpack, KernelClass, fmha_utils = _get_cute()
    num_tokens, hq, D = query.shape
    seq_len = int(seq_lens[0].item())          # 该请求当前总 KV 长度（含本 chunk）
    q_len = num_tokens
    nkv = key_cache.shape[1]

    # gather 连续 KV [seq_len, nkv, D]
    bt_row = block_table[0]
    k_cont, v_cont = _gather_kv(key_cache, value_cache, bt_row, seq_len)

    # ---- window + block-sparse(sink) 融合分支（PREFILL_SINK>0 启用）----
    # 仅对整段 prefill 且 seq_len > window+sink 时走两趟 CuTe+LSE。
    import os as _os
    _win = int(_os.environ.get("PREFILL_WINDOW", "4096"))
    _sink = int(_os.environ.get("PREFILL_SINK", "0"))
    if _sink > 0 and q_len == seq_len and seq_len > _win + _sink:
        try:
            from .cute_window_sink import window_sink_attn
            q_bhsd = query.permute(1, 0, 2).unsqueeze(0).contiguous()
            k_bhsd = k_cont.permute(1, 0, 2).unsqueeze(0).contiguous()
            v_bhsd = v_cont.permute(1, 0, 2).unsqueeze(0).contiguous()
            o = window_sink_attn(q_bhsd, k_bhsd, v_bhsd, scale, _win, _sink)  # (1,Hq,qlen,D)
            output.copy_(o[0].permute(1, 0, 2))
            return output
        except Exception as e:
            import sys as _sys
            print(f"[CUSTOM] window_sink fallback to single-window: {type(e).__name__}: {e}",
                  file=_sys.stderr)

    # 2) BHSD
    q_bhsd = query.permute(1, 0, 2).unsqueeze(0).contiguous()      # (1,Hq,qlen,D)
    k_bhsd = k_cont.permute(1, 0, 2).unsqueeze(0).contiguous()     # (1,Hkv,kvlen,D)
    v_bhsd = v_cont.permute(1, 0, 2).unsqueeze(0).contiguous()
    q_view, k_view, v_view, o_view, lse_view, out_bhsd = _make_views(q_bhsd, k_bhsd, v_bhsd)

    q_c = from_dlpack(q_view, assumed_align=16)
    k_c = from_dlpack(k_view, assumed_align=16)
    v_c = from_dlpack(v_view, assumed_align=16)
    o_c = from_dlpack(o_view, assumed_align=16)
    lse_c = from_dlpack(lse_view, assumed_align=16)

    # 3) mask 选择：q_len==kv_len -> 普通 causal；否则带 context 的 INFERENCE
    if q_len == seq_len:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK
    else:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE

    # 滑窗稀疏：只 attend 最近 win 个 key，仅当 seq_len 超过窗口时启用；win=0 关闭（dense）。
    import os as _os
    win = int(_os.environ.get("PREFILL_WINDOW", "4096"))
    use_window = (win > 0 and seq_len > win)
    wl = cutlass.Int32(win) if use_window else None
    wr = cutlass.Int32(0)

    key = (q_len, seq_len, hq, nkv, D, mask_type, win if use_window else 0)
    stream = cuda.CUstream(torch.cuda.current_stream(query.device).cuda_stream)
    sc = cutlass.Float32(float(scale))
    sl = cutlass.Float32(float(scale) * 1.4426950408889634)
    so = cutlass.Float32(1.0)

    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        op = KernelClass(
            qk_acc_dtype=cutlass.Float32, pv_acc_dtype=cutlass.Float32,
            mma_tiler=(64, 128, D), is_persistent=False, mask_type=mask_type,
            use_lpt_scheduler=False, store_lse=False, unit_output_scale=True,
            q_stage=2, kv_stage=5, epi_stage=2,
        )
        compiled = cute.compile(op, q_c, k_c, v_c, o_c, lse_c, sl, sc, so, wl, wr, stream)
        _COMPILE_CACHE[key] = compiled

    compiled(q_c, k_c, v_c, o_c, lse_c, sl, sc, so, wl, wr, stream)

    # 4) 写回 output [num_tokens, Hq, D]
    output.copy_(out_bhsd[0].permute(1, 0, 2))
    return output
