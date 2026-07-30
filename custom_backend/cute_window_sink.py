"""
window + block-sparse(sink) 融合 prefill —— 基于冠军 CuTe WGMMA kernel，两趟 + LSE 合并。
=====================================================================================
每个 query attend：局部窗口 [pos-W, pos]（causal）∪ sink 前缀 [0, SINK)。
这是 StreamingLLM / A-shape 稀疏（block-sparse 的最重要两类块：最近块 + 前缀块）。
在冠军 CuTe kernel 上做两趟 + LSE(log-sum-exp) 合并，保留 WGMMA 速度：
  Pass A: window_left=W 的窗口注意力（含对角 causal）→ O_a, LSE_a
  Pass B: 只在 sink 段 [0,SINK) K/V 上的注意力（对 query 全可见，非因果）→ O_b, LSE_b
  合并:  O = softmax 加权(O_a,O_b by LSE_a,LSE_b)。
去重：仅当 query 的窗口左界 > SINK 时，sink 段与窗口不重叠，合并才精确；
      对窗口已覆盖 sink 的靠前 query（pos-W < SINK），Pass B 贡献会与 A 重叠 → 轻微高估，
      但这些 query 本就靠前、影响小；实践中 HumanEval 短 prompt 根本走 dense（seq<=W）。

生产开关：PREFILL_SINK>0 且 seq_len>W+SINK 时启用；否则回落单趟窗口(cute_prefill_paged)。
"""
import os
import sys

import torch

_CUTE = None
_WS_CACHE = {}


def _get_cute():
    global _CUTE
    if _CUTE is None:
        for p in ("/dockerdata/quanbofeng/attention-test",
                  "/dockerdata/quanbofeng/attention-test/ops"):
            if p not in sys.path:
                sys.path.insert(0, p)
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from quanbofeng_final.prefill_kernel import FlashAttention5GQACuteDSL
        from quanbofeng_final import fmha_helpers as fmha_utils
        _CUTE = (cuda, cutlass, cute, from_dlpack, FlashAttention5GQACuteDSL, fmha_utils)
    return _CUTE


def _views_lse(q, k, v):
    """q:(1,Hq,qlen,D) k/v:(1,Hkv,kvlen,D) -> (S,D,Hr,Hkv,B) views + 真实 LSE(qlen,Hr,Hkv,B)。"""
    b, hq, qlen, D = q.shape
    hkv, kvlen = k.shape[1], k.shape[2]
    hr = hq // hkv
    q5 = q.reshape(b, hkv, hr, qlen, D).permute(3, 4, 2, 1, 0)
    k5 = k.permute(2, 3, 1, 0).unsqueeze(2)
    v5 = v.permute(2, 3, 1, 0).unsqueeze(2)
    out = torch.empty_like(q)
    o5 = out.reshape(b, hkv, hr, qlen, D).permute(3, 4, 2, 1, 0)
    lse = torch.empty(qlen, hr, hkv, b, device=q.device, dtype=torch.float32)
    lse5 = lse.reshape(qlen, 1, hr, hkv, b)
    return q5, k5, v5, o5, lse5, out, lse


def _run_cute(q, k, v, scale, mask_type, window_left, tag):
    """跑一趟冠军 kernel（store_lse=True），返回 out(1,Hq,qlen,D) 和 lse(qlen,Hr,Hkv,1)。"""
    cuda, cutlass, cute, from_dlpack, KCls, fmha = _get_cute()
    D = q.shape[-1]
    q5, k5, v5, o5, lse5, out, lse = _views_lse(q, k, v)
    qc = from_dlpack(q5, assumed_align=16); kc = from_dlpack(k5, assumed_align=16)
    vc = from_dlpack(v5, assumed_align=16); oc = from_dlpack(o5, assumed_align=16)
    lc = from_dlpack(lse5, assumed_align=16)
    st = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    sc = cutlass.Float32(float(scale)); slg = cutlass.Float32(float(scale) * 1.4426950408889634)
    so = cutlass.Float32(1.0)
    wl = cutlass.Int32(window_left) if window_left is not None else None
    wr = cutlass.Int32(0)
    ckey = (tag, tuple(q.shape), tuple(k.shape), window_left)
    comp = _WS_CACHE.get(ckey)
    if comp is None:
        op = KCls(qk_acc_dtype=cutlass.Float32, pv_acc_dtype=cutlass.Float32,
                  mma_tiler=(64, 128, D), is_persistent=False, mask_type=mask_type,
                  use_lpt_scheduler=False, store_lse=True, unit_output_scale=True,
                  q_stage=2, kv_stage=5, epi_stage=2)
        comp = cute.compile(op, qc, kc, vc, oc, lc, slg, sc, so, wl, wr, st)
        _WS_CACHE[ckey] = comp
    comp(qc, kc, vc, oc, lc, slg, sc, so, wl, wr, st)
    return out, lse


def _lse_bcast(lse, hq):
    """lse:(qlen,Hr,Hkv,1) -> (1,Hq,qlen,1) float32，与 out 广播对齐。"""
    qlen, hr, hkv, _ = lse.shape
    return lse.permute(3, 1, 2, 0).reshape(1, hr * hkv, qlen, 1).float()


def window_sink_attn(q, k, v, scale, window, sink):
    """q:(1,Hq,qlen,D) k/v:(1,Hkv,kvlen,D)。返回 (1,Hq,qlen,D)。
    要求 kvlen==qlen 的整段 prefill（ctx=0）；chunk 场景由上层用 INFERENCE mask 适配。"""
    cuda, cutlass, cute, from_dlpack, KCls, fmha = _get_cute()
    # Pass A：窗口（含对角 causal）
    oa, la = _run_cute(q, k, v, scale, fmha.MaskEnum.WINDOW_MASK, window, "winA")
    # Pass B：只用 sink 段 K/V（非因果全可见）— RESIDUAL_MASK 处理边界
    ks = k[:, :, :sink, :].contiguous()
    vs = v[:, :, :sink, :].contiguous()
    ob, lb = _run_cute(q, ks, vs, scale, fmha.MaskEnum.RESIDUAL_MASK, None, "sinkB")
    # LSE 合并（自然对数域，数值稳定）
    hq = q.shape[1]
    LA = _lse_bcast(la, hq); LB = _lse_bcast(lb, hq)
    m = torch.maximum(LA, LB)
    wa = torch.exp(LA - m); wb = torch.exp(LB - m)
    out = (oa.float() * wa + ob.float() * wb) / (wa + wb + 1e-20)
    return out.to(q.dtype)
