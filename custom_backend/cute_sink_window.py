"""
v11 实验：sink + 滑窗（StreamingLLM / A-shape）prefill，两趟 CuTe + LSE 合并。
每个 query attend： sink 前缀 [0, SINK) ∪ 局部窗口 [pos-W, pos]。
- Pass A：window_left=W 的局部窗口注意力（含对角），输出 O_a, LSE_a。
- Pass B：只在 sink 区 [0,SINK) 上的注意力（对 SINK 之后的 query 是纯 non-causal 全可见），
          输出 O_b, LSE_b。
- 合并：O = (O_a*exp(LSE_a) + O_b*exp(LSE_b)) / (exp(LSE_a)+exp(LSE_b))，用 LSE 稳定式。
注意：需 store_lse=True。sink 区与窗口区若重叠(query 在前 SINK+W 内)会重复计数 → 仅对
      pos >= SINK+W 的 query 用双趟；靠前 query 直接用 dense（本模块按 chunk 粒度近似处理）。

本模块目前作为离线验证/预研，未接入生产（生产 = v9 单趟 window）。
"""
import math
import sys
import torch

_CUTE = None


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


def _views_with_lse(q, k, v):
    """返回 q/k/v/o 的 (S,D,Hr,Hkv,B) view + 真实 LSE workspace(S,Hr,Hkv,B)。"""
    batch, q_heads, q_len, head_dim = q.shape
    kv_heads, kv_len = k.shape[1], k.shape[2]
    hr = q_heads // kv_heads
    q5 = q.reshape(batch, kv_heads, hr, q_len, head_dim).permute(3, 4, 2, 1, 0)
    k5 = k.permute(2, 3, 1, 0).unsqueeze(2)
    v5 = v.permute(2, 3, 1, 0).unsqueeze(2)
    out = torch.empty_like(q)
    o5 = out.reshape(batch, kv_heads, hr, q_len, head_dim).permute(3, 4, 2, 1, 0)
    lse = torch.empty(q_len, hr, kv_heads, batch, device=q.device, dtype=torch.float32)
    lse5 = lse.reshape(q_len, 1, hr, kv_heads, batch)
    return q5, k5, v5, o5, lse5, out, lse


def _run(op_cls, cutlass, cute, from_dlpack, fmha_utils, q, k, v, scale, mask_type, wl):
    q5, k5, v5, o5, lse5, out, lse = _views_with_lse(q, k, v)
    import cuda.bindings.driver as cuda
    D = q.shape[-1]
    op = op_cls(qk_acc_dtype=cutlass.Float32, pv_acc_dtype=cutlass.Float32,
                mma_tiler=(64, 128, D), is_persistent=False, mask_type=mask_type,
                use_lpt_scheduler=False, store_lse=True, unit_output_scale=True,
                q_stage=2, kv_stage=5, epi_stage=2)
    st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    sc = cutlass.Float32(scale); slg = cutlass.Float32(scale * 1.4426950408889634); so = cutlass.Float32(1.0)
    w = cutlass.Int32(wl) if wl is not None else None
    wr = cutlass.Int32(0)
    qc = from_dlpack(q5, assumed_align=16); kc = from_dlpack(k5, assumed_align=16)
    vc = from_dlpack(v5, assumed_align=16); oc = from_dlpack(o5, assumed_align=16)
    lc = from_dlpack(lse5, assumed_align=16)
    comp = cute.compile(op, qc, kc, vc, oc, lc, slg, sc, so, w, wr, st)
    comp(qc, kc, vc, oc, lc, slg, sc, so, w, wr, st)
    torch.cuda.synchronize()
    return out, lse  # out:(1,Hq,qlen,D)  lse:(qlen,Hr,Hkv,1)  (log-sum-exp, 自然对数)


def sink_window_prefill(q, k, v, scale, sink, window):
    """离线版：q(1,Hq,qlen,D) k/v(1,Hkv,kvlen,D)。返回 (1,Hq,qlen,D)。
    仅支持 qlen==kvlen 的整段 prefill 验证（ctx=0）。"""
    cuda, cutlass, cute, from_dlpack, KCls, fmha = _get_cute()
    # Pass A：滑窗（含对角 causal）
    oa, la = _run(KCls, cutlass, cute, from_dlpack, fmha, q, k, v, scale,
                  fmha.MaskEnum.WINDOW_MASK, window)
    # Pass B：只用 sink 段 K/V [0,sink)（对所有 query 全可见，non-causal）
    ks = k[:, :, :sink, :].contiguous()
    vs = v[:, :, :sink, :].contiguous()
    # 用一个大的 window_left 保证 sink 段内全可见但不含对角以外——这里 sink 段 kvlen=sink，
    # 对 query pos>=sink 全部可见；用 WINDOW_MASK + 大窗口 == 全可见（causal 在 sink 段末端）。
    # 简化：直接 no-mask（RESIDUAL 处理边界）
    ob, lb = _run(KCls, cutlass, cute, from_dlpack, fmha, q, ks, vs, scale,
                  fmha.MaskEnum.RESIDUAL_MASK, None)
    # LSE 合并（自然对数域）
    la_ = la.permute(0, 1, 2, 3).reshape(q.shape[2], -1)  # (qlen, Hq)
    # 对齐形状到 (1,Hq,qlen,1)
    def lse_to_bcast(lse):
        # lse:(qlen,Hr,Hkv,1) -> (1,Hq,qlen,1)
        qlen, hr, hkv, _ = lse.shape
        return lse.permute(3, 1, 2, 0).reshape(1, hr * hkv, qlen, 1).float()
    LA = lse_to_bcast(la); LB = lse_to_bcast(lb)
    m = torch.maximum(LA, LB)
    wa = torch.exp(LA - m); wb = torch.exp(LB - m)
    out = (oa.float() * wa + ob.float() * wb) / (wa + wb)
    return out.to(q.dtype)
