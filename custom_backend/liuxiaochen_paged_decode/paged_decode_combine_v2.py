#!/usr/bin/env python3
"""V2 combine: LSE-weighted split reduction for paged decode (pure GPU).

Independent implementation by Liu Xiaochen. One CTA (128 threads) per
(num_heads, num_seqs). Reads FP32 partial_o + FP32 LSE per split; writes the
final BF16 output to the packed query token of that sequence.

Decode-only: sequence seq_id's single query token index = query_start_loc[seq_id]
(read on GPU; no assumption that output[seq_id] is the token). Writes
output[q_token, head, d].
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

HEAD_DIM = 128


class LiuXiaochenPagedDecodeCombineV2:
    def __init__(self, num_seqs, num_heads, head_dim, num_splits_max):
        assert head_dim == HEAD_DIM
        self.num_seqs = num_seqs
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_splits_max = num_splits_max
        self.num_threads = 128

    @cute.jit
    def __call__(
        self,
        mPartial: cute.Tensor,  # [num_seqs, num_heads, num_splits_max, D] FP32
        mLSE: cute.Tensor,      # [num_seqs, num_heads, num_splits_max]     FP32
        mOut: cute.Tensor,      # [num_tokens, num_heads, D] BF16 (in-place)
        mQStart: cute.Tensor,   # [num_seqs+1] int32
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Smem:
            weights: cute.struct.MemRange[cutlass.Float32, self.num_splits_max]
            reduction: cute.struct.MemRange[cutlass.Float32, self.num_threads]

        self.kernel(mPartial, mLSE, mOut, mQStart, Smem).launch(
            grid=[self.num_heads, self.num_seqs, 1],
            block=[self.num_threads, 1, 1], stream=stream,
        )

    @cute.kernel
    def kernel(self, mPartial, mLSE, mOut, mQStart, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        head, seq_id, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Smem)
        weights = storage.weights.get_tensor(cute.make_layout(self.num_splits_max))
        reduction = storage.reduction.get_tensor(cute.make_layout(self.num_threads))

        q_token = mQStart[seq_id]

        # 1) global max of LSE over splits (skip -inf / empty).
        local_max = -cutlass.Float32.inf
        s = tidx
        while s < self.num_splits_max:
            local_max = cute.arch.fmax(local_max, mLSE[seq_id, head, s])
            s += self.num_threads
        reduction[tidx] = local_max
        cute.arch.sync_threads()
        gmax = -cutlass.Float32.inf
        for i in cutlass.range_constexpr(self.num_threads):
            gmax = cute.arch.fmax(gmax, reduction[i])
        cute.arch.sync_threads()

        # 2) weights = exp(lse - gmax); denom.
        local_sum = cutlass.Float32(0.0)
        s = tidx
        while s < self.num_splits_max:
            lse = mLSE[seq_id, head, s]
            w = cute.math.exp(lse - gmax, fastmath=True)
            # empty split -> lse=-inf -> w=0
            weights[s] = w
            local_sum += w
            s += self.num_threads
        reduction[tidx] = local_sum
        cute.arch.sync_threads()
        denom = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(self.num_threads):
            denom += reduction[i]
        inv = cute.arch.rcp_approx(denom) if denom > 0.0 else cutlass.Float32(0.0)
        s = tidx
        while s < self.num_splits_max:
            weights[s] = weights[s] * inv
            s += self.num_threads
        cute.arch.sync_threads()

        # 3) weighted sum over splits per dim -> output token.
        dim = tidx
        while dim < self.head_dim:
            acc = cutlass.Float32(0.0)
            for si in cutlass.range_constexpr(self.num_splits_max):
                acc += mPartial[seq_id, head, si, dim] * weights[si]
            mOut[q_token, head, dim] = acc.to(mOut.element_type)
            dim += self.num_threads
