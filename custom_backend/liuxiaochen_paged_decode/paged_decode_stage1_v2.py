#!/usr/bin/env python3
"""V2 Stage-1: direct paged-KV warp-MMA decode (synchronous load, correctness-first).

Independent implementation by Liu Xiaochen. Reuses the numerical core of the
author's old continuous-KV B5/B7 (warp-MMA m16n8k16 QK/PV + online softmax + LSE)
but changes the ONE key variable: K/V are read directly from a paged KV-cache
block pool via a block_table, NOT from a contiguous [B,Hkv,KV,D] tensor. No gather,
no index_select, no repeat, no contiguous workspace — the kernel resolves
logical_pos -> logical_block -> block_table -> physical_block -> slot and reads
key_cache/value_cache by their real strides.

Design reference (idea only, no code copied): M16 Pack-GQA + LSE combine from the
code-study phase (quanbofeng); paged addressing follows the tutorial's
paged_attention_triton contract.

Fixed target: decode-only (q_len==1 per seq), BF16, D=128, Hq=64, Hkv=8,
group=8, block_size=16, SM90. Grid = [num_splits_max, num_kv_heads, num_seqs];
one warp (32 threads) per (split, kv_head, seq) handling the kv_head's 8 q_heads.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.nvgpu import warp

LOG2E = 1.4426950408889634

N_BLOCK = 64          # tokens per logical KV tile
BLOCK_SIZE = 16       # paged block size
TILES_BLOCKS = N_BLOCK // BLOCK_SIZE  # 4 logical blocks per tile
MMA_M = 16
HEADS_PER_KV = 8
HEAD_DIM = 128
WARP = 32
D_I64 = HEAD_DIM // 4  # 32 int64 per row (4 bf16 each) — used for vectorized-ish scalar copy


class LiuXiaochenPagedDecodeStage1V2:
    def __init__(self, num_seqs, num_heads, num_kv_heads, head_dim,
                 block_size, num_splits_max, split_size_tokens, n_block=64):
        assert head_dim == HEAD_DIM and block_size == BLOCK_SIZE
        assert num_heads == 64 and num_kv_heads == 8
        assert n_block == N_BLOCK
        self.num_seqs = num_seqs
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_splits_max = num_splits_max
        self.split_size_tokens = split_size_tokens
        self.n_block = n_block
        self.heads_per_kv = num_heads // num_kv_heads  # 8
        self.num_threads = WARP
        self.tiles_per_split = split_size_tokens // n_block

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,        # [num_tokens, num_heads, D]
        mKcache: cute.Tensor,   # [num_blocks, num_kv_heads, block_size, D]
        mVcache: cute.Tensor,   # [num_blocks, num_kv_heads, block_size, D]
        mPartial: cute.Tensor,  # [num_seqs, num_heads, num_splits_max, D] FP32
        mLSE: cute.Tensor,      # [num_seqs, num_heads, num_splits_max]     FP32
        mQStart: cute.Tensor,   # [num_seqs+1] int32
        mSeqLens: cute.Tensor,  # [num_seqs] int32
        mBlockTable: cute.Tensor,  # [num_seqs, max_num_blocks] int32
        scale_log2: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.dtype = mQ.element_type
        D = self.head_dim
        NB = self.n_block

        smem_k = 64
        smem_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, smem_k), stride=(smem_k, 1)),
        )
        q_smem_layout = cute.tile_to_shape(smem_atom, (MMA_M, D), (0, 1))
        kv_smem_layout = cute.tile_to_shape(smem_atom, (NB, D), (0, 1))

        @cute.struct
        class Smem:
            sQ: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(q_smem_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(kv_smem_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(kv_smem_layout)], 1024]

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1), permutation_mnk=(16, 16, 16),
        )
        self.kernel(
            mQ, mKcache, mVcache, mPartial, mLSE, mQStart, mSeqLens, mBlockTable,
            scale_log2, q_smem_layout, kv_smem_layout, tiled_mma, Smem,
        ).launch(
            grid=[self.num_splits_max, self.num_kv_heads, self.num_seqs],
            block=[self.num_threads, 1, 1], stream=stream,
        )

    @cute.kernel
    def kernel(self, mQ, mKcache, mVcache, mPartial, mLSE, mQStart, mSeqLens, mBlockTable,
               scale_log2, q_smem_layout, kv_smem_layout, tiled_mma, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        split_id, kv_head, seq_id = cute.arch.block_idx()
        D = self.head_dim
        NB = self.n_block
        HPK = self.heads_per_kv
        lane = tidx & 31

        seq_len = mSeqLens[seq_id]
        q_start = mQStart[seq_id]          # decode-only: this seq's single query token index
        split_start = split_id * self.split_size_tokens
        split_end_full = split_start + self.split_size_tokens
        split_end = split_end_full if split_end_full < seq_len else seq_len
        # absolute causal upper bound for a decode query = seq_len (attends [0, seq_len))
        # (K/V of the current token already written to cache before this kernel).

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Smem)
        sQ = storage.sQ.get_tensor(q_smem_layout)
        sK = storage.sK.get_tensor(kv_smem_layout)
        sV = storage.sV.get_tensor(kv_smem_layout)
        sVt = cute.composition(sV, cute.make_layout((D, NB), stride=(NB, 1)))

        # ---- Load Q: 8 q_heads of this kv_head into rows 0..7; zero rows 8..15 ----
        q_token = q_start
        for e in cutlass.range_constexpr((MMA_M * D + self.num_threads - 1) // self.num_threads):
            lin = e * self.num_threads + tidx
            if lin < MMA_M * D:
                row = lin // D
                col = lin % D
                if row < HPK:
                    q_head = kv_head * HPK + row
                    sQ[row, col] = mQ[q_token, q_head, col]
                else:
                    sQ[row, col] = cutlass.Float32(0.0).to(self.dtype)
        cute.arch.sync_warp()

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_o = cute.make_rmem_tensor(thr_mma.partition_shape_C((MMA_M, D)), cutlass.Float32)
        acc_o.fill(0.0)

        ldm_q = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
        ldm_k = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
        ldm_v = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype)
        tiled_q = cute.make_tiled_copy_A(ldm_q, tiled_mma)
        tiled_k = cute.make_tiled_copy_B(ldm_k, tiled_mma)
        tiled_v = cute.make_tiled_copy_B(ldm_v, tiled_mma)
        thr_q = tiled_q.get_slice(tidx); thr_k = tiled_k.get_slice(tidx); thr_v = tiled_v.get_slice(tidx)
        tSsQ = thr_q.partition_S(sQ); tSrQv = thr_q.retile(tSrQ)
        tSsK = thr_k.partition_S(sK); tSrKv = thr_k.retile(tSrK)
        tOsVt = thr_v.partition_S(sVt); tOrVtv = thr_v.retile(tOrVt)
        for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            cute.copy(tiled_q, tSsQ[None, None, kk], tSrQv[None, None, kk])

        row_count = acc_o.shape[0][0] * acc_o.shape[1]
        row_max = cute.make_rmem_tensor((row_count,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((row_count,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        # identity coords to know which (row,col) of acc_s this lane owns.
        ident_s = cute.make_identity_tensor((MMA_M, NB))
        coord_s = self._mn(thr_mma.partition_C(ident_s))

        num_tiles = (self.split_size_tokens + NB - 1) // NB
        any_valid = False
        for tstep in cutlass.range_constexpr(num_tiles):
            tile_tok0 = split_start + tstep * NB
            if tile_tok0 < split_end:
                any_valid = True
                self._load_paged_tile(sK, sV, mKcache, mVcache, mBlockTable,
                                      seq_id, kv_head, tile_tok0, seq_len, tidx)
                cute.arch.sync_warp()
                self._qk_pv_block(tiled_mma, thr_mma, tSrQ, tSrK, tOrVt, acc_o,
                                  tiled_k, tiled_v, tSsK, tSrKv, tOsVt, tOrVtv,
                                  row_max, row_sum, scale_log2, coord_s,
                                  tile_tok0, seq_len, first=(tstep == 0))
                cute.arch.sync_warp()

        self._epilogue(thr_mma, acc_o, row_max, row_sum, mPartial, mLSE,
                       seq_id, kv_head, split_id, scale_log2, any_valid)

    @cute.jit
    def _load_paged_tile(self, sK, sV, mKcache, mVcache, mBlockTable,
                         seq_id, kv_head, tile_tok0, seq_len, tidx):
        # Fill SMEM rows 0..63 in LOGICAL token order. Each of the 4 logical
        # 16-token sub-blocks is resolved through block_table to a physical block.
        # Out-of-range tokens (>= seq_len) are zero-filled (masked later in QK).
        D = self.head_dim
        # 64 rows * 128 cols = 8192 elems; 32 threads -> 256 elems/thread.
        total = self.n_block * D
        per_thread = (total + self.num_threads - 1) // self.num_threads
        for e in cutlass.range_constexpr(per_thread):
            lin = e * self.num_threads + tidx
            if lin < total:
                row = lin // D              # 0..63 logical token within tile
                col = lin % D
                tok = tile_tok0 + row
                if tok < seq_len:
                    logical_block = tok // self.block_size
                    slot = tok % self.block_size
                    pb = mBlockTable[seq_id, logical_block]
                    sK[row, col] = mKcache[pb, kv_head, slot, col]
                    sV[row, col] = mVcache[pb, kv_head, slot, col]
                else:
                    sK[row, col] = cutlass.Float32(0.0).to(self.dtype)
                    sV[row, col] = cutlass.Float32(0.0).to(self.dtype)

    @cute.jit
    def _qk_pv_block(self, tiled_mma, thr_mma, tSrQ, tSrK, tOrVt, acc_o,
                     tiled_k, tiled_v, tSsK, tSrKv, tOsVt, tOrVtv,
                     row_max, row_sum, scale_log2, coord_s, tile_tok0, seq_len,
                     first: cutlass.Constexpr):
        acc_s = cute.make_rmem_tensor(thr_mma.partition_shape_C((MMA_M, self.n_block)), cutlass.Float32)
        acc_s.fill(0.0)
        for kk in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
            cute.copy(tiled_k, tSsK[None, None, kk], tSrKv[None, None, kk])
        for kk in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):
            cute.gemm(tiled_mma, acc_s, tSrQ[None, None, kk], tSrK[None, None, kk], acc_s)

        # Mask columns whose absolute token >= seq_len to -inf (tail).
        scores_mn = self._mn(acc_s)
        for r in cutlass.range_constexpr(cute.size(row_max)):
            for c in cutlass.range_constexpr(cute.size(scores_mn.shape[1])):
                col_tok = tile_tok0 + coord_s[r, c][1]
                if col_tok >= seq_len:
                    scores_mn[r, c] = -cutlass.Float32.inf

        self._softmax(thr_mma, acc_o, acc_s, row_max, row_sum, scale_log2, first)

        probs = cute.make_fragment_like(acc_s, self.dtype)
        probs.store(acc_s.load().to(self.dtype))
        divided = cute.logical_divide(probs.layout, (None, None, 2))
        pv_a_layout = cute.make_layout(
            ((divided.shape[0], divided.shape[2][0]), divided.shape[1], divided.shape[2][1]),
            stride=((divided.stride[0], divided.stride[2][0]), divided.stride[1], divided.stride[2][1]),
        )
        pv_a = cute.make_tensor(probs.iterator, pv_a_layout)
        for kk in cutlass.range_constexpr(cute.size(tOsVt.shape[2])):
            cute.copy(tiled_v, tOsVt[None, None, kk], tOrVtv[None, None, kk])
        for kk in cutlass.range_constexpr(cute.size(pv_a.shape[2])):
            cute.gemm(tiled_mma, acc_o, pv_a[None, None, kk], tOrVt[None, None, kk], acc_o)

    @cute.jit
    def _softmax(self, thr_mma, acc_o, acc_s, row_max, row_sum, scale_log2, first: cutlass.Constexpr):
        scores_mn = self._mn(acc_s)
        out_mn = self._mn(acc_o)
        prev_max = cute.make_fragment_like(row_max, cutlass.Float32)
        if cutlass.const_expr(not first):
            cute.basic_copy(row_max, prev_max)
        for row in cutlass.range_constexpr(cute.size(row_max)):
            sc = scores_mn[row, None].load()
            cur = sc.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
            cur = self._quad_max(cur)
            if cutlass.const_expr(not first):
                cur = cute.arch.fmax(prev_max[row], cur)
            p = cute.math.exp2(sc * scale_log2 - cur * scale_log2, fastmath=True)
            s = p.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
            if cutlass.const_expr(not first):
                corr = cute.math.exp2((prev_max[row] - cur) * scale_log2, fastmath=True)
                s += row_sum[row] * corr
                out_mn[row, None] = out_mn[row, None].load() * corr
            row_max[row] = cur
            row_sum[row] = s
            scores_mn[row, None] = p

    @cute.jit
    def _epilogue(self, thr_mma, acc_o, row_max, row_sum, mPartial, mLSE,
                  seq_id, kv_head, split_id, scale_log2, any_valid: cutlass.Constexpr):
        out_mn = self._mn(acc_o)
        ident = cute.make_identity_tensor((MMA_M, self.head_dim))
        coord = self._mn(thr_mma.partition_C(ident))
        for row in cutlass.range_constexpr(cute.size(row_max)):
            denom = self._quad_sum(row_sum[row])
            # avoid div-by-zero for empty split rows
            safe = denom if denom > 0.0 else cutlass.Float32(1.0)
            out_mn[row, None] = out_mn[row, None].load() * cute.arch.rcp_approx(safe)
        for row in cutlass.range_constexpr(cute.size(row_max)):
            packed = coord[row, 0][0]
            fd = coord[row, 0][1]
            denom = self._quad_sum(row_sum[row])
            if fd == 0 and packed < self.heads_per_kv:
                q_head = kv_head * self.heads_per_kv + packed
                scale = scale_log2 / cutlass.Float32(LOG2E)
                if denom > 0.0:
                    mLSE[seq_id, q_head, split_id] = row_max[row] * scale + cute.math.log(denom, fastmath=True)
                else:
                    mLSE[seq_id, q_head, split_id] = -cutlass.Float32.inf
        cols = cute.size(coord.shape[1])
        for row in cutlass.range_constexpr(cute.size(row_max)):
            packed = coord[row, 0][0]
            if packed < self.heads_per_kv:
                q_head = kv_head * self.heads_per_kv + packed
                for col in cutlass.range_constexpr(cols):
                    dim = coord[row, col][1]
                    if dim < self.head_dim:
                        mPartial[seq_id, q_head, split_id, dim] = out_mn[row, col]

    def _mn(self, acc):
        c = cute.make_layout(acc.layout.shape)
        mn = cute.make_layout(
            ((c.shape[0][1], c.shape[1]), (c.shape[0][0], c.shape[2])),
            stride=((c.stride[0][1], c.stride[1]), (c.stride[0][0], c.stride[2])),
        )
        return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))

    def _quad(self, v, op):
        v = op(v, cute.arch.shuffle_sync_bfly(v, offset=2, mask=-1, mask_and_clamp=31))
        v = op(v, cute.arch.shuffle_sync_bfly(v, offset=1, mask=-1, mask_and_clamp=31))
        return v

    def _quad_max(self, v):
        return self._quad(v, lambda x, y: cute.arch.fmax(x, y))

    def _quad_sum(self, v):
        return self._quad(v, lambda x, y: x + y)
