#!/usr/bin/env python3
"""V3 Stage-1: direct paged-KV warp-MMA decode with cp.async 2-stage double buffer.

Independent implementation by Liu Xiaochen. This is V2 with EXACTLY ONE variable
changed: the synchronous per-element paged K/V load (V2 `_load_paged_tile`) is
replaced by a cp.async (CopyG2SOp) 2-stage double-buffered pipeline that prefetches
the next 64-token logical tile while the warp-MMA consumes the current one. The MMA
atoms, QK/PV fragment mapping, tail masking, online-softmax math, Split-KV schedule,
empty-split neutral representation and partial_o/partial_lse output format are
byte-for-byte identical to V2 (goal: V3 == V2 bit-identical).

Paged cp.async addressing: a 64-token logical tile = 4 logical KV blocks
(block_size=16). Each logical block j (j in 0..3) is resolved through
block_table[seq, (tile_tok0//16)+j] to an arbitrary physical block; the whole
[16,128] physical block is issued as an async copy into SMEM rows [16j:16j+16] in
LOGICAL order (no gather, no contiguous workspace, no physical-block sorting). All
4 K sub-copies + 4 V sub-copies of a tile form ONE cp.async commit group.

Tail / empty split: a fully-invalid logical block (block_start_tok >= seq_len) is
NOT resolved through block_table (no OOB address) and its SMEM rows are
deterministically zero-filled. A partially-valid block is loaded whole; its
out-of-range columns are masked to -inf in QK (score excluded from softmax denom),
identical to V2. cp.async / zero-fill branches are warp-uniform (block_start_tok
and seq_len are uniform across the 32 lanes), so no divergence.

Reference (idea only, no code copied): the author's own continuous-KV B6
(decode_mma_stage1_b6.py) cp.async 2-stage pipeline; M16 Pack-GQA + LSE combine
studied during the code-review phase (quanbofeng); paged addressing follows the
tutorial's paged_attention_triton contract.

Fixed target: decode-only (q_len==1 per seq), BF16, D=128, Hq=64, Hkv=8,
group=8, block_size=16, SM90. Grid = [num_splits_max, num_kv_heads, num_seqs];
one warp (32 threads) per (split, kv_head, seq) handling the kv_head's 8 q_heads.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.nvgpu import cpasync, warp

LOG2E = 1.4426950408889634

N_BLOCK = 64          # tokens per logical KV tile
BLOCK_SIZE = 16       # paged block size
TILES_BLOCKS = N_BLOCK // BLOCK_SIZE  # 4 logical blocks per tile
MMA_M = 16
HEADS_PER_KV = 8
HEAD_DIM = 128
WARP = 32


class LiuXiaochenPagedDecodeStage1V3:
    def __init__(self, num_seqs, num_heads, num_kv_heads, head_dim,
                 block_size, num_splits_max, split_size_tokens, n_block=64,
                 cache_mode=cpasync.LoadCacheMode.GLOBAL):
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
        self.cache_mode = cache_mode

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
        stage_cosize = cute.cosize(kv_smem_layout)

        @cute.struct
        class Smem:
            sQ: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(q_smem_layout)], 1024]
            # 2-stage K and V ring buffers (2K + 2V, matching B6; NOT B8-S's 2K+1V).
            sK: cute.struct.Align[cute.struct.MemRange[self.dtype, 2 * stage_cosize], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dtype, 2 * stage_cosize], 1024]

        # cp.async copy atom for one [BLOCK_SIZE, D] physical block (16-byte vectors).
        # Reuse B6's validated thread/value layout for this swizzle (smem_k=64,
        # threads_minor = smem_k//copy_elems = 8 -> thr_layout (4,8), val (1,8)).
        copy_bits = 128
        copy_elems = copy_bits // self.dtype.width  # 8 bf16
        blk_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=self.cache_mode), self.dtype, num_bits_per_copy=copy_bits
        )
        threads_minor = smem_k // copy_elems  # 64//8 = 8
        thr_layout = cute.make_layout(
            (self.num_threads // threads_minor, threads_minor), stride=(threads_minor, 1)
        )  # (4, 8)
        val_layout = cute.make_layout((1, copy_elems))  # (1, 8)
        blk_copy = cute.make_tiled_copy_tv(blk_atom, thr_layout, val_layout)

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1), permutation_mnk=(16, 16, 16),
        )
        self.kernel(
            mQ, mKcache, mVcache, mPartial, mLSE, mQStart, mSeqLens, mBlockTable,
            scale_log2, q_smem_layout, kv_smem_layout, stage_cosize, blk_copy, tiled_mma, Smem,
        ).launch(
            grid=[self.num_splits_max, self.num_kv_heads, self.num_seqs],
            block=[self.num_threads, 1, 1], stream=stream,
        )

    @cute.kernel
    def kernel(self, mQ, mKcache, mVcache, mPartial, mLSE, mQStart, mSeqLens, mBlockTable,
               scale_log2, q_smem_layout, kv_smem_layout, stage_cosize: cutlass.Constexpr,
               blk_copy, tiled_mma, Smem: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        split_id, kv_head, seq_id = cute.arch.block_idx()
        D = self.head_dim
        NB = self.n_block
        HPK = self.heads_per_kv

        seq_len = mSeqLens[seq_id]
        q_start = mQStart[seq_id]
        split_start = split_id * self.split_size_tokens
        split_end_full = split_start + self.split_size_tokens
        split_end = split_end_full if split_end_full < seq_len else seq_len

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Smem)
        sQ = storage.sQ.get_tensor(q_smem_layout)
        # Two K/V stages viewed as [NB, D] each.
        sK0 = storage.sK.get_tensor(kv_smem_layout)
        sK1 = cute.make_tensor(sK0.iterator + stage_cosize, kv_smem_layout)
        sV0 = storage.sV.get_tensor(kv_smem_layout)
        sV1 = cute.make_tensor(sV0.iterator + stage_cosize, kv_smem_layout)

        # ---- Load Q once (scalar, identical to V2; Q is one-time, not the bottleneck) ----
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
        acc_o = cute.make_rmem_tensor(thr_mma.partition_shape_C((MMA_M, D)), cutlass.Float32)
        acc_o.fill(0.0)

        ldm_q = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
        tiled_q = cute.make_tiled_copy_A(ldm_q, tiled_mma)
        thr_q = tiled_q.get_slice(tidx)
        tSsQ = thr_q.partition_S(sQ); tSrQv = thr_q.retile(tSrQ)
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

        # ---- Prologue: prefetch tile 0 into stage 0 ----
        self._prefetch_paged_tile(sK0, sV0, mKcache, mVcache, mBlockTable, blk_copy,
                                  seq_id, kv_head, split_start, seq_len, tidx)
        cute.arch.cp_async_commit_group()

        # ---- Steady state: depth-1 ping-pong (prefetch tile t+1 while computing tile t) ----
        for step in cutlass.range_constexpr(num_tiles):
            tile_tok0 = split_start + step * NB
            read_stage = step % 2
            if cutlass.const_expr(read_stage == 0):
                sK_cur, sV_cur = sK0, sV0
                sK_nxt, sV_nxt = sK1, sV1
            else:
                sK_cur, sV_cur = sK1, sV1
                sK_nxt, sV_nxt = sK0, sV0

            if cutlass.const_expr(step + 1 < num_tiles):
                next_tok0 = split_start + (step + 1) * NB
                self._prefetch_paged_tile(sK_nxt, sV_nxt, mKcache, mVcache, mBlockTable, blk_copy,
                                          seq_id, kv_head, next_tok0, seq_len, tidx)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()

            if tile_tok0 < split_end:
                self._qk_pv_block(tiled_mma, thr_mma, tSrQ, acc_o, sK_cur, sV_cur,
                                  row_max, row_sum, scale_log2, coord_s,
                                  tile_tok0, seq_len, first=(step == 0))
            cute.arch.sync_warp()

        self._epilogue(thr_mma, acc_o, row_max, row_sum, mPartial, mLSE,
                       seq_id, kv_head, split_id, scale_log2)

    @cute.jit
    def _prefetch_paged_tile(self, sK_stage, sV_stage, mKcache, mVcache, mBlockTable, blk_copy,
                             seq_id, kv_head, tile_tok0, seq_len, tidx):
        # Issue cp.async for the 4 logical blocks of this 64-token tile into
        # SMEM rows in LOGICAL order. Fully-invalid blocks are zero-filled and
        # never resolved through block_table (no OOB). Warp-uniform branches.
        D = self.head_dim
        g2s = blk_copy.get_slice(tidx)
        zero = cutlass.Float32(0.0).to(self.dtype)
        for j in cutlass.range_constexpr(TILES_BLOCKS):
            block_start_tok = tile_tok0 + j * self.block_size
            sK_sub = cute.local_tile(sK_stage, (self.block_size, D), (j, 0))
            sV_sub = cute.local_tile(sV_stage, (self.block_size, D), (j, 0))
            if block_start_tok < seq_len:
                logical_block = block_start_tok // self.block_size
                pb = mBlockTable[seq_id, logical_block]
                k_off = cute.crd2idx((pb, kv_head, 0, 0), mKcache.layout)
                v_off = cute.crd2idx((pb, kv_head, 0, 0), mVcache.layout)
                gK_sub = cute.make_tensor(
                    cute.make_ptr(self.dtype, (mKcache.iterator + k_off).toint(),
                                  cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((self.block_size, D),
                                     stride=(mKcache.stride[2], mKcache.stride[3])),
                )
                gV_sub = cute.make_tensor(
                    cute.make_ptr(self.dtype, (mVcache.iterator + v_off).toint(),
                                  cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout((self.block_size, D),
                                     stride=(mVcache.stride[2], mVcache.stride[3])),
                )
                cute.copy(blk_copy, g2s.partition_S(gK_sub), g2s.partition_D(sK_sub))
                cute.copy(blk_copy, g2s.partition_S(gV_sub), g2s.partition_D(sV_sub))
            else:
                els = self.block_size * D
                for e in cutlass.range_constexpr((els + self.num_threads - 1) // self.num_threads):
                    lin = e * self.num_threads + tidx
                    if lin < els:
                        r = lin // D
                        c = lin % D
                        sK_sub[r, c] = zero
                        sV_sub[r, c] = zero

    @cute.jit
    def _qk_pv_block(self, tiled_mma, thr_mma, tSrQ, acc_o, sK_cur, sV_cur,
                     row_max, row_sum, scale_log2, coord_s, tile_tok0, seq_len,
                     first: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        D = self.head_dim
        NB = self.n_block
        sVt = cute.composition(sV_cur, cute.make_layout((D, NB), stride=(NB, 1)))

        # Per-stage fragments + ldmatrix copies (K/V live in the current stage).
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK_cur))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        ldm_k = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
        ldm_v = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype)
        tiled_k = cute.make_tiled_copy_B(ldm_k, tiled_mma)
        tiled_v = cute.make_tiled_copy_B(ldm_v, tiled_mma)
        thr_k = tiled_k.get_slice(tidx); thr_v = tiled_v.get_slice(tidx)
        tSsK = thr_k.partition_S(sK_cur); tSrKv = thr_k.retile(tSrK)
        tOsVt = thr_v.partition_S(sVt); tOrVtv = thr_v.retile(tOrVt)

        acc_s = cute.make_rmem_tensor(thr_mma.partition_shape_C((MMA_M, NB)), cutlass.Float32)
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
                  seq_id, kv_head, split_id, scale_log2):
        out_mn = self._mn(acc_o)
        ident = cute.make_identity_tensor((MMA_M, self.head_dim))
        coord = self._mn(thr_mma.partition_C(ident))
        for row in cutlass.range_constexpr(cute.size(row_max)):
            denom = self._quad_sum(row_sum[row])
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
