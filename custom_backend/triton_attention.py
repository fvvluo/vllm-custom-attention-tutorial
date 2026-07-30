# SPDX-License-Identifier: Apache-2.0
"""
FP8 (e4m3) 分页注意力 Triton kernel —— tensor-core tl.dot 版
================================================================

在教程默认"逐 token 标量"实现之上做两处优化，接口/语义完全不变
（`paged_attention_triton(...)`，可直接替换接入 vLLM CUSTOM 后端）：

  1. **真正的 fp8 tensor-core 矩阵乘**：把 QK^T / PV 从标量 `tl.sum` 改成 `tl.dot`。
     `tl.dot` 在 fp8(e4m3) 输入上走 Hopper WGMMA tensor core（H20 上 fp8 吞吐约 bf16 2x）。
     要让 tensor core 生效，必须凑出 M×K×N 的矩阵 tile：
       - 沿用本仓库 fp8_attn/ 的 GQA pack 思路——把共享同一个 KV 头的 GROUP(=num_heads/
         num_kv_heads) 个 Q 头当作 **M 维**（同一 token 的这些头 causal 上界相同，掩码统一）。
       - grid 从 (num_tokens × num_heads) 收紧为 **(num_tokens × num_kv_heads)**，program 数减少
         GROUP 倍，每个 program 做 [M,HEAD_SIZE]·[HEAD_SIZE,BLOCK_N] 的真实矩阵乘。
       - M 会 pad 到 16（tensor core 最小 tile），HEAD_SIZE=128 作 K，BLOCK_N 作 N。

  2. **分页读取（"我写的 KV cache"）**：按 block_table 把逻辑 KV 跳映射到物理块，沿 KV
     分块（BLOCK_N=block_size）读取（移植自本仓库 paged_kv/）。

数值：online softmax 全程 fp32；fp8 只用于 QK^T 的两个矩阵乘输入，PV 用 bf16 dot（P 靠近 0，
量化会放大误差）。use_fp8=False 时 QK 用 bf16 tl.dot 对照。

布局约定（与教程 / vLLM TRITON_ATTN 一致）：
  key_cache / value_cache: [num_blocks, num_kv_heads, block_size, head_size]，按 stride 寻址。
"""

import torch
import triton
import triton.language as tl

import os

_E4M3_MAX = tl.constexpr(448.0)
_FP8_DTYPE = tl.float8e4nv

# ---- 稀疏 decode 配置（env-gated，默认关；开启后仅作用于纯 decode + bf16 KV）----
_SPARSE_ENABLED = os.environ.get("CUSTOM_SPARSE", "0") == "1"
_SPARSITY = float(os.environ.get("CUSTOM_SPARSITY", "0.25"))
_SPARSE_MIN_LEN = int(os.environ.get("CUSTOM_SPARSE_MIN_LEN", "2048"))
_SPARSE_SINK = int(os.environ.get("CUSTOM_SPARSE_SINK", "1"))
_SPARSE_RECENT = int(os.environ.get("CUSTOM_SPARSE_RECENT", "8"))

# ---- Prefill 调优（env，可扫参）。实测最优: TILE_Q=64 warps=4 stages=3（软流水更深，8k prefill 11.1→9.0ms）----
_PF_TILE_Q = int(os.environ.get("CUSTOM_PF_TILE_Q", "64"))
_PF_WARPS = int(os.environ.get("CUSTOM_PF_WARPS", "4"))
_PF_STAGES = int(os.environ.get("CUSTOM_PF_STAGES", "3"))
# GQA-pack prefill（默认开）：group 个 q 头打进 M 维，每 KV block 只读一次（消 8x 冗余读）。
# GQA-pack 用较小 TILE_Q（M=GROUP*TILE_Q，避免过大 tile 溢出）；实测 TILE_Q=8 最优。
_PF_GQA_PACK = os.environ.get("CUSTOM_PF_GQA_PACK", "1") == "1"
_PF_GQA_TILE_Q = int(os.environ.get("CUSTOM_PF_GQA_TILE_Q", "8"))
# 滑窗 prefill（CUSTOM_PF_WINDOW>0 开启）：每 query 只 attend 最近 WINDOW token + 开头 SINK token，
# prefill O(seq^2)→O(seq*WINDOW)，是 100k TTFT<50s 的关键；超窗口长程依赖会丢（近似）。仅 GQA-pack 生效。
_PF_WINDOW = int(os.environ.get("CUSTOM_PF_WINDOW", "0"))
_PF_SINK = int(os.environ.get("CUSTOM_PF_SINK", "256"))

# ---- Decode 调优（env，可扫参）。decode split-KV kernel 的 launch 参数：----
# 长上下文 decode 是**KV 带宽受限**，split-KV（下方）已把带宽打满（98k 实测 1.85 TB/s）。
# 实测 warps/stages 对 decode 影响很小（bandwidth-bound）：warps=4 已足够占满，num_stages
# 加深软流水**几乎无收益甚至略降**（98k: s2=0.227 vs s3=0.228 ms）。故默认 warps=4 stages=2。
# 保留为 env 旋钮方便复现扫参。只改 launch 参数、不改数值语义，不影响正确性。
_DEC_WARPS = int(os.environ.get("CUSTOM_DEC_WARPS", "4"))
_DEC_STAGES = int(os.environ.get("CUSTOM_DEC_STAGES", "2"))
# NUM_SPLITS 覆盖（>0 时强制，用于基准对比：设 1 即模拟"优化前无 split-KV"）。默认 0=自动。
_DEC_SPLITS = int(os.environ.get("CUSTOM_DEC_SPLITS", "0"))


@triton.jit
def _fp8_paged_attn_kernel(
    out_ptr,              # [num_tokens, num_heads, head_size]
    q_ptr,                # [num_tokens, num_heads, head_size]
    k_cache_ptr,          # [num_blocks, num_kv_heads, block_size, head_size]
    v_cache_ptr,
    query_start_loc_ptr,  # [num_seqs + 1]
    seq_lens_ptr,         # [num_seqs]
    token_seq_idx_ptr,    # [num_tokens]
    block_table_ptr,      # [num_seqs, max_num_blocks]
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,    # 物理块槽位数（page block_size）
    BLOCK_N: tl.constexpr,       # KV tile 宽度（= PAGES_PER_TILE * BLOCK_SIZE，>=16）
    PAGES_PER_TILE: tl.constexpr,  # 每个 KV tile 聚合多少个物理块
    BLOCK_M: tl.constexpr,       # >= GROUP，pad 到 tensor-core tile（16）
    GROUP: tl.constexpr,         # num_heads // num_kv_heads
    max_num_blocks: tl.constexpr,
    q_stride_t, q_stride_h,
    o_stride_t, o_stride_h,
    kc_stride_b, kc_stride_h, kc_stride_s,
    vc_stride_b, vc_stride_h, vc_stride_s,
    block_table_stride,
    k_descale_ptr,               # 指向 fp8 KV cache 的 K descale（1 元素张量；避免 host .item() 同步以兼容 CUDA graph）
    v_descale_ptr,               # 指向 V descale
    NUM_SPLITS: tl.constexpr,    # split-KV：把每个 (token,kv_head) 的 KV 扫描切成多少段并行
    op_ptr,                      # split 部分结果 acc：[num_tokens, kv_heads, NUM_SPLITS, BLOCK_M, HEAD_SIZE]
    ml_ptr,                      # split 的 (m, l)：[num_tokens, kv_heads, NUM_SPLITS, 2, BLOCK_M]
    op_st, op_sh, op_ss, op_sm,  # op strides (token,head,split,m)
    ml_st, ml_sh, ml_ss, ml_se,  # ml strides (token,head,split,elem)
    USE_FP8: tl.constexpr,       # QK^T 是否走 fp8 tensor-core（bf16 KV 时的动态量化路径）
    KV_IS_FP8: tl.constexpr,     # KV cache 是否已是 e4m3（预量化常驻）
):
    # grid = (num_tokens, num_kv_heads, NUM_SPLITS)：每个 program 负责一个 (token, kv_head)
    # 的**一段** KV，把该 kv_head 下 GROUP 个 q 头一起算（M 维）。split-KV/flash-decoding：
    # bs=1 时原本 grid 只有 (1,8)=8 CTA，GPU 严重闲置 + 每 CTA 串行扫 100k KV（实测仅 0.16TB/s）；
    # 拆成 NUM_SPLITS 段后 grid=8*NUM_SPLITS，填满 SM、打满带宽，再由 combine kernel 跨段合并。
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    pid_split = tl.program_id(2)

    # ---- 1. 定位请求与 causal 上界（GROUP 个头共享同一 token，上界相同）----
    seq_idx = tl.load(token_seq_idx_ptr + token_idx)
    q_start = tl.load(query_start_loc_ptr + seq_idx)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    query_len = tl.load(query_start_loc_ptr + seq_idx + 1) - q_start
    idx_in_query = token_idx - q_start
    context_len = seq_len - query_len
    abs_pos = context_len + idx_in_query
    kv_upper = abs_pos + 1

    offs_m = tl.arange(0, BLOCK_M)          # M 维：GROUP 个 q 头（pad 到 BLOCK_M）
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_n = tl.arange(0, BLOCK_N)
    m_valid = offs_m < GROUP
    head_ids = kv_head_idx * GROUP + offs_m  # 每行对应的绝对 q 头号

    # fp8 KV descale：从张量指针里读标量（不在 host 上 .item()，以兼容 CUDA graph 捕获）
    if KV_IS_FP8:
        k_descale_val = tl.load(k_descale_ptr)
        v_descale_val = tl.load(v_descale_ptr)

    # ---- 2. 载入 Q tile [BLOCK_M, HEAD_SIZE] 并量化 ----
    q_ptrs = (q_ptr + token_idx * q_stride_t
              + head_ids[:, None] * q_stride_h + offs_d[None, :])
    q_f = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0).to(tl.float32)
    # 只要 QK 想用 fp8（in-kernel 动态量化 KV，或 KV 本身已是 fp8），Q 都量化到 e4m3。
    if USE_FP8 or KV_IS_FP8:
        q_amax = tl.maximum(tl.max(tl.abs(q_f), axis=1), 1e-12)     # [BLOCK_M]
        q_descale = q_amax / _E4M3_MAX
        q_fp8 = (q_f * (_E4M3_MAX / q_amax)[:, None]).to(_FP8_DTYPE)
    else:
        q_bf = q_f.to(tl.bfloat16)

    # ---- 3. 沿 KV 分块（每块一个物理块）做在线 softmax ----
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], tl.float32)

    # 每个 KV tile 聚合 PAGES_PER_TILE 个物理块 -> N=BLOCK_N。
    # 【带宽优化：3D tile 载入】block_table 的 gather 只发生在 **page 维**（每 tile 只查
    # PAGES_PER_TILE 个物理块号），而每个物理块内部的 [BLOCK_SIZE, HEAD_SIZE] 在显存里是
    # **连续**的（布局 [num_blocks, kv_heads, block_size, head_size]，同一 (block,kv_head)
    # 连续）。于是把指针组织成 [PAGES, BLOCK_SIZE, HEAD_SIZE] 的 3D 形状：page 维用 gather
    # 的物理块基址、内两维用连续 stride —— Triton 可对每页内 16×128 做合并/向量化访存，
    # 而非之前 [BLOCK_N,HEAD_SIZE] 的逐列 scatter。载入后 reshape 成 [BLOCK_N, HEAD_SIZE]。
    num_tiles = (kv_upper + BLOCK_N - 1) // BLOCK_N
    # 本 split 负责的 tile 区间 [t_start, t_end)（按 tile 粒度均分给 NUM_SPLITS 段）
    tiles_per_split = (num_tiles + NUM_SPLITS - 1) // NUM_SPLITS
    t_start = pid_split * tiles_per_split
    t_end = tl.minimum(t_start + tiles_per_split, num_tiles)
    offs_p = tl.arange(0, PAGES_PER_TILE)        # [PAGES]
    offs_s = tl.arange(0, BLOCK_SIZE)            # [BLOCK_SIZE] 页内槽位（连续）
    for t in range(t_start, t_end):
        n0 = t * BLOCK_N
        blk0 = t * PAGES_PER_TILE
        logical_blk = blk0 + offs_p               # [PAGES] 每页的逻辑块号
        pb = tl.load(block_table_ptr + seq_idx * block_table_stride + logical_blk,
                     mask=logical_blk < max_num_blocks, other=0)   # [PAGES] 物理块号
        # 3D 指针 [PAGES, BLOCK_SIZE, HEAD_SIZE]：page 维 gather 基址；页内两维连续 stride
        k_ptrs = (k_cache_ptr
                  + pb[:, None, None] * kc_stride_b + kv_head_idx * kc_stride_h
                  + offs_s[None, :, None] * kc_stride_s + offs_d[None, None, :])
        v_ptrs = (v_cache_ptr
                  + pb[:, None, None] * vc_stride_b + kv_head_idx * vc_stride_h
                  + offs_s[None, :, None] * vc_stride_s + offs_d[None, None, :])
        cols = n0 + offs_n
        col_active = cols < kv_upper
        # 每页是否越过 max_num_blocks（padding 页整块屏蔽）
        page_valid = logical_blk < max_num_blocks       # [PAGES]
        load_mask = page_valid[:, None, None] & (offs_s[None, :, None] >= 0)

        if KV_IS_FP8:
            # KV cache 已是 e4m3（预量化常驻）：3D 合并载入后 reshape 成 [BLOCK_N, HEAD_SIZE]。
            k_fp8 = tl.load(k_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE)
            v_fp8 = tl.load(v_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE)
            qk = tl.dot(q_fp8, k_fp8.T, out_dtype=tl.float32)
            qk = qk * (q_descale[:, None] * k_descale_val * scale)
        elif USE_FP8:
            # bf16 KV + in-kernel 动态量化到 e4m3 做 fp8 QK
            k_f = tl.load(k_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.float32)
            v_bf = tl.load(v_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.bfloat16)
            k_amax = tl.maximum(tl.max(tl.abs(k_f)), 1e-12)
            k_descale = k_amax / _E4M3_MAX
            k_fp8 = (k_f * (_E4M3_MAX / k_amax)).to(_FP8_DTYPE)
            qk = tl.dot(q_fp8, k_fp8.T, out_dtype=tl.float32)
            qk = qk * (q_descale[:, None] * k_descale * scale)
        else:
            # 纯 bf16 路径
            k_bf = tl.load(k_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.bfloat16)
            v_bf = tl.load(v_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.bfloat16)
            qk = tl.dot(q_bf, k_bf.T, out_dtype=tl.float32) * scale

        qk = tl.where(col_active[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])                            # [BLOCK_M, BLOCK_N]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        # PV 用 bf16 tensor-core：[M,N]·[N,D] -> [M,D]。
        # KV_IS_FP8 时 V 是 e4m3，先反量化回 bf16（乘 v_descale_val）再算，避免 P 量化误差。
        if KV_IS_FP8:
            v_bf = (v_fp8.to(tl.float32) * v_descale_val).to(tl.bfloat16)
        acc += tl.dot(p.to(tl.bfloat16), v_bf, out_dtype=tl.float32)
        m_i = m_new

    # 写本 split 的部分结果：未归一化的 acc（= Σ exp(s-m_i)·v）+ (m_i, l_i)，供 combine 跨 split 合并。
    op_base = (op_ptr + token_idx * op_st + kv_head_idx * op_sh + pid_split * op_ss)
    o_ptrs = op_base + offs_m[:, None] * op_sm + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=m_valid[:, None])
    ml_base = (ml_ptr + token_idx * ml_st + kv_head_idx * ml_sh + pid_split * ml_ss)
    tl.store(ml_base + 0 * ml_se + offs_m, m_i, mask=m_valid)   # m
    tl.store(ml_base + 1 * ml_se + offs_m, l_i, mask=m_valid)   # l


@triton.jit
def _fp8_decode_combine_kernel(
    out_ptr, op_ptr, ml_ptr,
    num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
    HEAD_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, GROUP: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    o_stride_t, o_stride_h,
    op_st, op_sh, op_ss, op_sm,
    ml_st, ml_sh, ml_ss, ml_se,
):
    # grid = (num_tokens, num_kv_heads)：跨 NUM_SPLITS 段做在线-softmax 合并。
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_sp = tl.arange(0, NUM_SPLITS)
    m_valid = offs_m < GROUP
    head_ids = kv_head_idx * GROUP + offs_m

    ml_base = ml_ptr + token_idx * ml_st + kv_head_idx * ml_sh
    # 各 split 的 m、l：[NUM_SPLITS, BLOCK_M]
    m_s = tl.load(ml_base + offs_sp[:, None] * ml_ss + 0 * ml_se + offs_m[None, :])
    l_s = tl.load(ml_base + offs_sp[:, None] * ml_ss + 1 * ml_se + offs_m[None, :])
    m_glob = tl.max(m_s, axis=0)                                # [BLOCK_M]
    alpha = tl.exp(m_s - m_glob[None, :])                       # [NUM_SPLITS, BLOCK_M]
    l_glob = tl.sum(alpha * l_s, axis=0)                        # [BLOCK_M]
    # 加权合并各 split 的未归一化 acc
    op_base = op_ptr + token_idx * op_st + kv_head_idx * op_sh
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], tl.float32)
    for s in range(0, NUM_SPLITS):
        o_s = tl.load(op_base + s * op_ss + offs_m[:, None] * op_sm + offs_d[None, :])
        w = tl.exp(tl.load(ml_base + s * ml_ss + 0 * ml_se + offs_m) - m_glob)  # [BLOCK_M]
        acc += o_s * w[:, None]
    acc = acc / l_glob[:, None]
    o_ptrs = out_ptr + token_idx * o_stride_t + head_ids[:, None] * o_stride_h + offs_d[None, :]
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_valid[:, None])


@triton.jit
def _fp8_prefill_gqa_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    query_start_loc_ptr,  # [num_seqs + 1]
    seq_lens_ptr,         # [num_seqs]
    block_table_ptr,      # [num_seqs, max_num_blocks]
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_Q: tl.constexpr,        # 每个 program 处理的 query token 数
    GROUP: tl.constexpr,         # num_heads // num_kv_heads（打进 M 维）
    BLOCK_N: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    max_num_blocks: tl.constexpr,
    q_stride_t, q_stride_h,
    o_stride_t, o_stride_h,
    kc_stride_b, kc_stride_h, kc_stride_s,
    vc_stride_b, vc_stride_h, vc_stride_s,
    block_table_stride,
    k_descale_val, v_descale_val,
    WINDOW_TILES: tl.constexpr,  # 滑窗宽度（KV tile 数）；0=全 causal（不滑窗）
    SINK_TILES: tl.constexpr,    # 保留开头 sink 块（KV tile 数）
    USE_FP8: tl.constexpr,
    KV_IS_FP8: tl.constexpr,
):
    # grid = (num_seqs, num_q_tiles, num_kv_heads)：把一个 kv_head 下 GROUP 个 q 头打进 M 维。
    # M = TILE_Q * GROUP，行 m 对应 (tq = m//GROUP, g = m%GROUP)：query 位置 row0+tq、q 头 kvh*GROUP+g。
    # **每个 KV block 只被本 program 读一次**（原 kernel 按 num_heads grid → 每块被读 GROUP=8 次冗余）。
    # causal 上界只依赖 query 位置 tq（同 tq 的 GROUP 行共享），掩码统一。
    s = tl.program_id(0)
    qt = tl.program_id(1)
    kv_head_idx = tl.program_id(2)

    q_start = tl.load(query_start_loc_ptr + s)
    q_end = tl.load(query_start_loc_ptr + s + 1)
    query_len = q_end - q_start
    seq_len = tl.load(seq_lens_ptr + s)
    context_len = seq_len - query_len

    row0 = qt * TILE_Q
    if row0 >= query_len:
        return

    BLOCK_M: tl.constexpr = TILE_Q * GROUP
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_n = tl.arange(0, BLOCK_N)
    tq = offs_m // GROUP                        # 每行的 query-tile 内位置
    g = offs_m % GROUP                          # 每行的 group 内 q 头偏移
    rows = row0 + tq                            # query 行（seq 内 0-based）
    row_valid = rows < query_len
    abs_pos = context_len + rows                # 每行绝对位置（causal 上界，只依赖 tq）
    head_ids = kv_head_idx * GROUP + g          # 每行对应的绝对 q 头

    q_ptrs = (q_ptr + (q_start + rows)[:, None] * q_stride_t
              + head_ids[:, None] * q_stride_h + offs_d[None, :])
    q_f = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0).to(tl.float32)
    if USE_FP8 or KV_IS_FP8:
        q_amax = tl.maximum(tl.max(tl.abs(q_f), axis=1), 1e-12)
        q_descale = q_amax / _E4M3_MAX
        q_fp8 = (q_f * (_E4M3_MAX / q_amax)[:, None]).to(_FP8_DTYPE)
    else:
        q_bf = q_f.to(tl.bfloat16)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE], tl.float32)

    kv_end = context_len + tl.minimum(row0 + TILE_Q, query_len)
    num_tiles = (kv_end + BLOCK_N - 1) // BLOCK_N
    offs_p = tl.arange(0, PAGES_PER_TILE)
    offs_s = tl.arange(0, BLOCK_SIZE)

    # 滑窗 + sink：WINDOW_TILES==0 时全 causal（visit 0..num_tiles-1）；否则只 visit
    #   sink tiles [0, SINK_TILES) + window tiles [win_lo, num_tiles)。
    # 用固定 trip-count 循环（SINK_TILES+WINDOW_TILES）+ 迭代号→真实 tile 映射，把 O(seq^2)
    # 降到 O(seq*W)。win_lo 由本 q-tile 最远位置决定，超窗口的远块被跳过（近似）。
    if WINDOW_TILES == 0:
        n_iter = num_tiles
        win_lo = 0
    else:
        win_lo = tl.maximum(num_tiles - WINDOW_TILES, 0)
        n_iter = SINK_TILES + tl.minimum(WINDOW_TILES, num_tiles)
    for it in range(0, n_iter):
        if WINDOW_TILES == 0:
            t = it
        else:
            # 前 SINK_TILES 次访问 sink 块；其余访问窗口块 [win_lo, num_tiles)
            t = tl.where(it < SINK_TILES, it, win_lo + (it - SINK_TILES))
        # sink 与窗口可能重叠（短序列）；重叠块被重复算会重复计入 softmax → 用 skip 标记屏蔽
        dup = (WINDOW_TILES != 0) and (it >= SINK_TILES) and (t < SINK_TILES)
        n0 = t * BLOCK_N
        blk0 = t * PAGES_PER_TILE
        logical_blk = blk0 + offs_p
        pb = tl.load(block_table_ptr + s * block_table_stride + logical_blk,
                     mask=logical_blk < max_num_blocks, other=0)
        k_ptrs = (k_cache_ptr + pb[:, None, None] * kc_stride_b + kv_head_idx * kc_stride_h
                  + offs_s[None, :, None] * kc_stride_s + offs_d[None, None, :])
        v_ptrs = (v_cache_ptr + pb[:, None, None] * vc_stride_b + kv_head_idx * vc_stride_h
                  + offs_s[None, :, None] * vc_stride_s + offs_d[None, None, :])
        cols = n0 + offs_n
        page_valid = logical_blk < max_num_blocks
        load_mask = page_valid[:, None, None] & (offs_s[None, :, None] >= 0)

        if KV_IS_FP8:
            k_fp8 = tl.load(k_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE)
            v_fp8 = tl.load(v_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE)
            qk = tl.dot(q_fp8, k_fp8.T, out_dtype=tl.float32) * (q_descale[:, None] * k_descale_val * scale)
        elif USE_FP8:
            k_f = tl.load(k_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.float32)
            v_bf = tl.load(v_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.bfloat16)
            k_amax = tl.maximum(tl.max(tl.abs(k_f)), 1e-12)
            k_descale = k_amax / _E4M3_MAX
            k_fp8 = (k_f * (_E4M3_MAX / k_amax)).to(_FP8_DTYPE)
            qk = tl.dot(q_fp8, k_fp8.T, out_dtype=tl.float32) * (q_descale[:, None] * k_descale * scale)
        else:
            k_bf = tl.load(k_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.bfloat16)
            v_bf = tl.load(v_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.bfloat16)
            qk = tl.dot(q_bf, k_bf.T, out_dtype=tl.float32) * scale

        causal_ok = (cols[None, :] <= abs_pos[:, None]) & (cols[None, :] < seq_len)
        if WINDOW_TILES != 0:
            causal_ok = causal_ok & (not dup)
        qk = tl.where(causal_ok, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if KV_IS_FP8:
            v_bf = (v_fp8.to(tl.float32) * v_descale_val).to(tl.bfloat16)
        acc += tl.dot(p.to(tl.bfloat16), v_bf, out_dtype=tl.float32)
        m_i = m_new

    out = acc / l_i[:, None]
    o_ptrs = (out_ptr + (q_start + rows)[:, None] * o_stride_t
              + head_ids[:, None] * o_stride_h + offs_d[None, :])
    tl.store(o_ptrs, out.to(out_ptr.dtype.element_ty), mask=row_valid[:, None])


@triton.jit
def _fp8_prefill_kernel(
    out_ptr, q_ptr, k_cache_ptr, v_cache_ptr,
    query_start_loc_ptr,  # [num_seqs + 1]
    seq_lens_ptr,         # [num_seqs]
    block_table_ptr,      # [num_seqs, max_num_blocks]
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_Q: tl.constexpr,        # 每个 program 处理的 query token 数（M 维）
    BLOCK_N: tl.constexpr,       # KV tile 宽度
    PAGES_PER_TILE: tl.constexpr,
    max_num_blocks: tl.constexpr,
    q_stride_t, q_stride_h,
    o_stride_t, o_stride_h,
    kc_stride_b, kc_stride_h, kc_stride_s,
    vc_stride_b, vc_stride_h, vc_stride_s,
    block_table_stride,
    k_descale_val, v_descale_val,
    USE_FP8: tl.constexpr,
    KV_IS_FP8: tl.constexpr,
):
    # grid = (num_seqs, num_q_tiles_per_seq, num_q_heads)：
    #   每个 program 处理 seq s、第 qt 个 query-tile（TILE_Q 行）、q 头 h。
    # 这是 flash-attention 风格的 query 分块：M=TILE_Q 大 tile 打满 tensor core，
    # 每个 KV block 只被本 q-tile 读一次（而非每 token 一次），并用 causal 三角裁剪。
    s = tl.program_id(0)
    qt = tl.program_id(1)
    h = tl.program_id(2)

    q_start = tl.load(query_start_loc_ptr + s)
    q_end = tl.load(query_start_loc_ptr + s + 1)
    query_len = q_end - q_start
    seq_len = tl.load(seq_lens_ptr + s)
    context_len = seq_len - query_len            # 该 seq 已有 context

    row0 = qt * TILE_Q
    if row0 >= query_len:                          # 本 tile 超出该 seq 的 query 范围
        return

    offs_m = tl.arange(0, TILE_Q)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_n = tl.arange(0, BLOCK_N)
    rows = row0 + offs_m                            # query 行（seq 内 0-based）
    row_valid = rows < query_len
    abs_pos = context_len + rows                    # 每行的绝对位置（causal 上界）
    kv_head_idx = h // (num_heads // num_kv_heads)

    # 载入 Q tile [TILE_Q, HEAD_SIZE]
    q_ptrs = (q_ptr + (q_start + rows)[:, None] * q_stride_t
              + h * q_stride_h + offs_d[None, :])
    q_f = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0).to(tl.float32)
    if USE_FP8 or KV_IS_FP8:
        q_amax = tl.maximum(tl.max(tl.abs(q_f), axis=1), 1e-12)
        q_descale = q_amax / _E4M3_MAX
        q_fp8 = (q_f * (_E4M3_MAX / q_amax)[:, None]).to(_FP8_DTYPE)
    else:
        q_bf = q_f.to(tl.bfloat16)

    m_i = tl.full([TILE_Q], -float("inf"), tl.float32)
    l_i = tl.zeros([TILE_Q], tl.float32)
    acc = tl.zeros([TILE_Q, HEAD_SIZE], tl.float32)

    # causal 三角裁剪：本 tile 的最大绝对位置决定 KV 扫描上界
    kv_end = context_len + tl.minimum(row0 + TILE_Q, query_len)   # 本 tile 最远可见 kv + 1
    num_tiles = (kv_end + BLOCK_N - 1) // BLOCK_N
    offs_p = tl.arange(0, PAGES_PER_TILE)
    offs_s = tl.arange(0, BLOCK_SIZE)
    for t in range(0, num_tiles):
        n0 = t * BLOCK_N
        blk0 = t * PAGES_PER_TILE
        logical_blk = blk0 + offs_p
        pb = tl.load(block_table_ptr + s * block_table_stride + logical_blk,
                     mask=logical_blk < max_num_blocks, other=0)
        k_ptrs = (k_cache_ptr + pb[:, None, None] * kc_stride_b + kv_head_idx * kc_stride_h
                  + offs_s[None, :, None] * kc_stride_s + offs_d[None, None, :])
        v_ptrs = (v_cache_ptr + pb[:, None, None] * vc_stride_b + kv_head_idx * vc_stride_h
                  + offs_s[None, :, None] * vc_stride_s + offs_d[None, None, :])
        cols = n0 + offs_n                          # kv 绝对位置
        page_valid = logical_blk < max_num_blocks
        load_mask = page_valid[:, None, None] & (offs_s[None, :, None] >= 0)

        if KV_IS_FP8:
            k_fp8 = tl.load(k_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE)
            v_fp8 = tl.load(v_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE)
            qk = tl.dot(q_fp8, k_fp8.T, out_dtype=tl.float32) * (q_descale[:, None] * k_descale_val * scale)
        elif USE_FP8:
            k_f = tl.load(k_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.float32)
            v_bf = tl.load(v_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.bfloat16)
            k_amax = tl.maximum(tl.max(tl.abs(k_f)), 1e-12)
            k_descale = k_amax / _E4M3_MAX
            k_fp8 = (k_f * (_E4M3_MAX / k_amax)).to(_FP8_DTYPE)
            qk = tl.dot(q_fp8, k_fp8.T, out_dtype=tl.float32) * (q_descale[:, None] * k_descale * scale)
        else:
            k_bf = tl.load(k_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.bfloat16)
            v_bf = tl.load(v_ptrs, mask=load_mask, other=0.0).reshape(BLOCK_N, HEAD_SIZE).to(tl.bfloat16)
            qk = tl.dot(q_bf, k_bf.T, out_dtype=tl.float32) * scale

        # causal + 边界 mask：kv 列必须 <= 本行绝对位置，且在 [0, seq_len)
        causal_ok = (cols[None, :] <= abs_pos[:, None]) & (cols[None, :] < seq_len)
        qk = tl.where(causal_ok, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if KV_IS_FP8:
            v_bf = (v_fp8.to(tl.float32) * v_descale_val).to(tl.bfloat16)
        acc += tl.dot(p.to(tl.bfloat16), v_bf, out_dtype=tl.float32)
        m_i = m_new

    out = acc / l_i[:, None]
    o_ptrs = (out_ptr + (q_start + rows)[:, None] * o_stride_t
              + h * o_stride_h + offs_d[None, :])
    tl.store(o_ptrs, out.to(out_ptr.dtype.element_ty), mask=row_valid[:, None])


def paged_attention_triton(
    query: torch.Tensor,        # [num_tokens, num_heads, head_size]
    key_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size]
    output: torch.Tensor,       # [num_tokens, num_heads, head_size]  (原地写入)
    query_start_loc: torch.Tensor,  # [num_seqs + 1] int32
    seq_lens: torch.Tensor,         # [num_seqs] int32
    token_seq_idx: torch.Tensor,    # [num_tokens] int32
    block_table: torch.Tensor,      # [num_seqs, max_num_blocks] int32
    scale: float,
    use_fp8: bool = True,
    k_descale: torch.Tensor | None = None,  # fp8 KV cache 的 per-tensor K descale（1 元素张量）
    v_descale: torch.Tensor | None = None,  # fp8 KV cache 的 per-tensor V descale
    is_prefill: bool | None = None,          # 由 backend 传入（用 max_query_len 预算，避免捕获路径 .item()）
) -> torch.Tensor:
    """FP8(e4m3) 分页 causal GQA attention（prefill+decode 通用），tensor-core tl.dot 实现。

    满足教程接口：签名与语义与默认实现一致，可直接替换接入 vLLM CUSTOM 后端。
    grid=(num_tokens, num_kv_heads)：把每个 kv_head 下的 GROUP 个 q 头当 M 维凑矩阵乘 tile。

    三种 KV 路径：
      - k_descale/v_descale 非 None 且 cache 是 fp8：**KV cache 预量化 e4m3 常驻**——直接读 fp8
        （每字节减半、不重量化），QK fp8 tensor-core，用传入标量 descale。← decode 带宽红利
      - use_fp8=True、bf16 KV：kernel 内动态量化 KV 到 e4m3 做 fp8 QK（旧路径，对照）。
      - use_fp8=False：纯 bf16 tl.dot。
    """
    num_tokens, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]
    max_num_blocks = block_table.shape[1]
    group = num_heads // num_kv_heads

    # KV cache 是否已是 e4m3（预量化常驻）
    kv_is_fp8 = key_cache.dtype in (torch.float8_e4m3fn,) and k_descale is not None
    # descale 以**张量**形式传给 decode kernel（kernel 内 tl.load），避免 host .item() 同步
    # 破坏 CUDA graph 捕获。非 fp8 时给 1.0 的占位张量。
    if kv_is_fp8:
        k_descale_t = k_descale.reshape(1)
        v_descale_t = v_descale.reshape(1)
    else:
        k_descale_t = torch.ones(1, dtype=torch.float32, device=query.device)
        v_descale_t = k_descale_t
    # k/v_descale_val 标量只被 prefill kernel 用（decode kernel 从张量指针 tl.load）。
    # ★ 关键：不在这里 .item()——那是 host-sync，会破坏 decode 的 CUDA graph 捕获
    # （报 "operation not permitted when stream is capturing"）。延迟到 prefill 分支内再算。
    # KV tile：聚合多个物理块，目标 N≈128 让 fp8 tensor core 生效、摊薄 per-block 量化开销
    target_n = 128
    pages_per_tile = max(1, target_n // block_size)
    block_n = pages_per_tile * block_size

    # 按 query_len 分流：prefill/chunked（存在 query_len>1）→ query-tiled prefill kernel；
    # 纯 decode（全部 query_len==1）→ per-token kernel（可被 CUDA graph 捕获）。
    # is_prefill 优先用 backend 传入的（基于 max_query_len 的 host int，不触发 GPU 同步）；
    # 未传入（独立调用/测试）时回退到 .item()（此路径不在 CUDA graph 捕获区内）。
    num_seqs = seq_lens.shape[0]
    if is_prefill is None:
        qlens = query_start_loc[1:] - query_start_loc[:-1]
        is_prefill = int(qlens.max().item()) > 1

    if is_prefill:
        qlens = query_start_loc[1:] - query_start_loc[:-1]
        max_qlen = int(qlens.max().item())   # prefill 走 eager，允许 host 计算
        # prefill kernel 用标量 descale（此处 .item() 安全：prefill 不参与 CUDA graph）。
        k_descale_val = float(k_descale.item()) if kv_is_fp8 else 1.0
        v_descale_val = float(v_descale.item()) if kv_is_fp8 else 1.0
        TILE_Q = _PF_TILE_Q
        num_q_tiles = (max_qlen + TILE_Q - 1) // TILE_Q
        # GQA-pack prefill（默认开）：grid 按 num_kv_heads（非 num_heads），group 个 q 头打进 M，
        # 每个 KV block 只读一次（原每块被读 GROUP 次冗余）。长上下文 prefill 是 KV-读受限，
        # 消掉 8x 冗余读是 100k TTFT<50s 的关键。GROUP*TILE_Q 太大时回退非 pack（避免 smem 溢出）。
        use_gqa_pack = _PF_GQA_PACK and (group >= 2) and (group * _PF_GQA_TILE_Q <= 256)
        if use_gqa_pack:
            gqa_tile_q = _PF_GQA_TILE_Q
            num_q_tiles = (max_qlen + gqa_tile_q - 1) // gqa_tile_q
            grid = (num_seqs, num_q_tiles, num_kv_heads)
            # 滑窗换算成 tile 数（block_n 个 token/tile）；WINDOW=0 → 全 causal。
            window_tiles = ((_PF_WINDOW + block_n - 1) // block_n) if _PF_WINDOW > 0 else 0
            sink_tiles = ((_PF_SINK + block_n - 1) // block_n) if _PF_WINDOW > 0 else 0
            _fp8_prefill_gqa_kernel[grid](
                output, query, key_cache, value_cache,
                query_start_loc, seq_lens, block_table,
                scale,
                num_heads=num_heads, num_kv_heads=num_kv_heads,
                HEAD_SIZE=head_size, BLOCK_SIZE=block_size,
                TILE_Q=gqa_tile_q, GROUP=group, BLOCK_N=block_n, PAGES_PER_TILE=pages_per_tile,
                max_num_blocks=max_num_blocks,
                q_stride_t=query.stride(0), q_stride_h=query.stride(1),
                o_stride_t=output.stride(0), o_stride_h=output.stride(1),
                kc_stride_b=key_cache.stride(0), kc_stride_h=key_cache.stride(1),
                kc_stride_s=key_cache.stride(2),
                vc_stride_b=value_cache.stride(0), vc_stride_h=value_cache.stride(1),
                vc_stride_s=value_cache.stride(2),
                block_table_stride=block_table.stride(0),
                k_descale_val=k_descale_val, v_descale_val=v_descale_val,
                WINDOW_TILES=window_tiles, SINK_TILES=sink_tiles,
                USE_FP8=(use_fp8 and not kv_is_fp8), KV_IS_FP8=kv_is_fp8,
                num_warps=_PF_WARPS, num_stages=_PF_STAGES,
            )
            return output
        grid = (num_seqs, num_q_tiles, num_heads)
        _fp8_prefill_kernel[grid](
            output, query, key_cache, value_cache,
            query_start_loc, seq_lens, block_table,
            scale,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            HEAD_SIZE=head_size, BLOCK_SIZE=block_size,
            TILE_Q=TILE_Q, BLOCK_N=block_n, PAGES_PER_TILE=pages_per_tile,
            max_num_blocks=max_num_blocks,
            q_stride_t=query.stride(0), q_stride_h=query.stride(1),
            o_stride_t=output.stride(0), o_stride_h=output.stride(1),
            kc_stride_b=key_cache.stride(0), kc_stride_h=key_cache.stride(1),
            kc_stride_s=key_cache.stride(2),
            vc_stride_b=value_cache.stride(0), vc_stride_h=value_cache.stride(1),
            vc_stride_s=value_cache.stride(2),
            block_table_stride=block_table.stride(0),
            k_descale_val=k_descale_val, v_descale_val=v_descale_val,
            USE_FP8=(use_fp8 and not kv_is_fp8), KV_IS_FP8=kv_is_fp8,
            num_warps=_PF_WARPS, num_stages=_PF_STAGES,
        )
        return output

    # M pad 到 >=16 的 2 的幂（tensor-core tile）
    block_m = max(16, triton.next_power_of_2(group))

    # ---- 稀疏 decode（CUSTOM_SPARSE=1）：只读 top-k 关键块 ----
    # 纯 decode(is_prefill=False) + bf16 KV(非 fp8 cache) + 序列够长时启用。真实模型注意力
    # 天然稀疏，top-k 选块保住质量；长上下文下读块数降 1/sparsity。gate 在此，其余路径不变。
    if (_SPARSE_ENABLED and not is_prefill and not kv_is_fp8
            and query.dtype in (torch.float16, torch.bfloat16)):
        min_len = _SPARSE_MIN_LEN
        # 用 host int 判断（decode eager 路径，无 CUDA graph）
        if int(seq_lens.max().item()) >= min_len:
            from .sparse_paged import sparse_paged_decode
            return sparse_paged_decode(
                query, key_cache, value_cache, output,
                seq_lens, token_seq_idx, block_table, scale,
                _fp8_decode_combine_kernel,
                sparsity=_SPARSITY, n_sink=_SPARSE_SINK,
                recent_win=_SPARSE_RECENT)

    # ---- split-KV / flash-decoding ----
    # bs=1 decode 时原 grid=(num_tokens, kv_heads) 只有 ~8 CTA，GPU 严重闲置、每 CTA 串行扫
    # 整条 KV（实测 100k 仅 0.16TB/s、6 tok/s 上限）。把每个 (token,kv_head) 的 KV 扫描切成
    # NUM_SPLITS 段并行，grid 变 (num_tokens, kv_heads, NUM_SPLITS)，填满 SM、打满带宽，
    # 再由 combine kernel 跨段做在线-softmax 合并。NUM_SPLITS 取“填满 SM 数”与“KV tile 数”的
    # 较小值（is_prefill=False 且 CUDA graph 捕获时用固定值以保证形状静态）。
    try:
        num_sm = torch.cuda.get_device_properties(query.device).multi_processor_count
    except Exception:  # noqa: BLE001
        num_sm = 78
    # 每个 (token,kv_head) 分到的 split 数：让总 CTA≈几倍 SM。用 num_tokens*kv_heads 决定，
    # 不依赖 seq_lens.max()（避免 host 同步，兼容 CUDA graph）；kernel 内按实际 num_tiles 再封顶，
    # 空 split（t_start>=t_end）会写入 -inf/0，combine 时被 softmax 自然忽略。
    target_ctas = 8 * num_sm
    denom = max(1, num_tokens * num_kv_heads)
    num_splits = max(1, min(target_ctas // denom, 128))
    # NUM_SPLITS 需是 2 的幂（combine kernel 里 tl.arange(0,NUM_SPLITS) 要求）
    num_splits = 1 << (num_splits.bit_length() - 1)
    # 基准对比覆盖：CUSTOM_DEC_SPLITS>0 时强制该值（须为 2 的幂）。设 1 即模拟无 split-KV。
    if _DEC_SPLITS > 0:
        num_splits = 1 << (_DEC_SPLITS.bit_length() - 1)
    op = torch.empty((num_tokens, num_kv_heads, num_splits, block_m, head_size),
                     dtype=torch.float32, device=query.device)
    ml = torch.empty((num_tokens, num_kv_heads, num_splits, 2, block_m),
                     dtype=torch.float32, device=query.device)

    grid = (num_tokens, num_kv_heads, num_splits)
    _fp8_paged_attn_kernel[grid](
        output, query, key_cache, value_cache,
        query_start_loc, seq_lens, token_seq_idx, block_table,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        HEAD_SIZE=head_size,
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        PAGES_PER_TILE=pages_per_tile,
        BLOCK_M=block_m,
        GROUP=group,
        max_num_blocks=max_num_blocks,
        q_stride_t=query.stride(0), q_stride_h=query.stride(1),
        o_stride_t=output.stride(0), o_stride_h=output.stride(1),
        kc_stride_b=key_cache.stride(0), kc_stride_h=key_cache.stride(1),
        kc_stride_s=key_cache.stride(2),
        vc_stride_b=value_cache.stride(0), vc_stride_h=value_cache.stride(1),
        vc_stride_s=value_cache.stride(2),
        block_table_stride=block_table.stride(0),
        NUM_SPLITS=num_splits,
        op_ptr=op, ml_ptr=ml,
        op_st=op.stride(0), op_sh=op.stride(1), op_ss=op.stride(2), op_sm=op.stride(3),
        ml_st=ml.stride(0), ml_sh=ml.stride(1), ml_ss=ml.stride(2), ml_se=ml.stride(3),
        k_descale_ptr=k_descale_t,
        v_descale_ptr=v_descale_t,
        USE_FP8=(use_fp8 and not kv_is_fp8),
        KV_IS_FP8=kv_is_fp8,
        num_warps=_DEC_WARPS,
        num_stages=_DEC_STAGES,
    )
    _fp8_decode_combine_kernel[(num_tokens, num_kv_heads)](
        output, op, ml,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        HEAD_SIZE=head_size, BLOCK_M=block_m, GROUP=group, NUM_SPLITS=num_splits,
        o_stride_t=output.stride(0), o_stride_h=output.stride(1),
        op_st=op.stride(0), op_sh=op.stride(1), op_ss=op.stride(2), op_sm=op.stride(3),
        ml_st=ml.stride(0), ml_sh=ml.stride(1), ml_ss=ml.stride(2), ml_se=ml.stride(3),
    )
    return output
