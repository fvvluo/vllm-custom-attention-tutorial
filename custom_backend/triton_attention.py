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

_E4M3_MAX = tl.constexpr(448.0)
_FP8_DTYPE = tl.float8e4nv


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
    USE_FP8: tl.constexpr,       # QK^T 是否走 fp8 tensor-core（bf16 KV 时的动态量化路径）
    KV_IS_FP8: tl.constexpr,     # KV cache 是否已是 e4m3（预量化常驻）
):
    # grid = (num_tokens, num_kv_heads)：每个 (query token, kv_head) 一个 program，
    # 把该 kv_head 下 GROUP 个 q 头一起算（M 维），凑出矩阵乘 tile 给 tensor core。
    token_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

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
    offs_p = tl.arange(0, PAGES_PER_TILE)        # [PAGES]
    offs_s = tl.arange(0, BLOCK_SIZE)            # [BLOCK_SIZE] 页内槽位（连续）
    for t in range(0, num_tiles):
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

    out = acc / l_i[:, None]
    o_ptrs = (out_ptr + token_idx * o_stride_t
              + head_ids[:, None] * o_stride_h + offs_d[None, :])
    tl.store(o_ptrs, out.to(out_ptr.dtype.element_ty), mask=m_valid[:, None])


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
    k_descale_val = float(k_descale.item()) if kv_is_fp8 else 1.0  # prefill 走 eager，标量即可
    v_descale_val = float(v_descale.item()) if kv_is_fp8 else 1.0
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
        TILE_Q = 64
        num_q_tiles = (max_qlen + TILE_Q - 1) // TILE_Q
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
            num_warps=4, num_stages=2,
        )
        return output

    # M pad 到 >=16 的 2 的幂（tensor-core tile）
    block_m = max(16, triton.next_power_of_2(group))
    grid = (num_tokens, num_kv_heads)
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
        k_descale_ptr=k_descale_t,
        v_descale_ptr=v_descale_t,
        USE_FP8=(use_fp8 and not kv_is_fp8),
        KV_IS_FP8=kv_is_fp8,
    )
    return output
