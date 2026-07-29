# SPDX-License-Identifier: Apache-2.0
"""
自定义 vLLM v1 Attention Backend（教学示例）
=================================================

本文件把 `triton_attention.py` 里的简易 Triton attention 接入 vLLM v1 引擎。
它实现了 v1 attention backend 的“三件套”：

  1. CustomTritonBackend        —— 描述后端能力、KV cache 形状、关联 Impl/Builder
  2. CustomTritonMetadataBuilder —— 每步前向把通用元数据转成本后端所需的元数据
  3. CustomTritonImpl            —— forward()：写 KV cache + 调用 Triton attention

关键设计（让教程尽量简单）：
  - `forward_includes_kv_cache_update = True`：把“写 KV cache”和“算 attention”都放在
    forward 里，学生只看一个方法就懂完整流程。
  - `_cudagraph_support = NEVER`（默认）：不参与 CUDA graph 捕获；启动服务时用
    `--enforce-eager` 关闭 torch.compile/CUDA graph，路径最简单、最好调试。

学生如何替换成自己的 kernel：
  只需修改 `triton_attention.py` 里 `paged_attention_triton(...)` 的实现（保持签名与
  语义不变），或在 `CustomTritonImpl.forward` 里改成调用你自己的函数即可。
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.platforms import current_platform

from .triton_attention import paged_attention_triton


# ============================================================================
# 1. 本后端使用的 attention 元数据
# ============================================================================
@dataclass
class CustomTritonMetadata:
    num_actual_tokens: int
    query_start_loc: torch.Tensor   # [num_seqs + 1]
    seq_lens: torch.Tensor          # [num_seqs]
    block_table: torch.Tensor       # [num_seqs, max_num_blocks]
    slot_mapping: torch.Tensor      # [num_actual_tokens]
    token_seq_idx: torch.Tensor     # [num_actual_tokens] 每 token 属于哪条请求
    causal: bool | torch.Tensor
    # 由 build() 用 common_attn_metadata.max_query_len（host int）预算，避免 forward 里 .item()
    # 触发 GPU 同步而破坏 CUDA graph 捕获。True=prefill/chunked，False=纯 decode(可被 graph 捕获)。
    is_prefill: bool = False


# ============================================================================
# 2. 元数据 Builder：把 vLLM 的通用元数据转成本后端所需的元数据
# ============================================================================
class CustomTritonMetadataBuilder(AttentionMetadataBuilder[CustomTritonMetadata]):
    # 支持对**纯 decode（query_len==1）批**做 CUDA graph 捕获——decode 的瓶颈是 64 层逐步
    # eager launch 开销，开 graph 后能大幅提速；prefill 仍走 eager（不捕获）。
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.device = device
        # decode 批分组：query_len<=1 的请求会被排到 batch 前部，便于纯 decode 批被 graph 捕获。
        self._init_reorder_batch_threshold(1)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CustomTritonMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        num_actual_tokens = common_attn_metadata.num_actual_tokens

        # 预计算 token -> 请求 的映射，供 Triton kernel 直接读取。
        token_ids = torch.arange(
            num_actual_tokens, device=query_start_loc.device, dtype=torch.int32
        )
        token_seq_idx = (
            torch.searchsorted(query_start_loc[1:], token_ids, right=True)
        ).to(torch.int32)

        # 用 common_attn_metadata.max_query_len（host int，无 GPU 同步）预判 prefill/decode，
        # forward 直接读该 bool，避免捕获路径里 .item() 同步。
        is_prefill = common_attn_metadata.max_query_len > 1

        return CustomTritonMetadata(
            num_actual_tokens=num_actual_tokens,
            query_start_loc=query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            token_seq_idx=token_seq_idx,
            causal=common_attn_metadata.causal,
            is_prefill=is_prefill,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> CustomTritonMetadata:
        # 捕获期只针对纯 decode 批（max_query_len==1）。参考 triton_attn：把 seq_lens 填 1
        # 避免按 max_model_len 捕获时极慢。
        m = self.build(0, common_attn_metadata)
        m.seq_lens.fill_(1)
        m.is_prefill = False
        return m


# ============================================================================
# 3. Backend：描述后端能力，关联 Impl 与 Builder
# ============================================================================
class CustomTritonBackend(AttentionBackend):
    # 本示例只支持 fp16/bf16（与 Qwen3-32B 的 bf16 匹配）。
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    # auto = 与模型同 dtype（bf16）；fp8/fp8_e4m3 = KV cache 预量化为 e4m3 常驻，
    # 每字节减半（decode 带宽红利），且 K/V 只在写入时量化一次、读时直接用 fp8。
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
    ]

    # 关键：forward 内部自己负责写 KV cache（写 + 读一体）。
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["CustomTritonImpl"]:
        return CustomTritonImpl

    @staticmethod
    def get_builder_cls() -> type[CustomTritonMetadataBuilder]:
        return CustomTritonMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Qwen3-32B head_dim=128；示例 kernel 对 head_size 是 2 的幂时最稳。
        return [32, 64, 128, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # 与 vLLM TRITON_ATTN 一致：K、V 打包进最后一维。
        # 逻辑形状 (num_blocks, num_kv_heads, block_size, 2 * head_size)
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)


# ============================================================================
# 4. Impl：真正干活的地方
# ============================================================================
class CustomTritonImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        # 本教学示例只支持最常见的 decoder self-attention + causal。
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "CustomTritonImpl 教学示例仅支持 AttentionType.DECODER"
            )
        if alibi_slopes is not None:
            raise NotImplementedError("CustomTritonImpl 教学示例不支持 alibi")
        if sliding_window is not None:
            raise NotImplementedError("CustomTritonImpl 教学示例不支持 sliding window")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,   # [num_tokens, num_heads, head_size]
        key: torch.Tensor,     # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,   # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, 2*head_size]
        attn_metadata: CustomTritonMetadata,
        output: torch.Tensor,  # [num_tokens, num_heads * head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # profiling / 预热阶段 attn_metadata 为 None，直接返回全 0。
        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        hs = self.head_size

        # ---- KV cache 布局说明（重要！）----
        # 本后端没有实现 get_kv_cache_stride_order，所以 vLLM 分配的 KV cache
        # 物理内存布局 = 逻辑形状 = (num_blocks, num_kv_heads, block_size, 2*hs)。
        # 我们把最后一维拆成 K、V 两半，得到形状
        #   (num_blocks, num_kv_heads, block_size, hs)
        # 的 key_cache / value_cache 视图。下面的读（Triton attention）与
        # 写（reshape_and_cache）都完全按“张量步长(stride)”寻址，因此只要读写
        # 用的是同一组视图，就一定能对上物理位置——这也是学生自定义 kernel 时
        # 最省心的做法：不用关心底层是 NHD 还是 HND，只按 stride 索引即可。
        key_cache, value_cache = kv_cache.split(hs, dim=-1)

        # ---- FP8 KV cache（预量化 e4m3 常驻）----
        # --kv-cache-dtype fp8 时：vLLM 把 cache 分配为 fp8（每字节减半），
        # reshape_and_cache 在写入时按 layer._k_scale/_v_scale 把 K/V 量化成 e4m3
        # 存进去（**只量化一次**）；kernel 读时直接用 fp8、再乘回 descale 恢复量级。
        # 这样 decode 每步从 HBM 搬的 KV 字节减半、且不必每步重量化——fp8 在 decode
        # 真正省带宽的地方。
        is_fp8_kv = self.kv_cache_dtype.startswith("fp8")
        if is_fp8_kv:
            fp8_dtype = current_platform.fp8_dtype()
            if key_cache.dtype != fp8_dtype:
                key_cache = key_cache.view(fp8_dtype)
                value_cache = value_cache.view(fp8_dtype)

        # ---- 1. 把本次新的 K/V 写入分页 KV cache ----
        # 复用 vLLM 现成的 triton_reshape_and_cache_flash 写入。该 op 在
        # kv_cache_dtype 为 fp8 时会用 k_scale/v_scale 把 K/V 量化成 e4m3 存入。
        # 注意：它约定 cache 形状是 [num_blocks, block_size, num_heads, head_size]
        # （slot 维在前），我们的视图是 head 维在前，故把 head/slot transpose 一下；
        # 完全按 stride 寻址，物理写入位置保持正确。
        triton_reshape_and_cache_flash(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            key_cache.transpose(1, 2),
            value_cache.transpose(1, 2),
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        # ---- 2. 调用（学生可替换的）Triton 分页注意力 ----
        # fp8 KV cache 时把 per-tensor descale（k_scale/v_scale 标量）传给 kernel，
        # 让它把读到的 fp8 K/V 乘回真实量级；auto(bf16) 时 descale=None。
        out_view = output[:num_actual_tokens].view(-1, self.num_heads, hs)
        paged_attention_triton(
            query=query[:num_actual_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            output=out_view,
            query_start_loc=attn_metadata.query_start_loc,
            seq_lens=attn_metadata.seq_lens,
            token_seq_idx=attn_metadata.token_seq_idx,
            block_table=attn_metadata.block_table,
            scale=self.scale,
            k_descale=(layer._k_scale if is_fp8_kv else None),
            v_descale=(layer._v_scale if is_fp8_kv else None),
            # 由 builder 用 max_query_len 预算，避免捕获路径里的 .item() 同步。
            is_prefill=attn_metadata.is_prefill,
        )
        return output
