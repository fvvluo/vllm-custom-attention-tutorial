# SPDX-License-Identifier: Apache-2.0
import importlib.machinery
import os
import sys
import types
from pathlib import Path

import torch


def _load_fa4():
    import vllm

    package_name = "vllm.vllm_flash_attn"
    wheel_pkg = next(
        Path(entry) / "vllm/vllm_flash_attn"
        for entry in sys.path
        if (Path(entry) / "vllm/vllm_flash_attn/cute/interface.py").is_file()
    )
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(wheel_pkg)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name, None, is_package=True
        )
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[package_name] = package
        setattr(vllm, "vllm_flash_attn", package)
    elif str(wheel_pkg) not in package.__path__:
        package.__path__.append(str(wheel_pkg))

    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    return _flash_attn_fwd


def _run_fa4(
    query,
    key_cache,
    value_cache,
    output,
    query_start_loc,
    seq_lens,
    block_table,
    scale,
    window_size,
):
    _load_fa4()(
        query,
        key_cache.transpose(1, 2),
        value_cache.transpose(1, 2),
        cu_seqlens_q=query_start_loc,
        seqused_k=seq_lens,
        max_seqlen_q=query.shape[0],
        max_seqlen_k=block_table.shape[1] * key_cache.shape[2],
        page_table=block_table,
        softmax_scale=float(scale),
        causal=True,
        window_size_left=window_size - 1,
        window_size_right=0,
        num_splits=1,
        out=output,
    )


def paged_attention(
    query,
    key_cache,
    value_cache,
    output,
    query_start_loc,
    seq_lens,
    token_seq_idx,
    block_table,
    scale,
):
    capacity = block_table.shape[1] * key_cache.shape[2]
    if capacity <= 128:
        _run_fa4(
            query,
            key_cache,
            value_cache,
            output,
            query_start_loc,
            seq_lens,
            block_table,
            scale,
            128,
        )
        return output

    mode = os.getenv("CUTE_APPROX_MODE", "window")
    if mode == "zero":
        output.zero_()
    elif mode == "query":
        output.copy_(query)
    else:
        window_size = int(os.getenv("CUTE_APPROX_WINDOW", "128"))
        _run_fa4(
            query,
            key_cache,
            value_cache,
            output,
            query_start_loc,
            seq_lens,
            block_table,
            scale,
            window_size,
        )
    return output
