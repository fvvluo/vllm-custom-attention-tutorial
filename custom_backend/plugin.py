# SPDX-License-Identifier: Apache-2.0
"""
vLLM general plugin 入口：把 CustomTritonBackend 注册到 CUSTOM 后端。

vLLM 会在【所有进程】（前端、EngineCore、worker）启动时加载
`vllm.general_plugins` 组下的插件并调用其入口函数。我们在这里把
AttentionBackendEnum.CUSTOM 指向本教程的自定义后端类，之后就能用
`--attention-backend CUSTOM` 选择它。

entry point 在 pyproject.toml 里声明：
    [project.entry-points."vllm.general_plugins"]
    custom_triton = "custom_backend.plugin:register"
"""


def register() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "custom_backend.custom_triton_backend.CustomTritonBackend",
    )
