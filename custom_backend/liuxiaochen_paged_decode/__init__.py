# SPDX-License-Identifier: Apache-2.0
"""V2 independent decode-only paged-KV warp-MMA prototype (Liu Xiaochen).

NOT registered as the active CUSTOM backend; standalone correctness/perf prototype.
"""

from .runner_v2 import paged_decode_v2, workspace_bytes

__all__ = ["paged_decode_v2", "workspace_bytes"]
