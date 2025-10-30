from typing import List

from mergekit.moe.arch import MoEOutputArchitecture
from mergekit.moe.deepseek import DeepseekMoE
from mergekit.moe.mixtral import MixtralMoE

ALL_OUTPUT_ARCHITECTURES: List[MoEOutputArchitecture] = [MixtralMoE(), DeepseekMoE()]

try:
    from mergekit.moe.qwen import QwenMoE
except ImportError:
    pass
else:
    ALL_OUTPUT_ARCHITECTURES.append(QwenMoE())

try:
    from mergekit.moe.qwen3 import Qwen3MoE
except ImportError:
    pass
else:
    ALL_OUTPUT_ARCHITECTURES.append(Qwen3MoE())

__all__ = [
    "ALL_OUTPUT_ARCHITECTURES",
    "MoEOutputArchitecture",
]
