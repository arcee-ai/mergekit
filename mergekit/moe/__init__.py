from typing import List

from mergekit.moe.arch import MoEOutputArchitecture
from mergekit.moe.deepseek import DeepseekMoE
from mergekit.moe.mixtral import MixtralMoE

ALL_OUTPUT_ARCHITECTURES: List[MoEOutputArchitecture] = [MixtralMoE(), DeepseekMoE()]

__all__ = [
    "ALL_OUTPUT_ARCHITECTURES",
    "MoEOutputArchitecture",
    "MixtralMoE",
    "DeepseekMoE",
]
