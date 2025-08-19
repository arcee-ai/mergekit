from typing import List

from llmtailor.moe.arch import MoEOutputArchitecture
from llmtailor.moe.deepseek import DeepseekMoE
from llmtailor.moe.mixtral import MixtralMoE

ALL_OUTPUT_ARCHITECTURES: List[MoEOutputArchitecture] = [MixtralMoE(), DeepseekMoE()]

try:
    from mergekit.moe.qwen import QwenMoE
except ImportError:
    pass
else:
    ALL_OUTPUT_ARCHITECTURES.append(QwenMoE())

__all__ = [
    "ALL_OUTPUT_ARCHITECTURES",
    "MoEOutputArchitecture",
]
