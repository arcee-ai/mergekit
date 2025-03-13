# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import ClassVar, List, Optional

from pydantic import BaseModel
from transformers import PretrainedConfig

from mergekit.architecture.base import (
    ModuleArchitecture,
    WeightInfo,
)
from mergekit.architecture.json_definitions import NAME_TO_ARCH

MISTRAL_INFO = NAME_TO_ARCH["MistralForCausalLM"][0]
MISTRAL_MODULE_ARCH = MISTRAL_INFO.modules["default"].architecture


class MixtralTensorNames(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "MixtralForCausalLM"
    num_local_experts: int

    def name(self) -> str:
        return "mixtral"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return MixtralTensorNames(num_local_experts=config.num_local_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return MISTRAL_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return MISTRAL_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        return MISTRAL_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        num_experts = self.num_local_experts
        prefix = f"model.layers.{index}"
        tensor_names = []
        for expert_idx in range(num_experts):
            for param in ("w1", "w2", "w3"):
                tensor_names.append(
                    prefix + f".block_sparse_moe.experts.{expert_idx}.{param}.weight"
                )
        tensor_names.append(prefix + ".block_sparse_moe.gate.weight")
        res = []
        for name in tensor_names:
            res.append(WeightInfo(name=name))
        for weight_info in MISTRAL_MODULE_ARCH.layer_weights(index, config):
            if ".mlp." in weight_info.name:
                continue
            res.append(weight_info)
        return res
