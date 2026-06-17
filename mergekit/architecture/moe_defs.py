# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import ClassVar, List, Optional

from pydantic import BaseModel
from transformers import PretrainedConfig

from mergekit.architecture.base import (
    ModuleArchitecture,
    WeightInfo,
)
from mergekit.architecture.json_definitions import NAME_TO_ARCH

AFMOE_PARTIAL_INFO = NAME_TO_ARCH["_AfmoePartialForCausalLM"][0]
AFMOE_PARTIAL_MODULE_ARCH = AFMOE_PARTIAL_INFO.modules["default"].architecture


class AfmoeModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "AfmoeForCausalLM"
    num_experts: int

    def name(self) -> str:
        return "afmoe"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return AfmoeModuleArchitecture(num_experts=config.num_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return AFMOE_PARTIAL_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return AFMOE_PARTIAL_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        return AFMOE_PARTIAL_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        res = AFMOE_PARTIAL_MODULE_ARCH.layer_weights(index, config) or []
        prefix = f"model.layers.{index}"
        for expert_idx in range(self.num_experts):
            for param in ("up_proj", "gate_proj", "down_proj"):
                res.append(
                    WeightInfo(
                        name=prefix + f".mlp.experts.{expert_idx}.{param}.weight",
                        optional=True,
                    )
                )
        return res


GLM4_INFO = NAME_TO_ARCH["Glm4MoeForCausalLM"][0]
GLM4_MODULE_ARCH = GLM4_INFO.modules["default"].architecture


class Glm4MoeModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "Glm4MoeForCausalLM"
    num_experts: int

    def name(self) -> str:
        return "glm4_moe"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return Glm4MoeModuleArchitecture(num_experts=config.n_routed_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return GLM4_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return GLM4_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        return GLM4_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        prefix = f"model.layers.{index}"

        if index < config.first_k_dense_replace:
            return GLM4_MODULE_ARCH.layer_weights(index, config)
        else:
            tensor_names = [
                prefix + ".mlp.experts.gate_up_proj",
                prefix + ".mlp.experts.down_proj",
                prefix + ".mlp.gate.weight",
                prefix + ".mlp.gate.e_score_correction_bias",
            ]
            shared_expert_names = [
                (prefix + ".mlp.shared_experts.gate_proj.weight", False),
                (prefix + ".mlp.shared_experts.up_proj.weight", False),
                (prefix + ".mlp.shared_experts.down_proj.weight", False),
            ]
            res = []
            for name in tensor_names:
                res.append(WeightInfo(name=name))
            for name, optional in shared_expert_names:
                res.append(WeightInfo(name=name, optional=optional))
            for weight_info in GLM4_MODULE_ARCH.layer_weights(index, config):
                if ".mlp." in weight_info.name:
                    continue
                res.append(weight_info)
            return res
