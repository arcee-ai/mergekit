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

MISTRAL_INFO = NAME_TO_ARCH["MistralForCausalLM"][0]
MISTRAL_MODULE_ARCH = MISTRAL_INFO.modules["default"].architecture


class MixtralModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "MixtralForCausalLM"
    num_local_experts: int

    def name(self) -> str:
        return "mixtral"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return MixtralModuleArchitecture(num_local_experts=config.num_local_experts)

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


QWEN3_INFO = NAME_TO_ARCH["Qwen3ForCausalLM"][0]
QWEN3_MODULE_ARCH = QWEN3_INFO.modules["default"].architecture


class Qwen3MoeModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "Qwen3MoeForCausalLM"
    num_experts: int

    def name(self) -> str:
        return "qwen3_moe"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return Qwen3MoeModuleArchitecture(num_experts=config.num_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return QWEN3_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return QWEN3_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        return QWEN3_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        prefix = f"model.layers.{index}"
        tensor_names = []
        for expert_idx in range(self.num_experts):
            for param in ("up_proj", "gate_proj", "down_proj"):
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.{param}.weight"
                )
        tensor_names.append(prefix + ".mlp.gate.weight")
        res = []
        for name in tensor_names:
            res.append(WeightInfo(name=name))
        for weight_info in QWEN3_MODULE_ARCH.layer_weights(index, config):
            if ".mlp." in weight_info.name:
                continue
            res.append(weight_info)
        return res


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

print(f"GLM4_INFO: {GLM4_INFO}")
print(f"GLM4_MODULE_ARCH: {GLM4_MODULE_ARCH}")

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
        tensor_names = []
        if index < config.first_k_dense_replace:
            res = []
            for weight_info in GLM4_MODULE_ARCH.layer_weights(index, config):
                res.append(weight_info)
            # print(f"index: {index} and res: {res}")
            return res
        else:
            for expert_idx in range(self.num_experts):
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.gate_proj.weight"
                )
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.up_proj.weight"
                )
                tensor_names.append(
                    prefix + f".mlp.experts.{expert_idx}.down_proj.weight"
                )
            tensor_names.append(prefix + ".mlp.gate.weight")
            # Add shared expert weights (optional - will be present if using shared expert)
            # Mark as optional so they can be missing if no shared expert is used
            shared_expert_names = [
                (prefix + ".mlp.shared_expert.gate_proj.weight", True),
                (prefix + ".mlp.shared_expert.up_proj.weight", True),
                (prefix + ".mlp.shared_expert.down_proj.weight", True),
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
