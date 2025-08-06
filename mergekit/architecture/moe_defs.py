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

LLAMA_INFO = NAME_TO_ARCH["LlamaForCausalLM"][0]
LLAMA_MODULE_ARCH = LLAMA_INFO.modules["default"].architecture


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


class Ernie4_5_MoeModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "Ernie4_5_MoeForCausalLM"
    moe_num_experts: int
    moe_num_shared_experts: int = 2
    moe_layer_start_index: int = 1

    def name(self) -> str:
        return "ernie4_5_moe"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return Ernie4_5_MoeModuleArchitecture(
            moe_num_experts=getattr(config, "moe_num_experts", 64),
            moe_num_shared_experts=getattr(config, "moe_num_shared_experts", 2),
            moe_layer_start_index=getattr(config, "moe_layer_start_index", 1),
        )

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return LLAMA_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        base_weights = LLAMA_MODULE_ARCH.post_weights(config)
        
        # Add MTP weights
        mtp_weights = [
            WeightInfo(name="model.mtp_block.0.input_layernorm.weight", optional=True),
            WeightInfo(name="model.mtp_block.0.self_attn.q_proj.weight", optional=True),
            WeightInfo(name="model.mtp_block.0.self_attn.k_proj.weight", optional=True),
            WeightInfo(name="model.mtp_block.0.self_attn.v_proj.weight", optional=True),
            WeightInfo(name="model.mtp_block.0.self_attn.o_proj.weight", optional=True),
            WeightInfo(name="model.mtp_block.0.post_attention_layernorm.weight", optional=True),
            WeightInfo(name="model.mtp_block.0.mlp.gate_proj.weight", optional=True),
            WeightInfo(name="model.mtp_block.0.mlp.up_proj.weight", optional=True),
            WeightInfo(name="model.mtp_block.0.mlp.down_proj.weight", optional=True),
            WeightInfo(name="model.mtp_emb_norm.0.weight", optional=True),
            WeightInfo(name="model.mtp_hidden_norm.0.weight", optional=True),
            WeightInfo(name="model.mtp_linear_proj.0.weight", optional=True),
        ]
        
        return base_weights + mtp_weights

    def num_layers_config_key(self) -> str:
        return LLAMA_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        prefix = f"model.layers.{index}"
        tensor_names = []
        
        # Check if this is a MoE layer
        is_moe_layer = index >= self.moe_layer_start_index
        
        if is_moe_layer:
            # Add MoE expert weights
            for expert_idx in range(self.moe_num_experts):
                for param in ("gate_proj", "up_proj", "down_proj"):
                    tensor_names.append(
                        prefix + f".mlp.experts.{expert_idx}.{param}.weight"
                    )
            
            # Add shared expert weights
            for param in ("gate_proj", "up_proj", "down_proj"):
                tensor_names.append(
                    prefix + f".mlp.shared_experts.{param}.weight"
                )
            
            # Add MoE gate and statistics
            tensor_names.append(prefix + ".mlp.gate.weight")
            tensor_names.append(prefix + ".mlp.moe_statics.e_score_correction_bias")
        else:
            # Dense layer (layer 0) - add regular MLP weights
            for param in ("gate_proj", "up_proj", "down_proj"):
                tensor_names.append(prefix + f".mlp.{param}.weight")
        
        res = []
        for name in tensor_names:
            res.append(WeightInfo(name=name))
        
        # Add attention and norm weights from base architecture
        for weight_info in LLAMA_MODULE_ARCH.layer_weights(index, config):
            if ".mlp." in weight_info.name:
                continue  # Skip MLP weights, we handle them above
            res.append(weight_info)
        
        return res


GRANITE_INFO = NAME_TO_ARCH["GraniteForCausalLM"][0]
GRANITE_MODULE_ARCH = GRANITE_INFO.modules["default"].architecture


class GptOssModuleArchitecture(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "GptOssForCausalLM"
    num_local_experts: int

    def name(self) -> str:
        return "gpt_oss"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return GptOssModuleArchitecture(num_local_experts=config.num_local_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return LLAMA_MODULE_ARCH.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return LLAMA_MODULE_ARCH.post_weights(config)

    def num_layers_config_key(self) -> str:
        return LLAMA_MODULE_ARCH.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        num_experts = self.num_local_experts
        prefix = f"model.layers.{index}"
        tensor_names = []
        
        # Add MoE expert weights (GPT OSS uses gate_up_proj/down_proj pattern)
        tensor_names.extend([
            prefix + ".mlp.experts.gate_up_proj",
            prefix + ".mlp.experts.gate_up_proj_bias",
            prefix + ".mlp.experts.down_proj", 
            prefix + ".mlp.experts.down_proj_bias"
        ])
        
        # Add router weights
        tensor_names.extend([
            prefix + ".mlp.router.weight",
            prefix + ".mlp.router.bias"
        ])
        
        res = []
        for name in tensor_names:
            res.append(WeightInfo(name=name))
        
        # Add attention and norm weights from base architecture
        for weight_info in LLAMA_MODULE_ARCH.layer_weights(index, config):
            if ".mlp." in weight_info.name:
                continue  # Skip MLP weights, we handle them above
            # Add GPT OSS specific attention sinks
            if weight_info.name.endswith(".self_attn.o_proj.weight"):
                res.append(weight_info)
                res.append(WeightInfo(name=weight_info.name.replace("o_proj.weight", "sinks")))
            else:
                res.append(weight_info)
        
        return res