# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import List, Optional

from pydantic import BaseModel
from transformers import PretrainedConfig

from mergekit.architecture.base import (
    ModelArchitecture,
    ModuleArchitecture,
    ModuleDefinition,
    WeightInfo,
)
from mergekit.common import get_config_value

QWEN35_DENSE_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForCausalLM",
}
QWEN35_MOE_ARCHITECTURES = {
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
}
QWEN35_ARCHITECTURES = QWEN35_DENSE_ARCHITECTURES | QWEN35_MOE_ARCHITECTURES


def _text_config(config: PretrainedConfig):
    return getattr(config, "text_config", config)


def _cfg(config: PretrainedConfig, key: str, default=None):
    try:
        return get_config_value(config, key)
    except Exception:
        return default


def _is_full_attention(config: PretrainedConfig, index: int) -> bool:
    layer_types = getattr(_text_config(config), "layer_types", None)
    if layer_types and index < len(layer_types):
        return layer_types[index] == "full_attention"
    # Qwen3.5 defaults to three linear-attention layers followed by one full-attention layer.
    return index % 4 == 3


class Qwen35LanguageModuleArchitecture(ModuleArchitecture, BaseModel, frozen=True):
    """Text decoder for Qwen3.5 dense and MoE checkpoints.

    Official Qwen3.5 repos are image-text-to-text wrappers whose language weights live
    under ``model.language_model``. Text-only exports use the usual ``model`` prefix.
    """

    root: str
    is_moe: bool = False
    num_experts: Optional[int] = None

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return [WeightInfo(name=f"{self.root}.embed_tokens.weight", is_embed=True)]

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return [
            WeightInfo(name=f"{self.root}.norm.weight"),
            WeightInfo(
                name="lm_head.weight",
                is_embed=True,
                optional=True,
                tied_names=(f"{self.root}.embed_tokens.weight",),
            ),
        ]

    def num_layers_config_key(self) -> str:
        return (
            "text_config.num_hidden_layers"
            if self.root == "model.language_model"
            else "num_hidden_layers"
        )

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        prefix = f"{self.root}.layers.{index}"
        res = [WeightInfo(name=f"{prefix}.input_layernorm.weight")]

        if _is_full_attention(config, index):
            res.extend(
                WeightInfo(name=f"{prefix}.self_attn.{name}")
                for name in (
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "q_norm.weight",
                    "k_norm.weight",
                )
            )
            if getattr(_text_config(config), "attention_bias", False):
                res.extend(
                    WeightInfo(name=f"{prefix}.self_attn.{name}", optional=True)
                    for name in (
                        "q_proj.bias",
                        "k_proj.bias",
                        "v_proj.bias",
                        "o_proj.bias",
                    )
                )
        else:
            res.extend(
                WeightInfo(name=f"{prefix}.linear_attn.{name}")
                for name in (
                    "dt_bias",
                    "A_log",
                    "conv1d.weight",
                    "norm.weight",
                    "out_proj.weight",
                    "in_proj_qkv.weight",
                    "in_proj_z.weight",
                    "in_proj_b.weight",
                    "in_proj_a.weight",
                )
            )

        if self.is_moe:
            res.append(WeightInfo(name=f"{prefix}.mlp.gate.weight"))
            res.extend(
                WeightInfo(name=f"{prefix}.mlp.{name}", optional=True)
                for name in (
                    "experts.gate_up_proj",
                    "experts.down_proj",
                )
            )
            for expert_idx in range(self.num_experts or 0):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    res.append(
                        WeightInfo(
                            name=f"{prefix}.mlp.experts.{expert_idx}.{proj}.weight",
                            optional=True,
                        )
                    )
            if getattr(_text_config(config), "shared_expert_intermediate_size", None):
                res.extend(
                    WeightInfo(name=f"{prefix}.mlp.{name}")
                    for name in (
                        "shared_expert.gate_proj.weight",
                        "shared_expert.up_proj.weight",
                        "shared_expert.down_proj.weight",
                        "shared_expert_gate.weight",
                    )
                )
        else:
            res.extend(
                WeightInfo(name=f"{prefix}.mlp.{name}")
                for name in (
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                )
            )

        res.append(WeightInfo(name=f"{prefix}.post_attention_layernorm.weight"))
        return res


class Qwen35MtpModuleArchitecture(ModuleArchitecture, BaseModel, frozen=True):
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_layers_key: str = "text_config.mtp_num_hidden_layers"

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return [
            WeightInfo(name="mtp.fc.weight", optional=True),
            WeightInfo(name="mtp.norm.weight", optional=True),
            WeightInfo(name="mtp.pre_fc_norm_embedding.weight", optional=True),
            WeightInfo(name="mtp.pre_fc_norm_hidden.weight", optional=True),
        ]

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return []

    def num_layers_config_key(self) -> Optional[str]:
        return self.num_layers_key

    def num_layers(self, config: PretrainedConfig) -> int:
        return int(
            _cfg(
                config,
                self.num_layers_key,
                _cfg(
                    config,
                    "text_config.mtp_num_hidden_layers",
                    _cfg(config, "mtp_num_hidden_layers", 0),
                ),
            )
            or 0
        )

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        prefix = f"mtp.layers.{index}"
        res = [
            WeightInfo(name=f"{prefix}.input_layernorm.weight", optional=True),
            WeightInfo(name=f"{prefix}.self_attn.q_proj.weight", optional=True),
            WeightInfo(name=f"{prefix}.self_attn.k_proj.weight", optional=True),
            WeightInfo(name=f"{prefix}.self_attn.v_proj.weight", optional=True),
            WeightInfo(name=f"{prefix}.self_attn.o_proj.weight", optional=True),
            WeightInfo(name=f"{prefix}.self_attn.q_norm.weight", optional=True),
            WeightInfo(name=f"{prefix}.self_attn.k_norm.weight", optional=True),
        ]

        if self.is_moe:
            num_experts = int(
                self.num_experts or getattr(_text_config(config), "num_experts", 0) or 0
            )
            res.append(WeightInfo(name=f"{prefix}.mlp.gate.weight", optional=True))
            res.extend(
                WeightInfo(name=f"{prefix}.mlp.{name}", optional=True)
                for name in (
                    "experts.gate_up_proj",
                    "experts.down_proj",
                )
            )
            for expert_idx in range(num_experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    res.append(
                        WeightInfo(
                            name=f"{prefix}.mlp.experts.{expert_idx}.{proj}.weight",
                            optional=True,
                        )
                    )
            if getattr(_text_config(config), "shared_expert_intermediate_size", None):
                res.extend(
                    WeightInfo(name=f"{prefix}.mlp.{name}", optional=True)
                    for name in (
                        "shared_expert.gate_proj.weight",
                        "shared_expert.up_proj.weight",
                        "shared_expert.down_proj.weight",
                        "shared_expert_gate.weight",
                    )
                )
        else:
            res.extend(
                WeightInfo(name=f"{prefix}.mlp.{name}", optional=True)
                for name in (
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                )
            )

        res.append(
            WeightInfo(name=f"{prefix}.post_attention_layernorm.weight", optional=True)
        )
        return res


class Qwen35VisionModuleArchitecture(ModuleArchitecture, BaseModel, frozen=True):
    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return [
            WeightInfo(name="model.visual.patch_embed.proj.weight", is_embed=True),
            WeightInfo(name="model.visual.patch_embed.proj.bias", optional=True),
            WeightInfo(name="model.visual.pos_embed.weight", is_embed=True),
            WeightInfo(name="model.visual.merger.norm.weight"),
            WeightInfo(name="model.visual.merger.norm.bias", optional=True),
            WeightInfo(name="model.visual.merger.linear_fc1.weight"),
            WeightInfo(name="model.visual.merger.linear_fc1.bias", optional=True),
            WeightInfo(name="model.visual.merger.linear_fc2.weight"),
            WeightInfo(name="model.visual.merger.linear_fc2.bias", optional=True),
        ]

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return []

    def num_layers_config_key(self) -> str:
        return "vision_config.depth"

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        prefix = f"model.visual.blocks.{index}"
        return [
            WeightInfo(name=f"{prefix}.norm1.weight"),
            WeightInfo(name=f"{prefix}.norm1.bias", optional=True),
            WeightInfo(name=f"{prefix}.norm2.weight"),
            WeightInfo(name=f"{prefix}.norm2.bias", optional=True),
            WeightInfo(name=f"{prefix}.attn.qkv.weight"),
            WeightInfo(name=f"{prefix}.attn.qkv.bias", optional=True),
            WeightInfo(name=f"{prefix}.attn.proj.weight"),
            WeightInfo(name=f"{prefix}.attn.proj.bias", optional=True),
            WeightInfo(name=f"{prefix}.mlp.linear_fc1.weight"),
            WeightInfo(name=f"{prefix}.mlp.linear_fc1.bias", optional=True),
            WeightInfo(name=f"{prefix}.mlp.linear_fc2.weight"),
            WeightInfo(name=f"{prefix}.mlp.linear_fc2.bias", optional=True),
        ]


def qwen35_architecture_for_config(config: PretrainedConfig) -> ModelArchitecture:
    arch_name = (
        config.architectures[0] if getattr(config, "architectures", None) else ""
    )
    is_moe = arch_name in QWEN35_MOE_ARCHITECTURES or config.model_type in {
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
    num_experts = (
        int(getattr(_text_config(config), "num_experts", 0) or 0) if is_moe else None
    )
    is_multimodal_wrapper = arch_name.endswith("ForConditionalGeneration") and hasattr(
        config, "vision_config"
    )
    root = "model.language_model" if is_multimodal_wrapper else "model"
    mtp_num_layers_key = (
        "text_config.mtp_num_hidden_layers"
        if is_multimodal_wrapper
        else "mtp_num_hidden_layers"
    )

    modules = {
        "text_decoder" if is_multimodal_wrapper else "default": ModuleDefinition(
            architecture=Qwen35LanguageModuleArchitecture(
                root=root,
                is_moe=is_moe,
                num_experts=num_experts,
            )
        )
    }
    if is_multimodal_wrapper:
        modules["vision_tower"] = ModuleDefinition(
            architecture=Qwen35VisionModuleArchitecture()
        )

    if _cfg(config, mtp_num_layers_key, 0):
        modules["mtp"] = ModuleDefinition(
            architecture=Qwen35MtpModuleArchitecture(
                is_moe=is_moe,
                num_experts=num_experts,
                num_layers_key=mtp_num_layers_key,
            )
        )

    return ModelArchitecture(
        modules=modules,
        architectures=[arch_name] if arch_name else [],
        model_type=config.model_type,
        tagalong_files=(
            [
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "vocab.json",
            ]
            if is_multimodal_wrapper
            else None
        ),
        vocab_size_config_key=(
            "text_config.vocab_size" if is_multimodal_wrapper else "vocab_size"
        ),
    )
