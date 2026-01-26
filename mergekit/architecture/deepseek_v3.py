# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import ClassVar, List, Optional

from pydantic import BaseModel
from transformers import PretrainedConfig

from mergekit.architecture.base import ModuleArchitecture, WeightInfo


class DeepseekV3HybridMoeModuleArchitecture(ModuleArchitecture, BaseModel):
    """
    DeepSeek V3 uses a hybrid dense + MoE MLP:
    - the first `first_k_dense_replace` layers are dense MLP weights:
        mlp.{gate_proj,up_proj,down_proj}.weight
    - the remaining layers are typically MoE (controlled by moe_layer_freq):
        mlp.gate.weight (+ optional e_score_correction_bias),
        mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight,
        mlp.experts.{i}.{gate_proj,up_proj,down_proj}.weight
    """

    ARCHITECTURE_NAME: ClassVar[str] = "DeepseekV3ForCausalLM"

    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    n_routed_experts: int = 0
    n_shared_experts: int = 0

    def name(self) -> str:
        return "deepseek_v3"

    @classmethod
    def from_config(cls, config: PretrainedConfig) -> "DeepseekV3HybridMoeModuleArchitecture":
        return cls(
            first_k_dense_replace=int(getattr(config, "first_k_dense_replace", 0) or 0),
            moe_layer_freq=max(1, int(getattr(config, "moe_layer_freq", 1) or 1)),
            n_routed_experts=int(getattr(config, "n_routed_experts", 0) or 0),
            n_shared_experts=int(getattr(config, "n_shared_experts", 0) or 0),
        )

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        # Observed in DeepSeek V3 safetensors indices
        return [WeightInfo(name="model.embed_tokens.weight", is_embed=True)]

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        # Observed in DeepSeek V3 safetensors indices
        return [
            WeightInfo(name="model.norm.weight"),
            WeightInfo(name="lm_head.weight", is_embed=True),
        ]

    def _is_moe_layer(self, index: int) -> bool:
        """
        DeepSeek V3 is dense for first_k_dense_replace layers, then MoE with frequency.
        """
        if index < self.first_k_dense_replace:
            return False
        # after dense block, apply MoE every `moe_layer_freq` layers
        return ((index - self.first_k_dense_replace) % self.moe_layer_freq) == 0

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        prefix = f"model.layers.{index}"
        res: List[WeightInfo] = []

        # Attention weights (based on observed tensor names in DeepSeek V3 checkpoints)
        for suffix in (
            ".self_attn.o_proj.weight",
            ".self_attn.q_a_proj.weight",
            ".self_attn.q_b_proj.weight",
            ".self_attn.kv_a_proj_with_mqa.weight",
            ".self_attn.kv_b_proj.weight",
            ".self_attn.q_a_layernorm.weight",
            ".self_attn.kv_a_layernorm.weight",
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
        ):
            res.append(WeightInfo(name=prefix + suffix))

        # Dense vs MoE MLP blocks
        if not self._is_moe_layer(index):
            for p in ("gate_proj", "up_proj", "down_proj"):
                res.append(WeightInfo(name=prefix + f".mlp.{p}.weight"))
            return res

        # MoE MLP weights
        res.append(WeightInfo(name=prefix + ".mlp.gate.weight"))
        # Not always present across variants; treat as optional for robustness.
        res.append(
            WeightInfo(
                name=prefix + ".mlp.gate.e_score_correction_bias",
                optional=True,
            )
        )

        if self.n_shared_experts and self.n_shared_experts > 0:
            for p in ("gate_proj", "up_proj", "down_proj"):
                res.append(WeightInfo(name=prefix + f".mlp.shared_experts.{p}.weight"))

        for expert_idx in range(self.n_routed_experts):
            for p in ("gate_proj", "up_proj", "down_proj"):
                res.append(
                    WeightInfo(name=prefix + f".mlp.experts.{expert_idx}.{p}.weight")
                )

        return res


