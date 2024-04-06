# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
from abc import ABC, abstractmethod
from typing import Optional

import transformers

from mergekit.architecture import (
    MISTRAL_INFO,
    ArchitectureInfo,
    JsonArchitectureInfo,
    WeightInfo,
)


class MoEOutputArchitecture(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for the architecture."""
        pass

    @abstractmethod
    def arch_is_compatible(self, arch: ArchitectureInfo) -> bool:
        """Return True if the architecture can be used as an \"expert\" in the MoE."""
        pass

    @abstractmethod
    def nominal_base_arch(self) -> ArchitectureInfo:
        """Return the nominal architecture for the base model."""
        pass

    @abstractmethod
    def generate_config(
        self,
        base_config: transformers.PretrainedConfig,
        num_experts: int,
        shared_experts: Optional[int] = None,
        experts_per_token: Optional[int] = None,
    ) -> transformers.PretrainedConfig:
        """Generate a config for the output MoE based on the base model's config."""
        pass

    @abstractmethod
    def remap_weight_name(self, weight: WeightInfo) -> str:
        """Remap a weight name from the base model to the MoE model."""
        pass

    @abstractmethod
    def router_weight_name(self, layer_idx: int) -> str:
        """Return the name of the weight for the router for a given layer."""
        pass


class MixtralMoE(MoEOutputArchitecture):
    def name(self) -> str:
        return "Mixtral"

    def arch_is_compatible(self, arch: ArchitectureInfo) -> bool:
        return isinstance(
            arch, JsonArchitectureInfo
        ) and arch.definition.expected_model_type in ("llama", "mistral")

    def nominal_base_arch(self) -> ArchitectureInfo:
        return MISTRAL_INFO

    def generate_config(
        self,
        base_config: transformers.PretrainedConfig,
        num_experts: int,
        shared_experts: Optional[int] = None,
        experts_per_token: Optional[int] = None,
    ) -> transformers.PretrainedConfig:
        if shared_experts:
            raise NotImplementedError("Shared experts not supported for Mixtral output")

        if not isinstance(base_config, transformers.MistralConfig):
            base_cfg_mistral = transformers.MistralConfig(**base_config.to_dict())
            base_cfg_mistral.sliding_window = None
            base_cfg_mistral.max_position_embeddings = (
                base_config.max_position_embeddings
            )
            base_config = base_cfg_mistral

        out_cfg = transformers.MixtralConfig(**base_config.to_dict())
        out_cfg.architectures = ["MixtralForCausalLM"]
        out_cfg.num_local_experts = num_experts
        out_cfg.num_experts_per_tok = experts_per_token or 2
        out_cfg.sliding_window = None

        if (out_cfg.num_local_experts & (out_cfg.num_local_experts - 1)) != 0:
            logging.warning(
                f"Your model has {out_cfg.num_local_experts} experts, which is "
                "not a power of two. The model will not be usable in llama.cpp."
            )
        return out_cfg

    def remap_weight_name(self, weight: WeightInfo) -> str:
        if ".mlp." not in weight.name:
            # Everything but MLP is identical to base Mistral
            return weight.name

        res = weight.name
        for needle, replacement in [
            (".mlp.gate_proj", ".block_sparse_moe.experts.{expert_idx}.w1"),
            (".mlp.down_proj", ".block_sparse_moe.experts.{expert_idx}.w2"),
            (".mlp.up_proj", ".block_sparse_moe.experts.{expert_idx}.w3"),
        ]:
            res = res.replace(needle, replacement)
        return res

    def router_weight_name(self, layer_idx: int) -> str:
        return f"block_sparse_moe.router.{layer_idx}.gate.weight"


ALL_OUTPUT_ARCHITECTURES = [MixtralMoE()]
