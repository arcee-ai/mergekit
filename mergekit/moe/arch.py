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
from typing import List, Optional

import torch
import tqdm
import transformers

from mergekit.architecture import (
    ArchitectureInfo,
    JsonArchitectureInfo,
    WeightInfo,
    get_architecture_info,
)
from mergekit.common import ModelReference
from mergekit.moe.common import initialize_io, select_dtype
from mergekit.moe.config import MoEMergeConfig
from mergekit.options import MergeOptions


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
    def write_model(
        self,
        out_path: str,
        config: MoEMergeConfig,
        merge_options: MergeOptions,
        out_dtype: Optional[torch.dtype] = None,
    ):
        """Write the config and tensors for the output MoE to the given path."""
        pass


class MixtralMoE(MoEOutputArchitecture):
    def name(self) -> str:
        return "Mixtral"

    def arch_is_compatible(self, arch: ArchitectureInfo) -> bool:
        return isinstance(
            arch, JsonArchitectureInfo
        ) and arch.definition.expected_model_type in ("llama", "mistral")

    def _generate_config(
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

    def _remap_weight_name(self, weight: WeightInfo) -> str:
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

    def _router_weight_name(self, layer_idx: int) -> str:
        return f"block_sparse_moe.router.{layer_idx}.gate.weight"

    def write_model(
        self,
        out_path: str,
        config: MoEMergeConfig,
        merge_options: MergeOptions,
        router_weights: List[torch.Tensor],
    ):
        base_model = ModelReference.model_validate(config.base_model)
        base_cfg = base_model.config(trust_remote_code=merge_options.trust_remote_code)

        assert len(router_weights) == base_cfg.num_hidden_layers, (
            f"Expected {base_cfg.num_hidden_layers} router weights, "
            f"got {len(router_weights)}"
        )

        out_dtype = select_dtype(config, base_cfg)
        out_cfg = self._generate_config(
            base_cfg,
            len(config.experts),
            len(config.shared_experts),
            config.experts_per_token,
        )
        out_cfg.torch_dtype = out_dtype
        out_cfg.save_pretrained(out_path)

        loaders, base_loader, writer = initialize_io(config, out_path, merge_options)
        for weight_info in tqdm.tqdm(
            get_architecture_info(base_cfg).all_weights(base_cfg),
            desc="Weights",
        ):
            tensor_name = self._remap_weight_name(weight_info)
            if "{expert_idx}" in tensor_name:
                for expert_index, expert in enumerate(config.experts):
                    expert_name = tensor_name.replace("{expert_idx}", str(expert_index))
                    expert_loader = loaders.get(expert.model_ref)
                    tensor = expert_loader.get_tensor(
                        weight_info.name, aliases=weight_info.aliases
                    )
                    if expert.noise_scale:
                        tensor += torch.randn_like(tensor) * expert.noise_scale
                    writer.save_tensor(
                        expert_name,
                        tensor.to(dtype=out_dtype),
                        clone=merge_options.clone_tensors,
                    )
            else:
                tensor = base_loader.get_tensor(
                    tensor_name, aliases=weight_info.aliases
                )
                writer.save_tensor(
                    tensor_name,
                    tensor.to(dtype=out_dtype),
                    clone=merge_options.clone_tensors,
                )

        for layer_idx, weight in enumerate(
            tqdm.tqdm(router_weights, desc="Router weights")
        ):
            writer.save_tensor(
                self._router_weight_name(layer_idx),
                weight.to(dtype=out_dtype),
                clone=merge_options.clone_tensors,
            )


ALL_OUTPUT_ARCHITECTURES = [MixtralMoE()]
