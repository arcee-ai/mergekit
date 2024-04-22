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
from typing import List, Optional

import torch
import tqdm
import transformers

from mergekit.architecture import MISTRAL_INFO, WeightInfo
from mergekit.moe.arch import MoEOutputArchitecture
from mergekit.moe.common import initialize_io, noise_and_scale, select_dtype
from mergekit.moe.config import MoEMergeConfig
from mergekit.options import MergeOptions


class MixtralMoE(MoEOutputArchitecture):
    def name(self) -> str:
        return "Mixtral"

    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
        if config.shared_experts:
            if explain:
                logging.warning("Mixtral does not support shared experts")
            return False

        model_types = []
        for model_ref in [config.base_model] + [e.source_model for e in config.experts]:
            model_cfg = model_ref.config(trust_remote_code=trust_remote_code)
            model_types.append(model_cfg.model_type)

        if len(set(model_types)) != 1:
            if explain:
                logging.warning(
                    "Mixtral requires all input models to have the same architecture"
                )
            return False
        if model_types[0] not in ("llama", "mistral"):
            if explain:
                logging.warning(
                    "Mixtral requires all input models to be Llama or Mistral models"
                )
            return False
        return True

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
        return f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"

    def write_model(
        self,
        out_path: str,
        config: MoEMergeConfig,
        merge_options: MergeOptions,
        router_weights: List[torch.Tensor],
        shared_router_weights: Optional[List[torch.Tensor]] = None,
    ):
        base_model = config.base_model
        base_cfg = base_model.config(trust_remote_code=merge_options.trust_remote_code)

        assert len(router_weights) == base_cfg.num_hidden_layers, (
            f"Expected {base_cfg.num_hidden_layers} router weights, "
            f"got {len(router_weights)}"
        )

        out_dtype = select_dtype(config, base_cfg)
        out_cfg = self._generate_config(
            base_cfg,
            len(config.experts),
            len(config.shared_experts or []),
            config.experts_per_token,
        )
        out_cfg.torch_dtype = out_dtype
        out_cfg.save_pretrained(out_path)

        loaders, base_loader, writer = initialize_io(config, out_path, merge_options)
        for weight_info in tqdm.tqdm(
            MISTRAL_INFO.all_weights(base_cfg),
            desc="Weights",
        ):
            tensor_name = self._remap_weight_name(weight_info)
            if "{expert_idx}" in tensor_name:
                for expert_index, expert in enumerate(config.experts):
                    expert_name = tensor_name.replace("{expert_idx}", str(expert_index))
                    expert_loader = loaders.get(expert.source_model)
                    tensor = expert_loader.get_tensor(
                        weight_info.name, aliases=weight_info.aliases
                    )
                    tensor = noise_and_scale(
                        tensor, expert, is_residual="down_proj" in tensor_name
                    )
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
                weight.to(dtype=out_dtype).contiguous(),
                clone=merge_options.clone_tensors,
            )

        writer.finalize()
