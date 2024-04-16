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

import json
import logging
import os
from typing import Dict, List, Optional

import torch
import tqdm
import transformers

from mergekit.architecture import get_architecture_info
from mergekit.moe.arch import MoEOutputArchitecture
from mergekit.moe.common import initialize_io, noise_and_scale, select_dtype
from mergekit.moe.config import MoEMergeConfig
from mergekit.options import MergeOptions


class DeepseekMoE(MoEOutputArchitecture):
    def name(self) -> str:
        return "DeepSeek MoE"

    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
        if config.shared_experts:
            if len(config.shared_experts) > 1:
                if explain:
                    logging.warning(
                        "DeepSeek MoE merge does not support more than one shared expert"
                    )
                return False

            if (
                config.shared_experts[0].positive_prompts
                or config.shared_experts[0].negative_prompts
            ):
                if explain:
                    logging.warning(
                        "DeepSeek MoE merge does not support gating shared experts"
                    )
                return False

        model_types = []
        for model_ref in (
            [config.base_model]
            + [e.source_model for e in config.experts]
            + [e.source_model for e in (config.shared_experts or [])]
        ):
            model_cfg = model_ref.config(trust_remote_code=trust_remote_code)
            model_types.append(model_cfg.model_type)

        if len(set(model_types)) != 1:
            if explain:
                logging.warning(
                    "Deepseek MoE requires all input models to have the same architecture"
                )
            return False
        if model_types[0] not in ("llama", "mistral"):
            if explain:
                logging.warning(
                    "Deepseek MoE requires all input models to be Llama or Mistral models"
                )
            return False
        return True

    def _generate_config(
        self,
        base_config: transformers.PretrainedConfig,
        num_experts: int,
        shared_experts: Optional[int] = None,
        experts_per_token: Optional[int] = None,
    ) -> Dict:
        if shared_experts and shared_experts > 1:
            raise NotImplementedError(
                "Shared experts must be 0 or 1 for DeepSeek output"
            )

        res = base_config.to_dict()
        res["architectures"] = ["DeepseekForCausalLM"]
        res["model_type"] = "deepseek"
        res["n_routed_experts"] = num_experts
        res["n_shared_experts"] = shared_experts or None
        res["num_experts_per_tok"] = experts_per_token or (1 if shared_experts else 2)
        res["first_k_dense_replace"] = 0
        res["moe_layer_freq"] = 1
        res["scoring_func"] = "softmax"
        res["norm_topk_prob"] = True
        res["moe_intermediate_size"] = res["intermediate_size"]
        res["auto_map"] = {
            "AutoConfig": "deepseek-ai/deepseek-moe-16b-base--configuration_deepseek.DeepseekConfig",
            "AutoModel": "deepseek-ai/deepseek-moe-16b-base--modeling_deepseek.DeepseekModel",
            "AutoModelForCausalLM": "deepseek-ai/deepseek-moe-16b-base--modeling_deepseek.DeepseekForCausalLM",
        }
        return res

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

        out_dtype = select_dtype(config, base_cfg)
        out_cfg = self._generate_config(
            base_cfg,
            len(config.experts),
            len(config.shared_experts or []),
            config.experts_per_token,
        )
        if out_dtype is not None:
            out_cfg["torch_dtype"] = str(out_dtype).removeprefix("torch.")
        with open(os.path.join(out_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(out_cfg, f, indent=4)

        shared_def = config.shared_experts[0] if config.shared_experts else None

        loaders, base_loader, writer = initialize_io(config, out_path, merge_options)
        shared_loader = loaders.get(shared_def.source_model) if shared_def else None
        for weight_info in tqdm.tqdm(
            get_architecture_info(base_cfg).all_weights(base_cfg),
            desc="Weights",
        ):
            tensor_name = weight_info.name
            if ".mlp." in tensor_name:
                for expert_idx, expert in enumerate(config.experts):
                    expert_name = tensor_name.replace(
                        ".mlp.", f".mlp.experts.{expert_idx}."
                    )
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

                if shared_def is not None:
                    shared_tensor = shared_loader.get_tensor(
                        weight_info.name, aliases=weight_info.aliases
                    )
                    shared_tensor = noise_and_scale(
                        shared_tensor,
                        shared_def,
                        is_residual="down_proj" in tensor_name,
                    )
                    writer.save_tensor(
                        tensor_name.replace(".mlp.", ".mlp.shared_experts."),
                        shared_tensor.to(dtype=out_dtype),
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
                f"model.layers.{layer_idx}.mlp.gate.weight",
                weight.to(dtype=out_dtype).contiguous(),
                clone=merge_options.clone_tensors,
            )

        writer.finalize()
