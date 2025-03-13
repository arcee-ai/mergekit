# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import List, Optional

import torch
import tqdm
import transformers

# explicitly import the config class so that we can catch errors upstream
# if the transformers version installed is too old
from transformers.models.qwen2_moe import Qwen2MoeConfig

from mergekit.architecture.json_definitions import NAME_TO_ARCH
from mergekit.moe.arch import MoEOutputArchitecture
from mergekit.moe.common import copy_tensor_out, initialize_io, select_dtype
from mergekit.moe.config import MoEMergeConfig
from mergekit.options import MergeOptions

QWEN2_INFO = NAME_TO_ARCH["Qwen2ForCausalLM"][0]


class QwenMoE(MoEOutputArchitecture):
    def name(self) -> str:
        return "Qwen MoE"

    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
        if len(config.shared_experts or []) != 1:
            if explain:
                logging.warning("Qwen MoE merge requires exactly one shared expert")
            return False

        if (
            config.gate_mode != "random"
            and not config.shared_experts[0].positive_prompts
        ):
            if explain:
                logging.warning("Qwen MoE requires the shared expert to have prompts")
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
                    "Qwen MoE requires all input models to have the same architecture"
                )
            return False
        if model_types[0] not in ("llama", "mistral", "qwen2"):
            if explain:
                logging.warning(
                    "Qwen MoE requires all input models to be Qwen2, Llama or Mistral models"
                )
            return False
        return True

    def _generate_config(
        self,
        base_config: transformers.PretrainedConfig,
        num_experts: int,
        experts_per_token: Optional[int] = None,
    ) -> Qwen2MoeConfig:
        out_cfg = Qwen2MoeConfig(**base_config.to_dict())
        out_cfg.architectures = ["Qwen2MoeForCausalLM"]
        out_cfg.num_experts = num_experts
        out_cfg.num_experts_per_tok = experts_per_token or 2
        out_cfg.decoder_sparse_step = 1
        out_cfg.norm_topk_prob = True
        out_cfg.sliding_window = None
        out_cfg.use_sliding_window = False
        out_cfg.shared_expert_intermediate_size = out_cfg.intermediate_size
        out_cfg.moe_intermediate_size = out_cfg.intermediate_size

        if (out_cfg.num_experts & (out_cfg.num_experts - 1)) != 0:
            logging.warning(
                f"Your model has {out_cfg.num_experts} experts, which is "
                "not a power of two. The model will not be usable in llama.cpp."
            )
        return out_cfg

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
            config.experts_per_token,
        )
        if out_dtype is not None:
            out_cfg.torch_dtype = out_dtype
        out_cfg.save_pretrained(out_path)

        shared_def = config.shared_experts[0]

        loaders, base_loader, writer = initialize_io(config, out_path, merge_options)
        shared_loader = loaders.get(shared_def.source_model) if shared_def else None
        for weight_info in tqdm.tqdm(
            QWEN2_INFO.all_weights(base_cfg),
            desc="Weights",
        ):
            tensor_name = weight_info.name
            if ".mlp." in tensor_name:
                for expert_idx, expert in enumerate(config.experts):
                    expert_name = tensor_name.replace(
                        ".mlp.", f".mlp.experts.{expert_idx}."
                    )
                    expert_loader = loaders.get(expert.source_model)
                    copy_tensor_out(
                        weight_info,
                        expert_loader,
                        writer,
                        expert=expert,
                        is_residual="down_proj" in tensor_name,
                        output_name=expert_name,
                        out_dtype=out_dtype,
                        clone=merge_options.clone_tensors,
                    )

                copy_tensor_out(
                    weight_info,
                    shared_loader,
                    writer,
                    expert=shared_def,
                    is_residual="down_proj" in tensor_name,
                    output_name=tensor_name.replace(".mlp.", ".mlp.shared_expert."),
                    out_dtype=out_dtype,
                    clone=merge_options.clone_tensors,
                )
            else:
                try:
                    tensor = base_loader.get_tensor(
                        tensor_name, aliases=weight_info.aliases
                    )
                except KeyError:
                    if tensor_name.endswith("_proj.bias"):
                        # qwen 2 moe wants attention bias, give it zeros
                        head_dim = out_cfg.hidden_size // out_cfg.num_attention_heads
                        num_heads = (
                            out_cfg.num_key_value_heads
                            if (
                                tensor_name.endswith("k_proj.bias")
                                or tensor_name.endswith("v_proj.bias")
                            )
                            else out_cfg.num_attention_heads
                        )
                        tensor = torch.zeros(num_heads * head_dim, dtype=out_dtype)
                    elif weight_info.optional:
                        continue
                    else:
                        raise

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
            writer.save_tensor(
                f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight",
                shared_router_weights[layer_idx].to(dtype=out_dtype).contiguous(),
                clone=merge_options.clone_tensors,
            )

        writer.finalize()
