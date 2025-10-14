# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import json
import logging
import os
from typing import List, Optional

import torch
import tqdm
import transformers

from mergekit.architecture import arch_info_for_config
from mergekit.architecture.json_definitions import NAME_TO_ARCH
from mergekit.moe.arch import MoEOutputArchitecture
from mergekit.moe.common import copy_tensor_out, initialize_io, select_dtype
from mergekit.moe.config import MoEMergeConfig
from mergekit.options import MergeOptions

KORMO_INFO = NAME_TO_ARCH["KORMoForCausalLM"][0]


class KORMoMoE(MoEOutputArchitecture):
    def name(self) -> str:
        return "KORMo MoE"

    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
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
                    "KORMo MoE requires all input models to have the same architecture"
                )
            return False
            
        if model_types[0] != "kormo":
            if explain:
                logging.warning(
                    "KORMo MoE requires input models to be KORMo architecture"
                )
            return False
            
        return True

    def _generate_config(
        self,
        base_config: transformers.PretrainedConfig,
        num_experts: int,
        num_shared_experts: int = 0,
        experts_per_token: Optional[int] = None,
    ) -> dict:
        res = base_config.to_dict()
        res["architectures"] = ["KORMoMoeForCausalLM"]
        res["model_type"] = "kormo_moe"
        res["num_experts"] = num_experts
        res["num_experts_per_tok"] = experts_per_token or 2
        res["decoder_sparse_step"] = 1
        res["norm_topk_prob"] = True
        res["moe_intermediate_size"] = res["intermediate_size"]
        
        if num_shared_experts > 0:
            res["shared_expert_intermediate_size"] = res["intermediate_size"]
        
        if (res["num_experts"] & (res["num_experts"] - 1)) != 0:
            logging.warning(
                f"Your model has {res['num_experts']} experts, which is "
                "not a power of two. The model will not be usable in llama.cpp."
            )
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
    
        # 출력 디렉토리 생성
        os.makedirs(out_path, exist_ok=True)
    
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
        shared_loader = loaders.get(shared_def.source_model) if shared_def else base_loader
        
        for weight_info in tqdm.tqdm(
            KORMO_INFO.all_weights(base_cfg),
            desc="Weights",
        ):
            tensor_name = weight_info.name
            if ".mlp." in tensor_name:
                # Expert weights 복사
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
    
                # Shared expert weights 복사
                if shared_def is not None:
                    shared_expert_name = tensor_name.replace(".mlp.", ".mlp.shared_expert.")
                    copy_tensor_out(
                        weight_info,
                        shared_loader,
                        writer,
                        expert=shared_def,
                        is_residual="down_proj" in tensor_name,
                        output_name=shared_expert_name,
                        out_dtype=out_dtype,
                        clone=merge_options.clone_tensors,
                    )
    
                # Gate weights는 레이어 단위로 저장
                # (이미 모든 expert를 처리했으면 gate도 저장)
                if expert_idx == len(config.experts) - 1:
                    layer_idx = int(tensor_name.split(".")[2])
                    gate_name = f"model.layers.{layer_idx}.mlp.gate.weight"
                    writer.save_tensor(
                        gate_name,
                        router_weights[layer_idx].to(dtype=out_dtype),
                        clone=merge_options.clone_tensors,
                    )
                    
                    if shared_router_weights is not None and shared_def is not None:
                        shared_gate_name = f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight"
                        writer.save_tensor(
                            shared_gate_name,
                            shared_router_weights[layer_idx].to(dtype=out_dtype),
                            clone=merge_options.clone_tensors,
                        )
            else:
                # MLP가 아닌 weights는 base model에서 복사
                copy_tensor_out(
                    weight_info,
                    base_loader,
                    writer,
                    out_dtype=out_dtype,
                    clone=merge_options.clone_tensors,
                )
        
        writer.finalize()