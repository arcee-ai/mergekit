# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import click
import torch
import torch.nn as nn
import tqdm
import transformers
from pydantic import BaseModel

from mergekit.architecture import WeightInfo, arch_info_for_config
from mergekit.card import generate_card_lora
from mergekit.common import ModelReference, get_auto_cls
from mergekit.graph import Executor, Task
from mergekit.io.tasks import FinalizeModel, LoadTensor, SaveTensor, TensorWriterTask
from mergekit.io.tensor_writer import TensorWriter
from mergekit.multigpu_executor import MultiGPUExecutor
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options

LOG = logging.getLogger("extract_lora")


@click.command("mergekit-extract-lora", cls=PrettyPrintHelp)
@click.option(
    "--model",
    required=True,
    help="Fine-tuned model path",
)
@click.option(
    "--base-model",
    required=True,
    help="Base model path",
)
@click.option(
    "--out-path",
    required=True,
    help="Output path for extracted LoRA adapter",
)
@click.option(
    "--max-rank",
    type=int,
    default=128,
    help="Maximum rank for LoRA decomposition",
)
@click.option(
    "--distribute-scale/--no-distribute-scale",
    is_flag=True,
    default=True,
    help="Distribute scale between A and B matrices",
)
@click.option(
    "--embed-lora/--no-embed-lora",
    is_flag=True,
    default=False,
    help="Extract LoRA weights for embeddings (vs. in modules_to_save)",
)
@click.option(
    "--save-module",
    "modules_to_save",
    type=str,
    multiple=True,
    default=[],
    help="Save the specified module(s) at full rank",
)
@click.option(
    "--exclude-regex",
    "-e",
    "exclude_regexes",
    type=str,
    multiple=True,
    help="Exclude modules matching the specified regex",
)
@click.option(
    "--include-regex",
    "-i",
    "include_regexes",
    type=str,
    multiple=True,
    help="Include modules matching the specified regex",
)
@click.option(
    "--sv-epsilon",
    type=float,
    default=0,
    help="Threshold for singular values to discard",
    show_default=True,
)
@click.option(
    "--skip-undecomposable",
    is_flag=True,
    help="Skip saving undecomposable modules",
    default=False,
)
@add_merge_options
def main(
    base_model: str,
    model: str,
    out_path: str,
    max_rank: int,
    distribute_scale: bool,
    embed_lora: bool,
    modules_to_save: List[str],
    exclude_regexes: List[str],
    include_regexes: List[str],
    sv_epsilon: float,
    skip_undecomposable: bool,
    merge_options: MergeOptions,
):
    merge_options.apply_global_options()

    if not modules_to_save:
        modules_to_save = []

    base_model_ref = ModelReference.model_validate(base_model)
    model_ref = ModelReference.model_validate(model)
    plan_result = plan_extraction(
        base_model_ref=base_model_ref.merged(
            cache_dir=merge_options.lora_merge_cache,
            trust_remote_code=merge_options.trust_remote_code,
            lora_merge_dtype=merge_options.lora_merge_dtype,
        ),
        model_ref=model_ref.merged(
            cache_dir=merge_options.lora_merge_cache,
            trust_remote_code=merge_options.trust_remote_code,
            lora_merge_dtype=merge_options.lora_merge_dtype,
        ),
        modules_to_save=modules_to_save,
        out_path=out_path,
        options=merge_options,
        max_rank=max_rank,
        distribute_scale=distribute_scale,
        embed_lora=embed_lora,
        exclude_regexes=exclude_regexes,
        include_regexes=include_regexes,
        sv_epsilon=sv_epsilon,
        skip_undecomposable=skip_undecomposable,
    )

    tasks = plan_result.tasks
    if merge_options.multi_gpu:
        executor = MultiGPUExecutor(
            tasks, storage_device="cpu" if not merge_options.low_cpu_memory else None
        )
    else:
        executor = Executor(
            tasks,
            math_device=merge_options.device,
            storage_device=(
                merge_options.device if merge_options.low_cpu_memory else "cpu"
            ),
        )

    module_real_ranks = {}
    for task, result in executor.run():
        if isinstance(task, TaskVectorDecompositionTask):
            module_real_ranks[task.weight_info.name.removesuffix(".weight")] = result[
                0
            ].shape[0]

    real_max_rank = max(module_real_ranks.values())
    config_dict = make_config_dict(
        base_ref=base_model_ref,
        max_rank=real_max_rank,
        modules_to_save=modules_to_save,
        target_modules=list(
            set(key.split(".")[-1] for key in module_real_ranks.keys())
        ),
        module_ranks=module_real_ranks,
    )
    with open(os.path.join(out_path, "adapter_config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    invocation = " ".join(sys.argv)
    with open(os.path.join(out_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            generate_card_lora(
                base_model_ref,
                model_ref,
                invocation,
                os.path.basename(out_path),
                base_vocab_size=plan_result.base_vocab_size,
                final_vocab_size=plan_result.final_vocab_size,
            )
        )

    LOG.info(f"LoRA adapter extracted to {out_path}")


def make_config_dict(
    base_ref: ModelReference,
    max_rank: int,
    modules_to_save: List[str],
    target_modules: List[str],
    module_ranks: Dict[str, int],
):
    different_ranked = {k: v for k, v in module_ranks.items() if v != max_rank}
    return {
        "base_model_name_or_path": base_ref.model.path,
        "peft_type": "LORA",
        "use_rslora": False,
        "target_modules": target_modules,
        "modules_to_save": modules_to_save,
        "task_type": "CAUSAL_LM",
        "r": max_rank,
        "lora_alpha": max_rank,
        "rank_pattern": different_ranked,
        "alpha_pattern": different_ranked,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "inference_mode": True,
    }


class TaskVectorDecompositionTask(Task[Tuple[torch.Tensor, torch.Tensor]]):
    weight_info: WeightInfo
    input_task: Task
    max_rank: int
    distribute_scale: bool = True
    transpose: bool = False
    sv_epsilon: float = 0

    def arguments(self) -> Dict[str, Any]:
        return {"task_vector": self.input_task}

    def execute(self, task_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transpose:
            task_vector = task_vector.T
        out_dtype = task_vector.dtype
        u, s, vh = torch.linalg.svd(
            task_vector.to(dtype=torch.float32), full_matrices=False
        )
        rank = min(self.max_rank, s.shape[0])
        if self.sv_epsilon > 0:
            rank = min((s > self.sv_epsilon).sum().item(), rank)
        if self.distribute_scale:
            sqrt_s = torch.diag(torch.sqrt(s[:rank]))
            scale_a = sqrt_s
            scale_b = sqrt_s
        else:
            scale_a = torch.diag(s[:rank])
            scale_b = torch.eye(rank)
        sqrt_s = torch.diag(torch.sqrt(s[:rank]))
        weight_a = scale_a @ vh[:rank]
        weight_b = u[:, :rank] @ scale_b

        return weight_a.to(dtype=out_dtype), weight_b.to(dtype=out_dtype)

    def group_label(self) -> Optional[str]:
        return self.input_task.group_label()

    def uses_accelerator(self):
        return True


class TaskVectorTask(Task[torch.Tensor]):
    base_tensor: Task
    model_tensor: Task

    def arguments(self) -> Dict[str, Any]:
        return {"base": self.base_tensor, "model": self.model_tensor}

    def execute(self, base: torch.Tensor, model: torch.Tensor) -> torch.Tensor:
        return model - base

    def group_label(self):
        return max(
            self.base_tensor.group_label() or "", self.model_tensor.group_label() or ""
        )

    def uses_accelerator(self):
        return True


class LoRAModuleSaveTask(Task):
    weight_info: WeightInfo
    writer_task: TensorWriterTask
    model_ref: ModelReference
    decomposition_task: TaskVectorDecompositionTask

    def arguments(self) -> Dict[str, Any]:
        return {"writer": self.writer_task, "decomp": self.decomposition_task}

    def execute(
        self, writer: TensorWriter, decomp: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        weight_a, weight_b = decomp
        if weight_a is None or weight_b is None:
            if not self.weight_info.optional:
                raise RuntimeError(
                    f"No SVD decomposition for required weight {self.weight_info.name}"
                )
            return
        lora_type = "lora_embedding" if self.decomposition_task.transpose else "lora"
        lora_suffix = ".weight" if not self.decomposition_task.transpose else ""
        base_name = self.weight_info.name.removesuffix(".weight")
        writer.save_tensor(
            f"base_model.model.{base_name}.{lora_type}_A{lora_suffix}", weight_a
        )
        writer.save_tensor(
            f"base_model.model.{base_name}.{lora_type}_B{lora_suffix}", weight_b
        )

    def priority(self) -> int:
        return 1000

    def group_label(self) -> Optional[str]:
        return self.decomposition_task.group_label()


def _wi_load(model_ref: ModelReference, weight_info: WeightInfo) -> LoadTensor:
    return LoadTensor(
        model=model_ref,
        tensor=weight_info.name,
        dtype=weight_info.force_dtype,
        optional=weight_info.optional,
        aliases=weight_info.aliases,
        tied_names=weight_info.tied_names,
    )


def _make_dummy_model(
    model_ref: ModelReference, trust_remote_code: bool = False
) -> transformers.PreTrainedModel:
    model_cfg = transformers.AutoConfig.from_pretrained(
        model_ref.model.path,
        revision=model_ref.model.revision,
        trust_remote_code=trust_remote_code,
    )
    auto_cls = get_auto_cls(model_cfg.architectures[0])
    with torch.device("meta"):
        res = auto_cls.from_config(model_cfg, trust_remote_code=trust_remote_code)
    return res


class PlanResults(BaseModel):
    tasks: List[Task]
    base_vocab_size: int
    final_vocab_size: int


def plan_extraction(
    base_model_ref: ModelReference,
    model_ref: ModelReference,
    modules_to_save: List[str],
    out_path: str,
    options: MergeOptions,
    max_rank: int,
    distribute_scale: bool = True,
    embed_lora: bool = False,
    exclude_regexes: Optional[List[str]] = None,
    include_regexes: Optional[List[str]] = None,
    sv_epsilon: float = 0,
    skip_undecomposable: bool = False,
) -> PlanResults:
    targets = []
    writer_task = TensorWriterTask(
        out_path=out_path,
        override_basename="adapter_model",
        max_shard_size=-1,
        safe_serialization=options.safe_serialization,
    )

    name_to_wi = all_weights_map(model_ref, options)
    dummy_base = _make_dummy_model(base_model_ref, options.trust_remote_code)
    dummy_model = _make_dummy_model(model_ref, options.trust_remote_code)

    embed_in = dummy_model.get_input_embeddings()
    embed_out = dummy_model.get_output_embeddings()

    ft_vocab = embed_in.weight.shape[0]
    base_vocab = dummy_base.get_input_embeddings().weight.shape[0]
    if ft_vocab != base_vocab and embed_lora:
        LOG.warning(
            f"Vocabulary size mismatch: fine-tuned model has {ft_vocab} tokens, base model has {base_vocab} tokens"
        )
        LOG.warning("Enforcing embeddings in modules_to_save, embed_lora=False")
        embed_lora = False
    del dummy_base

    warned_modules = set()

    def _should_extract(name: str) -> bool:
        if include_regexes and not any(re.search(r, name) for r in include_regexes):
            return False
        if any(re.search(r, name) for r in exclude_regexes):
            return False
        return True

    for name, module in tqdm.tqdm(
        list(dummy_model.named_modules()), desc="Planning operations"
    ):
        wi = name_to_wi.get(name + ".weight")
        bias_wi = name_to_wi.get(name + ".bias")
        if wi is None:
            if hasattr(module, "weight"):
                LOG.warning(
                    f"Weight {name} present in model but not in architecture info"
                )
                wi = WeightInfo(
                    name=name + ".weight",
                    optional=True,
                    is_embed=isinstance(module, nn.Embedding),
                )
            else:
                continue

        if (
            (not embed_lora)
            and (
                module == embed_in
                or module == embed_out
                or isinstance(module, nn.Embedding)
            )
            and not any(re.search(r, name) for r in exclude_regexes or [])
        ):
            # If embeddings are not explicitly excluded but embed_lora is False,
            # save them at full rank instead of decomposing
            key = name.split(".")[-1]
            if key not in modules_to_save:
                LOG.warning(f"Adding {key} to modules_to_save")
                modules_to_save.append(key)

        if name in modules_to_save or (name.split(".")[-1] in modules_to_save):
            LOG.info(f"Planning to save {name} at full rank")
            targets.extend(plan_module_to_save(model_ref, writer_task, wi, bias_wi))
        elif _should_extract(name):
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
                LOG.info(f"Planning LoRA extraction for {name}")
                targets.extend(
                    plan_lora_module(
                        base_model_ref,
                        model_ref,
                        wi,
                        bias_wi,
                        writer_task,
                        max_rank,
                        distribute_scale,
                        transpose=isinstance(module, nn.Embedding),
                        sv_epsilon=sv_epsilon,
                    )
                )
            else:
                key = name.split(".")[-1]
                message = (
                    f"{key} has unsupported module type {type(module).__name__} - "
                    + ("skipping" if skip_undecomposable else "saving at full rank")
                )
                if not skip_undecomposable:
                    # into modules_to_save it goes
                    if key not in modules_to_save:
                        modules_to_save.append(key)
                    targets.extend(
                        plan_module_to_save(model_ref, writer_task, wi, bias_wi)
                    )
                if key not in warned_modules:
                    LOG.warning(message)
                    warned_modules.add(key)

    save_tasks = [t for t in targets if isinstance(t, (SaveTensor, LoRAModuleSaveTask))]
    finalize = FinalizeModel(tensor_save_tasks=save_tasks, writer_task=writer_task)
    return PlanResults(
        tasks=targets + [finalize],
        base_vocab_size=base_vocab,
        final_vocab_size=ft_vocab,
    )


def plan_lora_module(
    base_model_ref: ModelReference,
    model_ref: ModelReference,
    wi: WeightInfo,
    bias_wi: Optional[WeightInfo],
    writer_task: TensorWriterTask,
    max_rank: int,
    distribute_scale: bool = True,
    transpose: bool = False,
    sv_epsilon: float = 0,
) -> List[Task]:
    targets = []
    base_load_task = _wi_load(base_model_ref, wi)
    model_load_task = _wi_load(model_ref, wi)
    tv_task = TaskVectorTask(base_tensor=base_load_task, model_tensor=model_load_task)
    decomp_task = TaskVectorDecompositionTask(
        weight_info=wi,
        input_task=tv_task,
        max_rank=max_rank,
        distribute_scale=distribute_scale,
        transpose=transpose,
        sv_epsilon=sv_epsilon,
    )
    targets.append(decomp_task)
    targets.append(
        LoRAModuleSaveTask(
            weight_info=wi,
            writer_task=writer_task,
            model_ref=model_ref,
            decomposition_task=decomp_task,
        )
    )
    if bias_wi is not None:
        base_bias_load_task = _wi_load(base_model_ref, bias_wi)
        model_bias_load_task = _wi_load(model_ref, bias_wi)
        tv_bias_task = TaskVectorTask(
            base_tensor=base_bias_load_task, model_tensor=model_bias_load_task
        )
        base_bias_name = bias_wi.name.removesuffix(".bias")
        name_out = f"base_model.model.{base_bias_name}.lora_B.bias"
        targets.append(
            SaveTensor(
                tensor_name=name_out,
                tensor_task=tv_bias_task,
                writer_task=writer_task,
                optional=bias_wi.optional,
                clone=False,
            )
        )
    return targets


def plan_module_to_save(
    model_ref: ModelReference,
    writer_task: TensorWriterTask,
    wi: WeightInfo,
    bias_wi: Optional[WeightInfo],
):
    save_tasks = []
    load_task = _wi_load(model_ref, wi)
    save_task = SaveTensor(
        tensor_name=f"base_model.model.{wi.name}",
        tensor_task=load_task,
        writer_task=writer_task,
        optional=wi.optional,
        clone=False,
    )
    save_tasks.append(save_task)
    if bias_wi is not None:
        bias_load_task = _wi_load(model_ref, bias_wi)
        bias_save_task = SaveTensor(
            tensor_name=f"base_model.model.{bias_wi.name}",
            tensor_task=bias_load_task,
            writer_task=writer_task,
            optional=bias_wi.optional,
            clone=False,
        )
        save_tasks.append(bias_save_task)
    return save_tasks


def all_weights_map(
    model_ref: ModelReference, options: MergeOptions
) -> Dict[str, WeightInfo]:
    name_to_wi = {}
    model_cfg = model_ref.config(trust_remote_code=options.trust_remote_code)
    arch_info = arch_info_for_config(model_cfg)
    for wi in arch_info.all_weights(model_cfg):
        name_to_wi[wi.name] = wi
    return name_to_wi


if __name__ == "__main__":
    main()
