# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from typing import Dict, List, Optional

import click
import torch
import tqdm
import yaml
from pydantic import BaseModel

import mergekit.merge_methods as merge_methods
from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference, dtype_from_name
from mergekit.config import ParameterSetting
from mergekit.graph import Executor, Task
from mergekit.io import LazyTensorLoader, ShardedTensorIndex
from mergekit.io.tasks import FinalizeModel, SaveTensor, TensorWriterTask
from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.task_adapter import (
    ExecuteMergeMethodTask,
    TensorDictWrapper,
)
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options
from mergekit.parameter_resolver import resolve_parameters


class InputModelDefinition(BaseModel, frozen=True):
    model: str
    parameters: Optional[Dict[str, ParameterSetting]] = None


class RawPyTorchMergeConfig(BaseModel, frozen=True):
    merge_method: str
    models: List[InputModelDefinition]
    parameters: Optional[Dict[str, ParameterSetting]] = None
    dtype: Optional[str] = None
    base_model: Optional[str] = None
    out_dtype: Optional[str] = None


class SimpleLoaderCache:
    loaders: Dict[str, LazyTensorLoader]
    lazy_unpickle: bool = False
    _instance: Optional["SimpleLoaderCache"] = None

    def __new__(cls) -> "SimpleLoaderCache":
        if cls._instance is None:
            cls._instance = super(SimpleLoaderCache, cls).__new__(cls)
            cls._instance.loaders = {}
        return cls._instance

    def get(self, model: str) -> LazyTensorLoader:
        if model not in self.loaders:
            self.loaders[model] = LazyTensorLoader(
                ShardedTensorIndex.from_file(model), lazy_unpickle=self.lazy_unpickle
            )
        return self.loaders[model]


class SimpleLoadTensor(Task[torch.Tensor]):
    model: str
    tensor_name: str
    dtype: Optional[str] = None
    device: Optional[str] = None

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self) -> torch.Tensor:
        loader = SimpleLoaderCache().get(self.model)
        tensor = loader.get_tensor(self.tensor_name, device=self.device or "cpu")
        if (dtype := dtype_from_name(self.dtype)) is not None:
            if not dtype.is_floating_point:
                raise ValueError("dtype must be a floating-point torch.dtype")
            # Cast before transfer to the math device, preserving checkpoint buffers.
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
        return tensor


def plan_flat_merge(
    config: RawPyTorchMergeConfig,
    out_path: str,
    tensor_union: bool,
    tensor_intersection: bool,
    options: MergeOptions,
) -> List[Task[torch.Tensor]]:
    merge_method = merge_methods.get(config.merge_method)

    configured_ids = [
        ModelReference.model_validate({"model": {"path": model.model}})
        for model in config.models
    ]
    base_id = (
        ModelReference.model_validate({"model": {"path": config.base_model}})
        if config.base_model is not None
        else None
    )
    if base_id is not None and base_id not in configured_ids:
        configured_ids.append(base_id)
    merge_method.validate_inputs(configured_ids, base_id)

    loaders = SimpleLoaderCache()
    loaders.lazy_unpickle = options.lazy_unpickle
    all_tensor_names = set()
    for model_def in tqdm.tqdm(config.models, desc="Preparing model loaders"):
        loader = loaders.get(model_def.model)
        all_tensor_names.update(loader.index.tensor_paths.keys())

    writer_task = TensorWriterTask(
        out_path=out_path,
        max_shard_size=options.out_shard_size,
        safe_serialization=options.safe_serialization,
        use_async=options.async_write,
        max_write_threads=options.write_threads,
    )

    save_tasks = []
    for tensor_name in tqdm.tqdm(list(all_tensor_names), desc="Planning operations"):
        inputs = {
            model_def.model: SimpleLoadTensor(
                model=model_def.model, tensor_name=tensor_name, dtype=config.dtype
            )
            for model_def in config.models
        }
        if config.base_model is not None and config.base_model not in inputs:
            inputs[config.base_model] = SimpleLoadTensor(
                model=config.base_model, tensor_name=tensor_name, dtype=config.dtype
            )

        has_tensor = [
            lt.model
            for lt in inputs.values()
            if lt.tensor_name in loaders.get(lt.model).index.tensor_paths
        ]
        if len(has_tensor) < len(inputs):
            if tensor_intersection:
                continue
            elif tensor_union:
                pass
            else:
                missing = set(inputs) - set(has_tensor)
                logging.warning(f"Tensor {tensor_name} not found in models:")
                for model in missing:
                    logging.warning(f"  {model}")
                logging.warning("Was found in:")
                for model in has_tensor:
                    logging.warning(f"  {model}")
                raise RuntimeError("Missing tensors")

        inputs = {
            ModelReference.model_validate({"model": {"path": k}}): v
            for k, v in inputs.items()
        }

        global_params, tensor_params = construct_param_dicts(
            config, merge_method, tensor_name
        )

        base_model = (
            ModelReference.model_validate({"model": {"path": config.base_model}})
            if config.base_model is not None
            else None
        )
        tensor_input = TensorDictWrapper(tensors=inputs)
        output_weight = WeightInfo(name=tensor_name)
        model_order = tuple(inputs)
        tensor_task = ExecuteMergeMethodTask.from_parameters(
            method_name=merge_method.spec.name,
            gather_tensors=tensor_input,
            model_order=model_order,
            base_model=base_model,
            output_weight=output_weight,
            parameters=global_params,
            input_parameters=tensor_params,
            dtype=config.dtype,
            out_dtype=config.out_dtype,
        )
        save_task = SaveTensor(
            tensor_name=tensor_name,
            tensor_task=tensor_task,
            writer_task=writer_task,
            clone=options.clone_tensors,
        )
        save_tasks.append(save_task)

    finalize = FinalizeModel(tensor_save_tasks=save_tasks, writer_task=writer_task)
    return save_tasks + [finalize]


def construct_param_dicts(
    config: RawPyTorchMergeConfig, merge_method: MergeMethod, tensor_name: str
):
    base_ref = (
        ModelReference.model_validate({"model": {"path": config.base_model}})
        if config.base_model is not None
        else None
    )
    model_settings = {
        ModelReference.model_validate(
            {"model": {"path": model.model}}
        ): model.parameters
        for model in config.models
    }
    # The planner adds an implicit base too; it must receive ALL-target parameters
    # from global settings/defaults even when it has no model-level settings.
    if base_ref is not None:
        model_settings.setdefault(base_ref, None)
    return resolve_parameters(
        merge_method.spec,
        tensor_name=tensor_name,
        inputs={model: tensor_name for model in model_settings},
        sources=lambda model: (model_settings.get(model), config.parameters),
        base_model=base_ref,
    )


@click.command("mergekit-pytorch", cls=PrettyPrintHelp)
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
@click.option(
    "--tensor-intersection",
    "-i",
    type=bool,
    default=False,
    is_flag=True,
    help="Only merge tensors that are present in all input models",
)
@click.option(
    "--tensor-union",
    "-u",
    type=bool,
    default=False,
    is_flag=True,
    help="Merge all tensors present in any input model",
)
@add_merge_options
def main(
    config_path: str,
    out_path: str,
    tensor_union: bool,
    tensor_intersection: bool,
    merge_options: MergeOptions,
):
    """Merge arbitrary PyTorch models.

    Uses similar configuration syntax to `mergekit-yaml`, minus the
    `slices` sections. Each input model should be the path on disk to a
    pytorch pickle file or safetensors file."""
    merge_options.apply_global_options()

    with open(config_path, "r", encoding="utf-8") as file:
        config_source = file.read()

    config = RawPyTorchMergeConfig.model_validate(yaml.safe_load(config_source))
    tasks = plan_flat_merge(
        config, out_path, tensor_union, tensor_intersection, merge_options
    )

    executor = Executor(
        tasks,
        math_device=merge_options.device,
        storage_device=(
            merge_options.device if merge_options.low_cpu_memory else "cpu"
        ),
    )
    executor.execute()
