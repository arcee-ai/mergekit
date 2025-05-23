# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

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
from mergekit.config import ParameterSetting, evaluate_setting
from mergekit.graph import Executor, Task
from mergekit.io import LazyTensorLoader, ShardedTensorIndex
from mergekit.io.tasks import FinalizeModel, SaveTensor, TensorWriterTask
from mergekit.merge_methods.base import MergeMethod, TensorDictWrapper
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options


class InputModelDefinition(BaseModel, frozen=True):
    model: str
    parameters: Optional[Dict[str, ParameterSetting]] = None


class RawPyTorchMergeConfig(BaseModel, frozen=True):
    merge_method: str
    parameters: Optional[Dict[str, ParameterSetting]]
    models: List[InputModelDefinition]
    dtype: Optional[str] = None
    base_model: Optional[str] = None


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
        if tensor is None:
            return None
        if dt := dtype_from_name(self.dtype):
            tensor = tensor.to(dtype=dt)
        return tensor


def plan_flat_merge(
    config: RawPyTorchMergeConfig,
    out_path: str,
    tensor_union: bool,
    tensor_intersection: bool,
    options: MergeOptions,
) -> List[Task[torch.Tensor]]:
    merge_method = merge_methods.get(config.merge_method)

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

        tensor_task = merge_method.make_task(
            output_weight=WeightInfo(name=tensor_name),
            tensors=TensorDictWrapper(tensors=inputs),
            parameters=ImmutableMap(global_params),
            tensor_parameters=ImmutableMap(
                data={
                    key: ImmutableMap(data=tensor_params[key]) for key in tensor_params
                }
            ),
            base_model=(
                ModelReference.model_validate({"model": {"path": config.base_model}})
                if config.base_model is not None
                else None
            ),
        )
        save_task = SaveTensor(
            tensor_name=tensor_name,
            tensor_task=tensor_task,
            writer_task=writer_task,
            clone=options.clone_tensors,
            dtype=config.dtype,
        )
        save_tasks.append(save_task)

    finalize = FinalizeModel(tensor_save_tasks=save_tasks, writer_task=writer_task)
    return save_tasks + [finalize]


def construct_param_dicts(
    config: RawPyTorchMergeConfig, merge_method: MergeMethod, tensor_name: str
):
    global_params = {}
    for param_def in merge_method.parameters():
        if param_def.name in config.parameters:
            value = evaluate_setting(tensor_name, config.parameters[param_def.name])
            if value is not None:
                global_params[param_def.name] = value

        if param_def.name not in global_params:
            if param_def.required:
                raise RuntimeError(
                    f"Missing required parameter {param_def.name} for merge method {merge_method}"
                )
            else:
                global_params[param_def.name] = param_def.default_value

    tensor_params = {}
    for param_def in merge_method.tensor_parameters():
        for model_def in config.models:
            mr = ModelReference.model_validate({"model": {"path": model_def.model}})
            tensor_params[mr] = tensor_params.get(mr, {})
            if value := evaluate_setting(
                tensor_name, model_def.parameters.get(param_def.name, [])
            ):
                tensor_params[mr][param_def.name] = value
            elif value := evaluate_setting(
                tensor_name, config.parameters.get(param_def.name, [])
            ):
                tensor_params[mr][param_def.name] = value
            elif param_def.required:
                raise RuntimeError(
                    f"Missing required parameter {param_def.name} for model {mr} tensor {tensor_name}"
                )
            else:
                tensor_params[mr][param_def.name] = param_def.default_value
    return global_params, tensor_params


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
