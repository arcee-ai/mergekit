import copy
import json
import logging
import os.path
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
from pydantic import BaseModel
from tqdm import tqdm
from typing_extensions import Literal

import merge_methods
from common import ModelReference, dtype_from_name
from sharded_tensor_index import LazyTensorLoader


class MergeConfig(BaseModel):
    models: List[ModelReference]
    out_path: str

    merge_method: Literal["ties", "linear"] = "ties"
    merge_cache: Optional[str] = None
    model_cache: Optional[str] = None

    options: Dict[str, Any] = {}
    overrides: Dict[str, Dict[str, Any]] = {}

    cuda: bool = False
    dtype: Literal[None, "bfloat16", "float16", "float32"] = None


class ModelMerger:
    config: MergeConfig
    _loaders: Dict[ModelReference, LazyTensorLoader]

    def __init__(self, config: MergeConfig):
        self.config = config
        self._loaders = {}

    def prepare_models(self):
        for model in self.config.models:
            if model not in self._loaders:
                tensor_index = model.merged(self.config.merge_cache).tensor_index(
                    self.config.model_cache
                )
                self._loaders[model] = LazyTensorLoader(tensor_index)

    def run(self):
        self.prepare_models()
        os.makedirs(self.config.out_path, exist_ok=True)

        shard_info = []
        for shard in self._loaders[self.config.models[0]].index.shards:
            shard_info.append((shard.filename, shard.contained_keys))

        weight_map = {}
        for shard_name, parameter_names in shard_info:
            logging.info(f"Processing shard {shard_name}")
            self.process_shard(weight_map, shard_name, parameter_names)

        # save index
        with open(
            os.path.join(self.config.out_path, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump({"metadata": {}, "weight_map": weight_map}, file)

    def process_shard(self, weight_map, shard_name, parameter_names):
        new_shard_name = shard_name_to_st(shard_name)
        out_tensors = {}

        for param_name in tqdm(parameter_names):
            logging.debug(f" - {param_name}")

            (dtype, method, merge_options) = self.resolve_options(param_name)
            merge_fn = merge_methods.get(method)

            tensors = self.gather_tensors(param_name, dtype=dtype)
            merged = merge_fn(merge_options, param_name, tensors)

            out_tensors[param_name] = merged.cpu()
            weight_map[param_name] = new_shard_name

        safetensors.torch.save_file(
            out_tensors,
            os.path.join(self.config.out_path, new_shard_name),
            metadata={"format": "pt"},
        )

    def resolve_options(
        self, param_name: str
    ) -> Tuple[Optional[torch.dtype], str, Dict[str, Any]]:
        dtype = self.config.dtype
        merge_method = self.config.merge_method
        merge_options = self.config.options

        if param_name in self.config.overrides:
            merge_options = copy.copy(merge_options)
            overrides = self.config.overrides[param_name]
            if "dtype" in overrides:
                dtype = overrides["dtype"]
                del overrides["dtype"]
            if "merge_method" in overrides:
                merge_method = overrides["merge_method"]
                del overrides["merge_method"]
            merge_options.update(overrides)

        if dtype:
            dtype = dtype_from_name(dtype)
        return (dtype, merge_method, merge_options)

    def gather_tensors(
        self, param_name: str, dtype: Optional[torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """Gather tensors for the given parameter name from all models."""
        math_dev = "cuda" if self.config.cuda else "cpu"

        tensors = {}
        for model, loader in self._loaders.items():
            x = loader.get_tensor(param_name)
            if x is None:
                logging.warning(f"{model} has no tensor '{param_name}'")
                continue

            if dtype:
                x = x.to(dtype)
            tensors[model] = x.to(math_dev)
        return tensors


def shard_name_to_st(name: str) -> str:
    if name.endswith(".bin"):
        name = name[: -len(".bin")] + ".safetensors"
    return name.replace("pytorch_model", "model")
