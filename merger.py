import copy
import json
import logging
import os.path
from typing import Any, Callable, Dict, List, Optional
from typing_extensions import Literal

import safetensors.torch
import torch
from pydantic import BaseModel
from tqdm import tqdm

from common import ModelReference, dtype_from_name
from sharded_tensor_index import LazyTensorLoader
from ties import ties_merge_tensor


class MergeConfig(BaseModel):
    models: List[ModelReference]
    out_path: str

    cuda: bool = False
    dtype: Literal[None, "bfloat16", "float16", "float32"] = None

    merge_method: Literal["ties"] = "ties"
    merge_cache: Optional[str] = None
    model_cache: Optional[str] = None

    options: Dict[str, Any] = {}
    overrides: Dict[str, Dict[str, Any]] = {}


class ModelMerger:
    config: MergeConfig
    _merge: Callable[[Dict, str, Dict[ModelReference, torch.Tensor]], torch.Tensor]
    _loaders: Dict[ModelReference, LazyTensorLoader]

    def __init__(self, config: MergeConfig):
        self.config = config
        if self.config.merge_method == "ties":
            self._merge = ties_merge_tensor
        else:
            raise RuntimeError(
                f"Unimplemented merge method '{self.config.merge_method}'"
            )

        self._loaders = {}

    def prepare_models(self):
        for model in self.config.models:
            if not model in self._loaders:
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

            new_shard_name = shard_name_to_st(shard_name)
            out_tensors = {}

            for param_name in tqdm(parameter_names):
                logging.debug(f" - {param_name}")

                tensors = self.gather_tensors(param_name)
                merge_options = self.parameter_merge_options(param_name)
                merged = self._merge(merge_options, param_name, tensors)

                out_tensors[param_name] = merged.cpu()
                weight_map[param_name] = new_shard_name

            safetensors.torch.save_file(
                out_tensors,
                os.path.join(self.config.out_path, new_shard_name),
                metadata={"format": "pt"},
            )

        # save index
        with open(
            os.path.join(self.config.out_path, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as fd:
            json.dump({"metadata": {}, "weight_map": weight_map}, fd)

    def parameter_merge_options(self, param_name: str) -> Dict[str, Any]:
        """Determine the merge options to use for a given parameter."""
        merge_options = copy.copy(self.config.options)
        if param_name in self.config.overrides:
            merge_options.update(self.config.overrides[param_name])
        return merge_options

    def gather_tensors(self, param_name: str) -> Dict[str, torch.Tensor]:
        """Gather tensors for the given parameter name from all models."""
        math_dev = "cuda" if self.config.cuda else "cpu"
        dtype = self.parameter_dtype(param_name)

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

    def parameter_dtype(self, param_name: str) -> Optional[torch.dtype]:
        """Determine the dtype to store a given parameter with."""
        if (
            param_name in self.config.overrides
            and "dtype" in self.config.overrides[param_name]
        ):
            dtype = dtype_from_name(self.config.overrides[param_name]["dtype"])
        elif self.config.dtype is not None:
            dtype = dtype_from_name(self.config.dtype)
        else:
            dtype = None
        return dtype


def shard_name_to_st(name: str) -> str:
    if name.endswith(".bin"):
        name = name[: -len(".bin")] + ".safetensors"
    return name.replace("pytorch_model", "model")
