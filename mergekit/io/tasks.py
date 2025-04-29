# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import os
import re
import threading
from typing import Dict, Optional, Tuple

import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference, dtype_from_name
from mergekit.graph import Task
from mergekit.io.lazy_tensor_loader import LazyTensorLoader
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions


class LoaderCache:
    loaders: Dict[ModelReference, LazyTensorLoader] = {}
    lora_cache_dir: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    lazy_unpickle: bool = False
    trust_remote_code: bool = False
    lora_merge_dtype: Optional[str] = None

    # singleton instance per thread
    _instance = threading.local()

    def __new__(cls) -> "LoaderCache":
        if not hasattr(cls._instance, "value"):
            cls._instance.value = super(LoaderCache, cls).__new__(cls)
        return cls._instance.value

    def get(self, model: ModelReference) -> LazyTensorLoader:
        if model not in self.loaders:
            merged = model.merged(
                cache_dir=self.lora_cache_dir,
                trust_remote_code=self.trust_remote_code,
                lora_merge_dtype=self.lora_merge_dtype,
            )
            self.loaders[model] = merged.lazy_loader(
                cache_dir=self.hf_cache_dir, lazy_unpickle=self.lazy_unpickle
            )
        return self.loaders[model]

    def flush_all(self):
        for loader in self.loaders.values():
            loader.flush()

    def setup(self, options: MergeOptions):
        self.lora_cache_dir = options.lora_merge_cache
        self.hf_cache_dir = options.transformers_cache
        self.lazy_unpickle = options.lazy_unpickle
        self.trust_remote_code = options.trust_remote_code
        self.lora_merge_dtype = options.lora_merge_dtype


shard_name_re = re.compile(r"model\-([0-9]+)-of-([0-9]+)")


def _normalized_shard_name(path: str) -> int:
    name, _ext = os.path.splitext(os.path.basename(path))
    name = name.lower().replace("pytorch_model", "model")
    if m := shard_name_re.search(name):
        frac = int(m.group(1)) / int(m.group(2))
        name = f"model-{int(frac * 100):03d}pct"
    return name


class LoadTensor(Task[Optional[torch.Tensor]]):
    model: ModelReference
    tensor: str
    dtype: Optional[str] = None
    device: Optional[str] = None
    optional: bool = False
    aliases: Optional[Tuple[str, ...]] = None
    tied_names: Optional[Tuple[str, ...]] = None
    per_gpu: bool = False

    def arguments(self) -> Dict[str, Task]:
        return {}

    def _resolve_name(self, loader: LazyTensorLoader) -> Optional[str]:
        all_names = (
            [self.tensor] + list(self.aliases or []) + list(self.tied_names or [])
        )
        for name in all_names:
            if name in loader.index.tensor_paths:
                return name
        return None

    def execute(self) -> Optional[torch.Tensor]:
        loader = LoaderCache().get(self.model)
        name = self._resolve_name(loader)
        if not name:
            if not self.optional:
                raise RuntimeError(
                    f"Tensor {self.tensor} required but not present in model {self.model}"
                )
            return None

        x = loader.get_tensor(name, device=self.device or "cpu")
        if self.dtype and (dtype := dtype_from_name(self.dtype)) != x.dtype:
            x = x.to(dtype=dtype)
        return x

    def priority(self) -> int:
        return -1000

    def group_label(self) -> Optional[str]:
        loader = LoaderCache().get(self.model)
        name = self._resolve_name(loader)
        # if name:
        #     shard_path = loader.index.tensor_paths[name]
        #     return _normalized_shard_name(shard_path)
        # return None
        return name

    def duplicate_per_gpu(self):
        return self.per_gpu


class GatherTensors(Task[Dict[ModelReference, torch.Tensor]]):
    weight_info: ImmutableMap[ModelReference, WeightInfo]
    dtype: Optional[str] = None
    device: Optional[str] = None

    def arguments(self) -> Dict[str, Task]:
        return {
            f"{str(model)}:{wi.name}": LoadTensor(
                model=model,
                tensor=wi.name,
                dtype=wi.force_dtype or self.dtype,
                device=self.device,
                optional=wi.optional,
                aliases=wi.aliases,
                tied_names=wi.tied_names,
            )
            for (model, wi) in self.weight_info.items()
        }

    def group_label(self) -> Optional[str]:
        return max(t.group_label() or "" for t in self.arguments().values())

    def priority(self) -> int:
        return -10

    def execute(self, **kwargs) -> Dict[ModelReference, torch.Tensor]:
        key2model = {
            f"{str(model)}:{wi.name}": model for (model, wi) in self.weight_info.items()
        }
        return {
            key2model[key]: kwargs[key] for key in key2model if kwargs[key] is not None
        }


class TensorWriterTask(Task[TensorWriter]):
    out_path: str
    max_shard_size: int
    safe_serialization: bool = True
    override_basename: Optional[str] = None

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **_kwargs) -> TensorWriter:
        return TensorWriter(
            self.out_path,
            max_shard_size=self.max_shard_size,
            safe_serialization=self.safe_serialization,
            override_basename=self.override_basename,
        )

    def priority(self):
        return 10000

    def main_thread_only(self):
        return True


class SaveTensor(Task[None]):
    tensor_name: str
    tensor_task: Task
    writer_task: TensorWriterTask
    clone: bool
    optional: bool = False
    dtype: Optional[str] = None
    force_main_thread: bool = False

    def arguments(self) -> Dict[str, Task]:
        return {"writer": self.writer_task, "tensor": self.tensor_task}

    def priority(self) -> int:
        return 1000

    def group_label(self) -> Optional[str]:
        return self.tensor_task.group_label()

    def main_thread_only(self):
        return self.force_main_thread

    def execute(self, writer: TensorWriter, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            if not self.optional:
                raise RuntimeError(f"No value for required tensor {self.tensor_name}")
            return
        if self.dtype:
            tensor = tensor.to(dtype=dtype_from_name(self.dtype))
        writer.save_tensor(name=self.tensor_name, tensor=tensor, clone=self.clone)


class FinalizeModel(Task[None]):
    tensor_save_tasks: Tuple[Task, ...]
    writer_task: TensorWriterTask

    def arguments(self) -> Dict[str, Task]:
        return {
            "writer": self.writer_task,
            **{f"_unused_{idx}": t for idx, t in enumerate(self.tensor_save_tasks)},
        }

    def execute(self, writer: TensorWriter, **kwargs) -> None:
        writer.finalize()

    def main_thread_only(self):
        return True


class ReturnTensor(Task[torch.Tensor]):
    weight_info: WeightInfo
    tensor_task: Task[torch.Tensor]
    dtype: Optional[str] = None

    def arguments(self) -> Dict[str, Task]:
        return {"tensor": self.tensor_task}

    def priority(self) -> int:
        return 10000

    def group_label(self) -> Optional[str]:
        return self.tensor_task.group_label()

    def execute(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dtype and (dtype := dtype_from_name(self.dtype)) != tensor.dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor
