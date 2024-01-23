import os
from typing import Dict, Optional, Tuple

import torch
from torch._tensor import Tensor

from mergekit.common import ImmutableMap, ModelReference, dtype_from_name
from mergekit.graph import Task
from mergekit.io.lazy_tensor_loader import LazyTensorLoader
from mergekit.io.tensor_writer import TensorWriter


class LoaderCache:
    loaders: Dict[ModelReference, LazyTensorLoader] = {}
    lora_cache_dir: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    lazy_unpickle: bool = False
    trust_remote_code: bool = False

    # singleton instance
    _instance: Optional["LoaderCache"] = None

    def __new__(cls) -> "LoaderCache":
        if cls._instance is None:
            cls._instance = super(LoaderCache, cls).__new__(cls)
        return cls._instance

    def get(self, model: ModelReference) -> LazyTensorLoader:
        if model not in self.loaders:
            merged = model.merged(
                cache_dir=self.lora_cache_dir, trust_remote_code=self.trust_remote_code
            )
            self.loaders[model] = LazyTensorLoader(
                merged.tensor_index(cache_dir=self.hf_cache_dir),
                lazy_unpickle=self.lazy_unpickle,
            )
        return self.loaders[model]

    def flush_all(self):
        for loader in self.loaders.values():
            loader.flush()


def _normalized_shard_name(path: str) -> int:
    name, _ext = os.path.splitext(os.path.basename(path))
    return name.lower().replace("pytorch_model", "model")


class LoadTensor(Task[torch.Tensor]):
    model: ModelReference
    tensor: str
    dtype: Optional[str] = None
    device: Optional[str] = None

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self) -> torch.Tensor:
        loader = LoaderCache().get(self.model)
        x = loader.get_tensor(self.tensor, device=self.device or "cpu")
        if self.dtype:
            x = x.to(dtype=dtype_from_name(self.dtype))
        return x

    def priority(self) -> int:
        return -1000

    def group_label(self) -> Optional[str]:
        loader = LoaderCache().get(self.model)
        shard_path = loader.index.tensor_paths[self.tensor]
        return _normalized_shard_name(shard_path)


class GatherTensors(Task[Dict[ModelReference, torch.Tensor]]):
    tensor_names: ImmutableMap[ModelReference, str]
    dtype: Optional[str] = None
    device: Optional[str] = None

    def arguments(self) -> Dict[str, Task]:
        return {
            f"{str(model)}:{tensor_name}": LoadTensor(
                model=model, tensor=tensor_name, dtype=self.dtype, device=self.device
            )
            for (model, tensor_name) in self.tensor_names.items()
        }

    def group_label(self) -> Optional[str]:
        return max(t.group_label() or "" for t in self.arguments().values())

    def priority(self) -> int:
        return -10

    def execute(self, **kwargs) -> Dict[ModelReference, Tensor]:
        key2model = {
            f"{str(model)}:{tensor_name}": model
            for (model, tensor_name) in self.tensor_names.items()
        }
        return {key2model[key]: kwargs[key] for key in key2model}


class TensorWriterTask(Task[TensorWriter]):
    out_path: str
    max_shard_size: int
    safe_serialization: bool = True

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **_kwargs) -> TensorWriter:
        return TensorWriter(
            self.out_path,
            max_shard_size=self.max_shard_size,
            safe_serialization=self.safe_serialization,
        )


class SaveTensor(Task[None]):
    tensor_name: str
    tensor_task: Task
    writer_task: TensorWriterTask
    clone: bool

    def arguments(self) -> Dict[str, Task]:
        return {"writer": self.writer_task, "tensor": self.tensor_task}

    def priority(self) -> int:
        return 1000

    def group_label(self) -> Optional[str]:
        return self.tensor_task.group_label()

    def execute(self, writer: TensorWriter, tensor: torch.Tensor) -> None:
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
