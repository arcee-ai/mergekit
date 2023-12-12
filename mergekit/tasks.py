import os
from typing import Dict, List, Optional

import torch
import transformers
from pydantic import BaseModel
from torch._tensor import Tensor

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration
from mergekit.graph import Task
from mergekit.io.lazy_tensor_loader import LazyTensorLoader
from mergekit.io.tensor_writer import TensorWriter
from mergekit.tokenizer import build_tokenizer


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
                merged.tensor_index(self.lora_cache_dir, cache_dir=self.hf_cache_dir),
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
            x = x.to(dtype=self.dtype)
        return x

    def group_label(self) -> Optional[str]:
        loader = LoaderCache().get(self.model)
        shard_path = loader.index.tensor_paths[self.tensor]
        return _normalized_shard_name(shard_path)


class GatherTensors(Task[Dict[ModelReference, torch.Tensor]]):
    tensor_names: Dict[ModelReference, str]
    dtype: Optional[str] = None
    device: Optional[str] = None

    def arguments(self) -> Dict[str, Task]:
        return {
            f"{str(model)}:{tensor_name}": LoadTensor(
                model=model, tensor=tensor_name, dtype=self.dtype, device=self.device
            )
            for (model, tensor_name) in self.tensor_names
        }

    def execute(self, **kwargs) -> Dict[ModelReference, Tensor]:
        key2model = {
            f"{str(model)}:{tensor_name}": model
            for (model, tensor_name) in self.tensor_names
        }
        return {key2model[key]: kwargs[key] for key in key2model}


class TensorWriterTask(Task[TensorWriter]):
    out_path: str
    max_shard_size: int

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **_kwargs) -> TensorWriter:
        return TensorWriter(self.out_path, self.max_shard_size)


class SaveTensor(Task[None]):
    tensor_name: str
    tensor_task: Task
    writer_task: TensorWriterTask
    clone: bool

    def arguments(self) -> Dict[str, Task]:
        return {"writer": self.writer_task, "tensor": self.tensor_task}

    def execute(self, writer: TensorWriter, tensor: torch.Tensor) -> None:
        writer.save_tensor(name=self.tensor_name, tensor=tensor, clone=self.clone)


class FinalizeModel(Task[None]):
    tensor_save_tasks: List[Task]
    writer_task: TensorWriterTask

    def arguments(self) -> Dict[str, Task]:
        return {
            "writer": self.writer_task,
            **{f"_unused_{idx}": t for idx, t in enumerate(self.tensor_save_tasks)},
        }

    def execute(self, writer: TensorWriter, **kwargs) -> None:
        writer.finalize()


class TokenizerInfo(BaseModel, allow_arbitrary_types=True):
    tokenizer: transformers.PreTrainedTokenizerBase
    permutations: Optional[Dict[ModelReference, torch.Tensor]]


class BuildTokenizer(Task[TokenizerInfo]):
    merge_config: MergeConfiguration
    trust_remote_code: bool = False

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **_kwargs) -> TokenizerInfo:
        tokenizer, permutations = build_tokenizer(
            self.merge_config, trust_remote_code=self.trust_remote_code
        )
        return TokenizerInfo(tokenizer=tokenizer, permutations=permutations)
