# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import safetensors
import torch

from mergekit.io.lazy_unpickle import (
    DeferredLoad,
    LazyUnpickleModule,
    TorchArchiveReader,
    torch_lazy_load,
)


class TensorLoader(ABC):
    """Base class for (potentially lazy) tensor loaders."""

    @abstractmethod
    def get_tensor(self, key: str) -> torch.Tensor: ...

    @abstractmethod
    def keys(self) -> Sequence[str]: ...

    @classmethod
    def get(
        cls,
        shard_path: str,
        use_lazy_unpickle: bool = False,
        device: Optional[str] = None,
    ) -> "TensorLoader":
        if shard_path.lower().endswith(".safetensors"):
            return safetensors.safe_open(
                shard_path, framework="pt", device=device or "cpu"
            )
        elif use_lazy_unpickle:
            return LazyPickleLoader(shard_path, device=device)
        return DumbPytorchLoader(shard_path, device=device)


class LazyPickleLoader(TensorLoader):
    """Loader for pytorch files using a custom unpickler and vigorous monkeypatching."""

    zip_reader: TorchArchiveReader
    index: Dict[str, DeferredLoad]
    device: Optional[str] = None
    archive_path: str
    _cached_archive: Optional[Dict[str, torch.Tensor]] = None

    def __init__(self, path: str, device: Optional[str] = None):
        self.archive_path = path
        self.zip_reader = TorchArchiveReader(path)
        self.device = device

        with torch_lazy_load():
            self.index = torch.load(path, pickle_module=LazyUnpickleModule)

    def get_tensor(self, key: str) -> torch.Tensor:
        if key not in self.index:
            raise KeyError(key)

        value = self.index[key]

        if hasattr(value, "execute"):
            return value.execute(self.zip_reader, map_location=self.device)

        if isinstance(value, torch.Tensor):
            return value.to(self.device) if self.device else value

        if self._cached_archive is None:
            self._cached_archive = torch.load(
                self.archive_path, map_location=self.device
            )
        return self._cached_archive[key]

    def keys(self) -> Sequence[str]:
        return self.index.keys()


class DumbPytorchLoader(TensorLoader):
    """Naive `torch.load` shard loading."""

    tensors: Dict[str, torch.Tensor]

    def __init__(self, path: str, device: Optional[str] = None):
        self.tensors = torch.load(path, map_location=device, weights_only=True)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.tensors[key]

    def keys(self) -> Sequence[str]:
        return self.tensors.keys()
