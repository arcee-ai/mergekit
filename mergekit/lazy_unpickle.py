# Copyright (C) 2023 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import codecs
import collections
import contextlib
import logging
import operator
import os
import pickle
import zipfile
from functools import reduce
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import accelerate
import numpy
import torch
from pydantic import BaseModel, PrivateAttr

ACCEPTABLE_TYPES = {
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("numpy.core.multiarray", "scalar"): numpy.core.multiarray.scalar,
    ("numpy", "dtype"): numpy.core.multiarray.scalar,
    ("_codecs", "encode"): codecs.encode,
    **{
        ("torch", name): getattr(torch, name)
        for name in [
            "DoubleStorage",
            "FloatStorage",
            "HalfStorage",
            "LongStorage",
            "IntStorage",
            "ShortStorage",
            "CharStorage",
            "ByteStorage",
            "BoolStorage",
            "BFloat16Storage",
        ]
    },
}


class DeferredLoad(BaseModel, arbitrary_types_allowed=True):
    name: str
    location: str
    dtype: torch.dtype
    file_offset: Optional[int] = None
    shape: Optional[Union[torch.Size, Tuple[int, ...]]] = None
    stride: Optional[Tuple[int, ...]] = None
    requires_grad: bool = False
    _backward_hooks: Any = PrivateAttr(None)

    @staticmethod
    def rebuild(
        load: "DeferredLoad",
        offset: int,
        shape: Union[torch.Size, Tuple[int, ...]],
        stride: Tuple[int, ...],
    ) -> "DeferredLoad":
        load.shape = shape
        load.stride = stride

        load.file_offset = offset * dtype_bytes(load.dtype)
        return load

    def execute(
        self,
        loader: "LazyZipReader",
        map_location: Any = None,
    ) -> torch.Tensor:
        total_params = reduce(operator.mul, self.shape)
        total_bytes = total_params * dtype_bytes(self.dtype)

        f = loader.load(file_name=self.name, offset=self.file_offset)
        storage = torch.UntypedStorage.from_buffer(
            f.read(total_bytes), "little", dtype=self.dtype
        )
        storage = torch.serialization._get_restore_location(map_location)(
            storage, self.location
        )

        tensor = torch.tensor([], dtype=self.dtype, device=storage.device)
        tensor.set_(storage, 0, self.shape, self.stride)
        tensor.requires_grad = self.requires_grad
        tensor._backward_hooks = self._backward_hooks
        return tensor


class LazyTorchUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in ACCEPTABLE_TYPES:
            return ACCEPTABLE_TYPES[(module, name)]
        raise pickle.UnpicklingError(f"Unsupported type {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        if not isinstance(pid, tuple) or pid[0] != "storage":
            logging.warning(f"Unpickling object with unexpected PID: {repr(pid)}")
            return super().persistent_load(pid)

        storage_type, key, location, _ = pid[1:]
        return DeferredLoad(name=key, location=location, dtype=get_dtype(storage_type))


class LazyZipReader:
    archive: zipfile.ZipFile
    archive_name: str
    file_name: Optional[str] = None
    file: Optional[zipfile.ZipExtFile] = None

    def __init__(self, path: str):
        self.archive = zipfile.ZipFile(path, mode="r")
        self.archive_name = os.path.basename(os.path.normpath(path)).split(".")[0]

    def load(self, file_name: str, offset: int) -> zipfile.ZipExtFile:
        if self.file_name != file_name or (
            self.file is not None and self.file.tell() > offset
        ):
            if self.file is not None:
                self.file.close()

            try:
                fd = self.archive.open(f"archive/data/{file_name}", mode="r")
            except:
                fd = self.archive.open(
                    f"{self.archive_name}/data/{file_name}", mode="r"
                )
            self.file = fd
            self.file_name = file_name

        skip_bytes = offset - self.file.tell()
        assert skip_bytes >= 0
        self.file.seek(skip_bytes, os.SEEK_CUR)

        return self.file


@contextlib.contextmanager
def use_torch_lazy_load():
    try:
        old_unpickler = pickle.Unpickler
        pickle.Unpickler = LazyTorchUnpickler

        old_load = pickle.load

        def load_monkeypatch(*args, **kwargs):
            return pickle.Unpickler(*args, **kwargs).load()

        pickle.load = load_monkeypatch

        old_rebuild_tensor = torch._utils._rebuild_tensor
        torch._utils._rebuild_tensor = DeferredLoad.rebuild

        with accelerate.init_empty_weights():
            yield

    finally:
        torch._utils._rebuild_tensor = old_rebuild_tensor
        pickle.Unpickler = old_unpickler
        pickle.load = old_load


class LazyPickleLoader:
    zip_reader: LazyZipReader
    index: Dict[str, DeferredLoad]

    def __init__(self, path: str):
        self.zip_reader = LazyZipReader(path)
        with use_torch_lazy_load():
            self.index = torch.load(path)

    def get_tensor(self, key: str, map_location: Any = None) -> torch.Tensor:
        if key not in self.index:
            raise KeyError(key)

        return self.index[key].execute(self.zip_reader, map_location=map_location)

    def keys(self) -> Sequence[str]:
        return self.index.keys()


def dtype_bytes(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        ti = torch.finfo(dtype)
    else:
        ti = torch.iinfo(dtype)
    return max(1, ti.bits // 8)


def get_dtype(storage_type: Any):
    if isinstance(storage_type, torch.dtype):
        return storage_type
    dtype = storage_type.dtype
    if not isinstance(dtype, torch.dtype):
        dtype = storage_type(0).dtype
    return dtype
