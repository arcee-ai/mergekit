# Copyright (C) 2024 Charles O. Goddard
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

import json
import logging
import os
import os.path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import safetensors
import safetensors.torch
import torch
from torch import Tensor

from mergekit.io.loader import TensorLoader


@dataclass
class ShardInfo:
    filename: str
    contained_keys: List[str]


@dataclass
class ShardedTensorIndex:
    base_path: str
    is_safetensors: bool
    tensor_paths: Dict[str, str]
    shards: List[ShardInfo]

    def load_shard(
        self, shard: Union[ShardInfo, str], device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        if isinstance(shard, ShardInfo):
            shard = shard.filename

        shard_path = os.path.join(self.base_path, shard)
        res = {}
        if self.is_safetensors or shard_path.lower().endswith(".safetensors"):
            res = safetensors.torch.load_file(shard_path, device=device)
        else:
            res = torch.load(shard_path, map_location=device, weights_only=True)
            if "state_dict" in res:
                res = res["state_dict"]
        return res

    @classmethod
    def from_disk(cls, base_path: str) -> "ShardedTensorIndex":
        model_path = None
        for model_file_name in ["model.safetensors", "pytorch_model.bin"]:
            candidate_path = os.path.join(base_path, model_file_name)
            if os.path.exists(candidate_path) or os.path.exists(
                candidate_path + ".index.json"
            ):
                model_path = candidate_path
                break

        if not model_path:
            raise RuntimeError(f"Unable to find model files at {base_path}")

        is_safetensors = model_path.endswith(".safetensors")
        tensor_paths = None
        shards = []

        if os.path.exists(model_path + ".index.json"):
            # shared model - parse index
            with open(model_path + ".index.json", "r") as fd:
                weight_map = json.load(fd)["weight_map"]
            tensor_paths = weight_map

            shard_names = list(sorted(set(tensor_paths[e] for e in tensor_paths)))
            for shard_name in shard_names:
                info = ShardInfo(
                    shard_name,
                    [key for key in tensor_paths if tensor_paths[key] == shard_name],
                )
                shards.append(info)

        elif os.path.exists(model_path):
            shard_name = os.path.basename(model_path)

            # get list of tensors contained in single-file checkpoint
            if model_path.lower().endswith(".safetensors"):
                with safetensors.safe_open(model_path, framework="pt") as st:
                    tensor_paths = {key: shard_name for key in st.keys()}
            else:
                # this is ugly but not much else can be done
                shard = torch.load(model_path, map_location="meta")
                if "state_dict" in shard:
                    shard = shard["state_dict"]

                tensor_paths = {key: shard_name for key in shard}

            shards.append(
                ShardInfo(os.path.basename(model_path), list(tensor_paths.keys()))
            )

        return ShardedTensorIndex(base_path, is_safetensors, tensor_paths, shards)


class LazyTensorLoader:
    index: ShardedTensorIndex
    current_shard: Optional[TensorLoader]
    lazy_unpickle: bool

    def __init__(self, index: ShardedTensorIndex, lazy_unpickle: bool = True):
        self.index = index
        self.current_shard = None
        self.lazy_unpickle = lazy_unpickle

    def get_tensor(self, key: str, device: str = "cpu") -> Optional[Tensor]:
        if self.current_shard is None or key not in self.current_shard.keys():
            if key not in self.index.tensor_paths:
                raise KeyError(key)

            self.current_shard = None
            self.current_keys = None

            shard_file = self.index.tensor_paths[key]
            shard_full_path = os.path.join(self.index.base_path, shard_file)
            logging.info(f"Opening shard {shard_full_path}")
            self.current_shard = TensorLoader.get(
                shard_full_path, use_lazy_unpickle=self.lazy_unpickle, device=device
            )

        return self.current_shard.get_tensor(key).to(device)

    def flush(self):
        self.current_shard = None
        self.current_keys = None
