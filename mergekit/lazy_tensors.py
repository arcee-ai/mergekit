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

import json
import logging
import os
import os.path
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import safetensors
import safetensors.torch
import torch
from torch import Tensor


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

    def as_safetensors(self) -> "ShardedTensorIndex":
        if self.is_safetensors:
            return self

        new_shards = []
        new_tensor_paths = {}
        for shard in self.shards:
            new_name = shard.filename.replace("pytorch_model.bin", "model.safetensors")
            new_shards.append(ShardInfo(new_name, shard.contained_keys))
            for key in shard.contained_keys:
                new_tensor_paths[key] = new_name

        return ShardedTensorIndex(self.base_path, True, new_tensor_paths, new_shards)

    def save(self, output_path: Optional[str] = None):
        if output_path is None:
            output_path = self.base_path

        data = {"metadata": {}, "weight_map": self.tensor_paths}
        index_name = (
            "model.safetensors.index.json"
            if self.is_safetensors
            else "pytorch_model.bin.index.json"
        )
        with open(os.path.join(output_path, index_name), "w", encoding="utf-8") as file:
            json.dump(data, file)

    def load_shard(
        self, shard: Union[ShardInfo, str], device: str = "cpu"
    ) -> Dict[str, Tensor]:
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
                shard = torch.load(model_path)
                if "state_dict" in shard:
                    shard = shard["state_dict"]

                tensor_paths = {key: shard_name for key in shard}

            shards.append(
                ShardInfo(os.path.basename(model_path), list(tensor_paths.keys()))
            )

        return ShardedTensorIndex(base_path, is_safetensors, tensor_paths, shards)


class LazyTensorLoader:
    index: ShardedTensorIndex
    current_shard: Union[None, Dict[str, Tensor], safetensors.safe_open]
    current_keys: Optional[Set[str]]

    def __init__(self, index: ShardedTensorIndex):
        self.index = index
        self.current_shard = None
        self.current_keys = None

    def get_tensor(self, key: str, device: str = "cpu") -> Optional[Tensor]:
        if self.current_shard is None or key not in self.current_keys:
            if key not in self.index.tensor_paths:
                raise KeyError(key)

            self.current_shard = None
            self.current_keys = None

            shard_file = self.index.tensor_paths[key]
            logging.info(f"loading {self.index.base_path}/{shard_file}")
            if shard_file.lower().endswith(".safetensors"):
                self.current_shard = safetensors.safe_open(
                    os.path.join(self.index.base_path, shard_file),
                    framework="pt",
                    device=device,
                )
                self.current_keys = set(self.current_shard.keys())
            else:
                self.current_shard = torch.load(
                    os.path.join(self.index.base_path, shard_file),
                    weights_only=True,
                    map_location=device,
                )
                if "state_dict" in self.current_shard:
                    self.current_shard = self.current_shard["state_dict"]
                self.current_keys = set(self.current_shard.keys())

        if isinstance(self.current_shard, dict):
            return self.current_shard[key]
        return self.current_shard.get_tensor(key).to(device)

    def unload(self):
        self.current_shard = None
        self.current_keys = None


class TensorWriter:
    out_path: str
    max_shard_size: int
    shards_written: int
    weight_map = Dict[str, str]
    current_shard: Dict[str, torch.Tensor]
    current_shard_size: int

    def __init__(
        self, out_path: str, max_shard_size: int = 1000 * 1000 * 1000 * 5
    ) -> None:
        os.makedirs(out_path, exist_ok=True)

        self.out_path = out_path
        self.max_shard_size = max_shard_size
        self.shards_written = 0
        self.weight_map = {}
        self.current_shard = {}
        self.current_shard_size = 0

    def save_tensor(self, name: str, tensor: torch.Tensor, clone: bool = False):
        tensor_size = tensor.view(-1).shape[0]
        if (
            self.current_shard
            and self.current_shard_size + tensor_size > self.max_shard_size
        ):
            self.flush_current_shard()

        if clone:
            tensor = tensor.clone()

        self.current_shard[name] = tensor
        self.current_shard_size += tensor_size

    def flush_current_shard(self):
        if not self.current_shard:
            return

        logging.info("writing shard to disk")

        shard_name = f"model-{self.shards_written+1}.safetensors"
        for key in self.current_shard:
            self.weight_map[key] = shard_name
        safetensors.torch.save_file(
            self.current_shard,
            os.path.join(self.out_path, shard_name),
            metadata={"format": "pt"},
        )
        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written = self.shards_written + 1

    def finalize(self):
        self.flush_current_shard()

        # standardize shard names to hf format
        total_shards = self.shards_written
        name_remap = {}
        for idx in range(total_shards):
            name_remap[
                f"model-{idx+1}.safetensors"
            ] = f"model-{idx+1:05d}-of-{total_shards:05d}.safetensors"

        for old_name, new_name in name_remap.items():
            os.rename(
                os.path.join(self.out_path, old_name),
                os.path.join(self.out_path, new_name),
            )

        for key in self.weight_map:
            self.weight_map[key] = name_remap[self.weight_map[key]]

        with open(
            os.path.join(self.out_path, "model.safetensors.index.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "metadata": {"mergekit_version": "0.0.2.2"},
                    "weight_map": self.weight_map,
                },
                file,
            )
