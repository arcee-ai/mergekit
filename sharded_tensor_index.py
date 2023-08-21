import json
import os
import os.path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import safetensors
import safetensors.torch
import torch
from torch import Tensor


@dataclass
class ShardInfo:
    filename: str
    contained_keys: List[str]


@dataclass
class AllToOne:
    value: Any

    def __contains__(self, key: Any):
        return True

    def __getitem__(self, key: Any):
        return self.value

    def __len__(self):
        return 1

    def values(self):
        return [self.value]


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
            tensor_paths = AllToOne(model_path)
            shards.append(ShardInfo(os.path.basename(model_path), ["*"]))

        return ShardedTensorIndex(base_path, is_safetensors, tensor_paths, shards)


class LazyTensorLoader:
    index: ShardedTensorIndex
    current_shard: Optional[Dict[str, Tensor]]

    def __init__(self, index: ShardedTensorIndex):
        self.index = index
        self.current_shard = None

    def get_tensor(self, key: str, device: str = "cpu") -> Optional[Tensor]:
        if not self.current_shard or key not in self.current_shard:
            if key not in self.index.tensor_paths:
                raise KeyError(key)
            self.current_shard = self.index.load_shard(
                self.index.tensor_paths[key], device
            )

        return self.current_shard[key]
