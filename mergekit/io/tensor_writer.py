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
from typing import Dict

import safetensors
import torch


class TensorWriter:
    out_path: str
    max_shard_size: int
    shards_written: int
    weight_map = Dict[str, str]
    current_shard: Dict[str, torch.Tensor]
    current_shard_size: int
    safe_serialization: bool

    def __init__(
        self,
        out_path: str,
        max_shard_size: int = 1000 * 1000 * 1000 * 5,
        safe_serialization: bool = True,
    ) -> None:
        os.makedirs(out_path, exist_ok=True)

        self.out_path = out_path
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
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

        logging.info(f"Writing shard #{self.shards_written+1} to disk")

        prefix, extension = self._get_name_components()
        shard_name = f"{prefix}-{self.shards_written+1}.{extension}"
        for key in self.current_shard:
            self.weight_map[key] = shard_name

        shard_path = os.path.join(self.out_path, shard_name)
        if self.safe_serialization:
            self._save_st(shard_path)
        else:
            torch.save(self.current_shard, shard_path)

        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written = self.shards_written + 1

    def finalize(self):
        self.flush_current_shard()

        logging.info("Finalizing shard names")

        prefix, extension = self._get_name_components()

        # standardize shard names to hf format
        total_shards = self.shards_written
        name_remap = {}
        for idx in range(total_shards):
            name_remap[
                f"{prefix}-{idx+1}.{extension}"
            ] = f"{prefix}-{idx+1:05d}-of-{total_shards:05d}.{extension}"

        for old_name, new_name in name_remap.items():
            os.rename(
                os.path.join(self.out_path, old_name),
                os.path.join(self.out_path, new_name),
            )

        for key in self.weight_map:
            self.weight_map[key] = name_remap[self.weight_map[key]]

        with open(
            os.path.join(self.out_path, f"{prefix}.{extension}.index.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "metadata": {"mergekit_version": "0.0.4.1"},
                    "weight_map": self.weight_map,
                },
                file,
            )

    def _get_name_components(self):
        if self.safe_serialization:
            return "model", "safetensors"
        return "pytorch_model", "bin"

    def _save_st(self, shard_path: str):
        def _do_save():
            safetensors.torch.save_file(
                self.current_shard,
                shard_path,
                metadata={"format": "pt"},
            )

        try:
            _do_save()
        except RuntimeError as e:
            if (
                len(e.args) > 0
                and isinstance(e.args[0], str)
                and "share memory" in e.args[0]
            ):
                logging.warning(
                    "Your model has duplicated tensors but the --clone-tensors "
                    "flag is not set."
                )
                self.current_shard = {
                    key: self.current_shard[key].clone() for key in self.current_shard
                }
                _do_save()
            else:
                raise
