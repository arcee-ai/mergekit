# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import json
import logging
import os
import threading
from typing import Dict, Optional

import safetensors
import torch

LOG = logging.getLogger(__name__)


class TensorWriter:
    out_path: str
    override_basename: Optional[str]
    max_shard_size: int
    shards_written: int
    weight_map = Dict[str, str]
    current_shard: Dict[str, torch.Tensor]
    current_shard_size: int
    safe_serialization: bool
    lock: threading.Lock

    def __init__(
        self,
        out_path: str,
        max_shard_size: int = 1000 * 1000 * 1000 * 5,
        safe_serialization: bool = True,
        override_basename: Optional[str] = None,
    ) -> None:
        os.makedirs(out_path, exist_ok=True)

        self.out_path = out_path
        self.override_basename = override_basename
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.shards_written = 0
        self.weight_map = {}
        self.current_shard = {}
        self.current_shard_size = 0
        self.lock = threading.Lock()

    def save_tensor(self, name: str, tensor: torch.Tensor, clone: bool = False):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if clone:
            tensor = tensor.clone()

        tensor_size = tensor.numel() * tensor.element_size()
        with self.lock:
            if (
                self.current_shard
                and self.max_shard_size >= 0
                and self.current_shard_size + tensor_size > self.max_shard_size
            ):
                self._flush_current_shard()

            self.current_shard[name] = tensor
            self.current_shard_size += tensor_size

    def _flush_current_shard(self):
        if not self.current_shard:
            return

        LOG.info(f"Writing shard #{self.shards_written + 1} to disk")

        prefix, extension = self._get_name_components()
        shard_name = f"{prefix}-{self.shards_written + 1}.{extension}"

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
        with self.lock:
            self._flush_current_shard()

            LOG.info("Finalizing shard names")

            prefix, extension = self._get_name_components()

            # standardize shard names to hf format
            total_shards = self.shards_written
            name_remap = {}
            for idx in range(total_shards):
                name_remap[f"{prefix}-{idx + 1}.{extension}"] = (
                    f"{prefix}-{idx + 1:05d}-of-{total_shards:05d}.{extension}"
                )

            if total_shards < 2:
                name_remap[f"{prefix}-1.{extension}"] = f"{prefix}.{extension}"

            for old_name, new_name in name_remap.items():
                os.rename(
                    os.path.join(self.out_path, old_name),
                    os.path.join(self.out_path, new_name),
                )

            if total_shards < 2:
                return

            for key in self.weight_map:
                self.weight_map[key] = name_remap[self.weight_map[key]]

            with open(
                os.path.join(self.out_path, f"{prefix}.{extension}.index.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "metadata": {
                            "mergekit_version": "0.1.2",
                        },
                        "weight_map": self.weight_map,
                    },
                    file,
                )

    def _get_name_components(self):
        if self.override_basename:
            return self.override_basename, (
                "safetensors" if self.safe_serialization else "bin"
            )
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
                LOG.warning(
                    "Your model has duplicated tensors but the --clone-tensors "
                    "flag is not set."
                )
                self.current_shard = {
                    key: self.current_shard[key].clone() for key in self.current_shard
                }
                _do_save()
            else:
                raise
