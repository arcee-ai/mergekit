# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import json
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional

import safetensors
import torch

LOG = logging.getLogger(__name__)


class TensorWriter:
    out_path: str
    override_basename: Optional[str]
    max_shard_size: int
    safe_serialization: bool
    use_async: bool

    shards_written: int
    weight_map: Dict[str, str]
    current_shard: Dict[str, torch.Tensor]
    current_shard_size: int

    _lock: threading.RLock
    _executor: Optional[ThreadPoolExecutor]
    _write_futures: List[Future]

    def __init__(
        self,
        out_path: str,
        max_shard_size: int = 1000 * 1000 * 1000 * 5,
        safe_serialization: bool = True,
        override_basename: Optional[str] = None,
        use_async: bool = False,
        max_write_threads: int = 1,
    ) -> None:
        os.makedirs(out_path, exist_ok=True)

        self.out_path = out_path
        self.override_basename = override_basename
        self.max_shard_size = max_shard_size
        self.safe_serialization = safe_serialization
        self.use_async = use_async

        self.shards_written = 0
        self.weight_map = {}
        self.current_shard = {}
        self.current_shard_size = 0
        self.total_size = 0

        self._lock = threading.RLock()
        self._write_futures = []
        if self.use_async:
            self._executor = ThreadPoolExecutor(max_workers=max_write_threads)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()

    def save_tensor(self, name: str, tensor: torch.Tensor, clone: bool = False):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if clone:
            tensor = tensor.clone()

        tensor_size = tensor.numel() * tensor.element_size()
        with self._lock:
            if (
                self.current_shard
                and self.max_shard_size > 0
                and self.current_shard_size + tensor_size > self.max_shard_size
            ):
                self._flush_current_shard()

            self.current_shard[name] = tensor
            self.current_shard_size += tensor_size

    def _flush_current_shard(self):
        """
        Dispatches the current shard to be written to disk by a background thread.

        This method must be called within a lock.
        """
        if not self.current_shard:
            return

        shard_to_write = self.current_shard
        shard_index = self.shards_written

        self.total_size += self.current_shard_size
        self.current_shard = {}
        self.current_shard_size = 0
        self.shards_written += 1

        prefix, extension = self._get_name_components()
        shard_name = f"{prefix}-{shard_index + 1}.{extension}"
        shard_path = os.path.join(self.out_path, shard_name)
        for key in shard_to_write:
            self.weight_map[key] = shard_name

        if self.use_async:
            LOG.info(f"Dispatching shard #{shard_index + 1} to be written to disk.")

            future = self._executor.submit(
                self._write_shard_task, shard_to_write, shard_index, shard_path
            )
            self._write_futures.append(future)
        else:
            # directly execute
            self._write_shard_task(
                shard_data=shard_to_write,
                shard_index=shard_index,
                shard_path=shard_path,
            )

    def _write_shard_task(
        self, shard_data: Dict[str, torch.Tensor], shard_index: int, shard_path: str
    ):
        LOG.info(f"Writing shard #{shard_index + 1}...")
        if self.safe_serialization:
            self._save_st(shard_data, shard_path)
        else:
            torch.save(shard_data, shard_path)
        LOG.info(f"Finished writing shard #{shard_index + 1}.")

    def finalize(self):
        with self._lock:
            self._flush_current_shard()

        if self.use_async:
            if self._write_futures:
                LOG.info(
                    f"Waiting for {len(self._write_futures)} shard{'s' if len(self._write_futures) > 1 else ''} to finish writing..."
                )
                for future in self._write_futures:
                    future.result()
                LOG.info("All shards have been written to disk.")
                self._write_futures.clear()
            self._executor.shutdown()

        with self._lock:
            LOG.info("Finalizing shard names and creating index file.")
            prefix, extension = self._get_name_components()
            total_shards = self.shards_written

            # Standardize shard names to Hugging Face format
            name_remap = {}
            if total_shards == 1:
                name_remap[f"{prefix}-1.{extension}"] = f"{prefix}.{extension}"
            else:
                for idx in range(total_shards):
                    old_name = f"{prefix}-{idx + 1}.{extension}"
                    new_name = (
                        f"{prefix}-{idx + 1:05d}-of-{total_shards:05d}.{extension}"
                    )
                    name_remap[old_name] = new_name

            for old_name, new_name in name_remap.items():
                old_path = os.path.join(self.out_path, old_name)
                new_path = os.path.join(self.out_path, new_name)
                os.rename(old_path, new_path)

            # Write index file if needed
            if total_shards > 1:
                for key in self.weight_map:
                    self.weight_map[key] = name_remap.get(
                        self.weight_map[key], self.weight_map[key]
                    )

                index_filename = f"{prefix}.{extension}.index.json"
                index_path = os.path.join(self.out_path, index_filename)
                with open(index_path, "w", encoding="utf-8") as f:
                    content = {
                        "metadata": {
                            "total_size": self.total_size,
                            "mergekit_version": "0.1.4",
                        },
                        "weight_map": self.weight_map,
                    }
                    json.dump(content, f, indent=2)

    def _get_name_components(self):
        if self.override_basename:
            basename = self.override_basename
        else:
            basename = "model" if self.safe_serialization else "pytorch_model"

        extension = "safetensors" if self.safe_serialization else "bin"
        return basename, extension

    def _save_st(self, shard_data: dict, shard_path: str):
        def _do_save(sd):
            safetensors.torch.save_file(
                sd,
                shard_path,
                metadata={"format": "pt"},
            )

        try:
            _do_save(shard_data)
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
                shard_data = {key: shard_data[key].clone() for key in shard_data}
                _do_save(shard_data)
            else:
                raise
