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

import logging
import os
import os.path
from typing import List, Optional, Union

import huggingface_hub
import numpy as np
import peft
import torch
import transformers
from pydantic import BaseModel
from transformers import AutoConfig, PretrainedConfig

from mergekit.lazy_tensors import ShardedTensorIndex


class ModelReference(BaseModel):
    """A reference to a language model.

    Can be a hf hub path (username/repo), or local. Optionally includes a LoRA."""

    path: str
    lora_path: Optional[str] = None

    def merged(self, cache_dir: Optional[str] = None) -> "ModelReference":
        """Merge the LoRA if applicable and return a reference to the result."""
        if not self.lora_path:
            return self

        if not cache_dir:
            raise RuntimeError("Need to specify cache dir to merge adapters")

        out_path = os.path.join(
            cache_dir,
            os.path.basename(self.path) + "_" + os.path.basename(self.lora_path),
        )

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            logging.info(f"Loading {self.path} for merge...")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            model = peft.PeftModel.from_pretrained(
                model, self.lora_path, is_trainable=False
            )
            logging.info(f"Merging {self.lora_path} into {self.path}")
            model = model.merge_and_unload()
            model.save_pretrained(out_path, safe_serialization=True)
            del model

        return ModelReference(path=out_path)

    def config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(self.path)

    def tensor_index(self, cache_dir: Optional[str] = None) -> ShardedTensorIndex:
        assert self.lora_path is None

        path = self.path
        if not os.path.exists(path):
            has_safetensors = any(
                fn.lower().endswith(".safetensors")
                for fn in huggingface_hub.list_repo_files(path, repo_type="model")
            )
            patterns = ["tokenizer.model", "*.json"]
            if has_safetensors:
                patterns.append("*.safetensors")
            else:
                patterns.append("*.bin")

            path = huggingface_hub.snapshot_download(
                path, cache_dir=cache_dir, allow_patterns=patterns
            )

        return ShardedTensorIndex.from_disk(path)

    @classmethod
    def parse(cls, value: str) -> "ModelReference":
        """Parse a ModelReference. Format: '<MODEL_PATH>(+<LORA_PATH>)?'"""

        chunks = value.split("+")
        if len(chunks) == 1:
            return ModelReference(path=value)
        elif len(chunks) == 2:
            return ModelReference(path=chunks[0], lora_path=chunks[1])
        raise ValueError(f"Can't parse {value}")

    def __str__(self) -> str:
        if self.lora_path:
            return f"{self.path}+{self.lora_path}"
        return self.path

    class Config:
        frozen = True


def dtype_from_name(name: Optional[str]) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    elif name == "float16":
        return torch.float16
    elif name == "float32":
        return torch.float32
    raise RuntimeError(f'Unimplemented dtype "{name}"')


def rectify_embed_sizes(param_name: str, tensors: List[torch.Tensor]):
    if "lm_head" in param_name or "embed_tokens" in param_name:
        # special case - if lm_head.weight or embed_tokens.weight have a size
        # mismatch, take the largest common submatrix of all of them
        if take_common_submatrix(tensors):
            logging.warning(
                f"Using common submatrix of size {tensors[0].shape} for {param_name}"
            )


def take_common_submatrix(tensors: List[torch.Tensor]) -> bool:
    min_size = [None, None]
    for t in tensors:
        for idx in range(2):
            if min_size[idx] is None or t.shape[idx] < min_size[idx]:
                min_size[idx] = t.shape[idx]

    if not all(t.shape == torch.Size(min_size) for t in tensors):
        for idx in range(len(tensors)):
            tensors[idx] = tensors[idx][: min_size[0], : min_size[1]]
        return True
    return False


def gradient_weights(gradient: List[float], num_samples: int) -> List[float]:
    assert len(gradient) > 1, "Need at least two values to define gradient"

    samples_per_weight = num_samples // (len(gradient) - 1)

    res = []
    for y0, y1 in zip(gradient[:-1], gradient[1:]):
        res.extend(np.linspace(y0, y1, num=samples_per_weight))
    while len(res) < num_samples:
        res.append(gradient[-1])
    return res


def parse_kmb(value: Union[str, int]) -> int:
    if isinstance(value, int):
        return value
    elif value.isnumeric():
        return int(value)
    elif value[-1].lower() == "k":
        return int(value[:-1]) * 1000
    elif value[-1].lower() == "m":
        return int(value[:-1]) * 1000 * 1000
    elif value[-1].lower() == "b":
        return int(value[:-1]) * 1000 * 1000 * 1000
    else:
        raise ValueError(value)
