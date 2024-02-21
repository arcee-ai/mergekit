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

import binascii
import logging
import os
import os.path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    get_args,
)

import huggingface_hub
import immutables
import peft
import torch
import transformers
from pydantic import BaseModel, model_validator
from pydantic_core import core_schema
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import TypeVar

from mergekit.io import ShardedTensorIndex


class ModelPath(BaseModel, frozen=True):
    path: str
    revision: Optional[str] = None

    @model_validator(mode="before")
    def validate_string(cls, value):
        if isinstance(value, str):
            at_ct = value.count("@")
            if at_ct > 1:
                raise RuntimeError(f"Invalid model path - multiple @: {value}")
            elif at_ct == 1:
                path, rev = value.split("@")
                return {"path": path, "revision": rev}
            else:
                return {"path": value}
        return value

    def __str__(self):
        if self.revision:
            return f"{self.path}@{self.revision}"
        return self.path

    def _unique_id(self):
        return (
            os.path.basename(self.path)
            + "_"
            + str(binascii.crc32(self.__str__().encode()))
        )


class ModelReference(BaseModel, frozen=True):
    """A reference to a language model.

    Can be a hf hub path (username/repo), or local. Optionally includes a LoRA."""

    model: ModelPath
    lora: Optional[ModelPath] = None

    def merged(
        self, cache_dir: Optional[str] = None, trust_remote_code: bool = False
    ) -> "ModelReference":
        """Merge the LoRA if applicable and return a reference to the result."""
        if not self.lora:
            return self

        if not cache_dir:
            raise RuntimeError("Need to specify cache dir to merge adapters")

        out_path = os.path.join(
            cache_dir,
            self.model._unique_id() + "_" + self.lora._unique_id(),
        )

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
            logging.info(f"Loading {self.model} for merge...")
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model.path,
                revision=self.model.revision,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            model = peft.PeftModel.from_pretrained(
                model, self.lora.path, revision=self.lora.revision, is_trainable=False
            )
            logging.info(f"Merging {self.lora} into {self.model}")
            model = model.merge_and_unload()
            model.save_pretrained(out_path, safe_serialization=True)
            del model

        return ModelReference(model=out_path)

    def config(self, trust_remote_code: bool = False) -> PretrainedConfig:
        return AutoConfig.from_pretrained(
            self.model.path,
            revision=self.model.revision,
            trust_remote_code=trust_remote_code,
        )

    def tensor_index(self, cache_dir: Optional[str] = None) -> ShardedTensorIndex:
        assert self.lora is None

        path = self.model.path
        if not os.path.exists(path):
            has_safetensors = any(
                fn.lower().endswith(".safetensors")
                for fn in huggingface_hub.list_repo_files(
                    path, repo_type="model", revision=self.model.revision
                )
            )
            patterns = ["tokenizer.model", "*.json"]
            if has_safetensors:
                patterns.append("*.safetensors")
            else:
                patterns.append("*.bin")

            path = huggingface_hub.snapshot_download(
                path,
                revision=self.model.revision,
                cache_dir=cache_dir,
                allow_patterns=patterns,
            )

        return ShardedTensorIndex.from_disk(path)

    @model_validator(mode="before")
    def validate_string(cls, value):
        if isinstance(value, str):
            chunks = value.split("+")
            if len(chunks) == 1:
                return {"model": value}
            elif len(chunks) == 2:
                return {"model": chunks[0], "lora": chunks[1]}
            raise RuntimeError(f"Can't parse {value}")
        return value

    @classmethod
    def parse(cls, value: str) -> "ModelReference":
        """Parse a ModelReference. Format: '<MODEL_PATH>(+<LORA_PATH>)?'"""
        return ModelReference.model_validate(value)

    def __str__(self) -> str:
        if self.lora:
            return f"{str(self.model)}+{str(self.lora)}"
        return str(self.model)


def dtype_from_name(name: Optional[str]) -> torch.dtype:
    if name.startswith("torch."):
        name = name[len("torch.") :]

    if name == "bfloat16":
        return torch.bfloat16
    elif name == "float16":
        return torch.float16
    elif name == "float32":
        return torch.float32
    raise RuntimeError(f'Unimplemented dtype "{name}"')


def rectify_embed_sizes(param_name: str, tensors: List[torch.Tensor]):
    # TODO: use arch_info.embed_weights() instead
    if ("lm_head" in param_name or "embed_tokens" in param_name) and all(
        len(t.shape) == 2 for t in tensors
    ):
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


T_K = TypeVar("T_K")
T_V = TypeVar("T_V")


class ImmutableMap(Generic[T_K, T_V]):
    data: immutables.Map[T_K, T_V]

    def __init__(self, data: Mapping[T_K, T_V]):
        self.data = data

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        instance_schema = core_schema.is_instance_schema(cls)

        args = get_args(source)
        if args:
            dict_schema = handler(Dict[args[0], args[1]])
        else:
            dict_schema = handler(Dict)

        non_instance_schema = core_schema.with_info_after_validator_function(
            lambda value, _info: immutables.Map(value), dict_schema
        )
        return core_schema.union_schema([instance_schema, non_instance_schema])

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, key: T_K) -> T_V:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> Iterator[T_K]:
        return self.data.keys()

    def items(self) -> Iterator[Tuple[T_K, T_V]]:
        return self.data.items()

    def values(self) -> Iterator[T_V]:
        return self.data.values()
