# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

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
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
    get_args,
)

import huggingface_hub
import immutables
import peft
import torch
import transformers
from pydantic import BaseModel, model_serializer, model_validator
from pydantic_core import core_schema
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import TypeVar

from mergekit.io import LazyTensorLoader, ShardedTensorIndex


def set_config_value(config: PretrainedConfig, key: str, value: Any):
    """Set a value in a PretrainedConfig object."""
    parts = key.split(".")
    obj = config
    for idx, part in enumerate(parts[:-1]):
        if not hasattr(obj, part):
            raise RuntimeError(
                f"Config {config} has no attribute {'.'.join(parts[: idx + 1])}"
            )
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def get_config_value(config: PretrainedConfig, key: str) -> Any:
    """Get a value from a PretrainedConfig object."""
    parts = key.split(".")
    obj = config
    for idx, part in enumerate(parts):
        if not hasattr(obj, part):
            raise RuntimeError(
                f"Config {config} has no attribute {'.'.join(parts[: idx + 1])}"
            )
        obj = getattr(obj, part)
    return obj


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
    override_architecture: Optional[str] = None

    def merged(
        self,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        lora_merge_dtype: Optional[str] = None,
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

            config = self.config(trust_remote_code)
            auto_cls = get_auto_cls(config.architectures[0])

            logging.info(f"Loading {self.model} for merge...")
            model = auto_cls.from_pretrained(
                self.model.path,
                revision=self.model.revision,
                torch_dtype=dtype_from_name(lora_merge_dtype),
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

        return ModelReference(model=ModelPath(path=out_path))

    def config(self, trust_remote_code: bool = False) -> PretrainedConfig:
        res = AutoConfig.from_pretrained(
            self.model.path,
            revision=self.model.revision,
            trust_remote_code=trust_remote_code,
        )
        if self.override_architecture:
            res.architectures = [self.override_architecture]
        return res

    def local_path(
        self, cache_dir: Optional[str] = None, ignore_lora: bool = False
    ) -> str:
        if not ignore_lora:
            assert (
                self.lora is None
            ), "LoRA not merged - use .merged() to get a local path"

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
        return path

    def tensor_index(self, cache_dir: Optional[str] = None) -> ShardedTensorIndex:
        return ShardedTensorIndex.from_disk(self.local_path(cache_dir))

    def lazy_loader(
        self, cache_dir: Optional[str] = None, lazy_unpickle: bool = True
    ) -> LazyTensorLoader:
        return LazyTensorLoader(
            self.tensor_index(cache_dir),
            lazy_unpickle=lazy_unpickle,
        )

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

    @model_serializer()
    def serialize(self):
        if self.override_architecture is not None:
            return {
                "model": self.model,
                "lora": self.lora,
                "override_architecture": self.override_architecture,
            }
        res = str(self)
        if '"' in res or " " in res:
            return self
        return res

    @classmethod
    def parse(cls, value: str) -> "ModelReference":
        """Parse a ModelReference. Format: '<MODEL_PATH>(+<LORA_PATH>)?'"""
        return ModelReference.model_validate(value)

    def __str__(self) -> str:
        if self.lora:
            return f"{str(self.model)}+{str(self.lora)}"
        return str(self.model)


def dtype_from_name(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None

    if name.startswith("torch."):
        name = name[len("torch.") :]

    if name == "bfloat16":
        return torch.bfloat16
    elif name == "float16":
        return torch.float16
    elif name == "float32":
        return torch.float32
    elif name == "int64":
        return torch.int64
    raise RuntimeError(f'Unimplemented dtype "{name}"')


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


ARCH_NAME_TO_AUTO_CLS = {}

try:
    import transformers.models.auto.modeling_auto as tf_auto
except ImportError:
    tf_auto = None

if tf_auto is not None:
    for map_name, cls_name in [
        ("MODEL_MAPPING_NAMES", "AutoModel"),
        (
            "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES",
            "AutoModelForAudioClassification",
        ),
        (
            "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES",
            "AutoModelForImageClassification",
        ),
        ("MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES", "AutoModelForSpeechSeq2Seq"),
        (
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES",
            "AutoModelForSequenceClassification",
        ),
        ("MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES", "AutoModelForSeq2SeqLM"),
        (
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES",
            "AutoModelForTokenClassification",
        ),
        ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", "AutoModelForImageTextToText"),
        ("MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES", "AutoModelForTextToWaveform"),
        ("MODEL_FOR_MASKED_LM_MAPPING_NAMES", "AutoModelForMaskedLM"),
        ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "AutoModelForCausalLM"),
    ]:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            logging.info(f"Could not find {cls_name} in transformers")
            continue
        if hasattr(tf_auto, map_name):
            name_to_arch_name = getattr(tf_auto, map_name)
            for arch_name in name_to_arch_name.values():
                ARCH_NAME_TO_AUTO_CLS[arch_name] = cls


class AutoClassProtocol(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> transformers.PreTrainedModel: ...

    def from_config(
        self,
        config: transformers.PretrainedConfig,
        *model_args,
        **kwargs,
    ) -> transformers.PreTrainedModel: ...


def get_auto_cls(arch_name: str) -> AutoClassProtocol:
    """Get the AutoModel class for a given architecture name."""
    if arch_name in ARCH_NAME_TO_AUTO_CLS:
        return ARCH_NAME_TO_AUTO_CLS[arch_name]

    if arch_name.endswith("ForMaskedLM"):
        auto_cls = transformers.AutoModelForMaskedLM
    elif arch_name.endswith("ForSequenceClassification"):
        auto_cls = transformers.AutoModelForSequenceClassification
    elif arch_name.endswith("ForTokenClassification"):
        auto_cls = transformers.AutoModelForTokenClassification
    else:
        if not arch_name.endswith("ForCausalLM") or arch_name.endswith("LMHeadModel"):
            logging.warning(
                f"Unknown model type {arch_name} - assuming AutoModelForCausalLM"
            )
        auto_cls = transformers.AutoModelForCausalLM
    return auto_cls


def get_torch_accelerator_module(accelerator_name: Optional[str] = None):
    if accelerator_name is not None:
        accelerator_type = torch.device(accelerator_name).type
        return getattr(torch, accelerator_type)
    else:
        return (
            getattr(torch, torch.accelerator.current_accelerator().type)
            if hasattr(torch, "accelerator")
            else torch.cuda
        )


def get_torch_accelerator_count(accelerator_name: Optional[str] = None):
    torch_accelerator_module = torch.cuda
    if accelerator_name is not None:
        accelerator = torch.device(accelerator_name)
        # if user passes the device index in `accelerator_name`, then 1
        if accelerator.index != None:
            return 1
        torch_accelerator_module = getattr(torch, accelerator.type)
    else:
        torch_accelerator_module = (
            getattr(torch, torch.accelerator.current_accelerator().type)
            if hasattr(torch, "accelerator")
            else torch.cuda
        )
    return torch_accelerator_module.device_count()


def get_torch_accelerator_type(accelerator_name: Optional[str] = None):
    if accelerator_name is not None:
        return torch.device(accelerator_name).type
    else:
        return (
            torch.accelerator.current_accelerator().type
            if hasattr(torch, "accelerator")
            else "cuda"
        )
