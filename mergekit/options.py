# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import functools
import logging
import typing
from typing import Any, Callable, Optional, Union

import click
import torch
import transformers
from click.core import Context, HelpFormatter, Parameter
from pydantic import BaseModel, model_validator

from mergekit.common import parse_kmb


class MergeOptions(BaseModel, frozen=True):
    allow_crimes: bool = False
    transformers_cache: Optional[str] = None
    lora_merge_cache: Optional[str] = None
    lora_merge_dtype: Optional[str] = None
    cuda: bool = False
    low_cpu_memory: bool = False
    out_shard_size: int = parse_kmb("5B")
    copy_tokenizer: bool = True
    clone_tensors: bool = False
    trust_remote_code: bool = False
    random_seed: Optional[int] = None
    lazy_unpickle: bool = False
    write_model_card: bool = True
    safe_serialization: bool = True
    verbosity: int = 0
    quiet: bool = False
    read_to_gpu: bool = False
    multi_gpu: bool = False
    num_threads: Optional[int] = None
    gpu_rich: bool = False

    def apply_global_options(self):
        if self.verbosity > 1:
            log_level = logging.DEBUG
        elif self.verbosity == 1:
            log_level = logging.INFO
        else:
            log_level = logging.WARNING
        logging.basicConfig(level=log_level)
        if self.verbosity > 5:
            logging.debug("whoah buddy that's a lot of verbosity, two is plenty")
        if self.random_seed is not None:
            transformers.trainer_utils.set_seed(self.random_seed)
        if self.num_threads is not None:
            torch.set_num_threads(self.num_threads)
            torch.set_num_interop_threads(self.num_threads)

    @model_validator(mode="before")
    def handle_gpu_rich(cls, value):
        if isinstance(value, dict) and value.get("gpu_rich"):
            value["cuda"] = True
            value["low_cpu_memory"] = True
            value["read_to_gpu"] = True
            value["multi_gpu"] = True
        return value


OPTION_HELP = {
    "allow_crimes": "Allow mixing architectures",
    "transformers_cache": "Override storage path for downloaded models",
    "lora_merge_cache": "Path to store merged LORA models",
    "lora_merge_dtype": "Override dtype when applying LoRAs",
    "cuda": "Perform matrix arithmetic on GPU",
    "low_cpu_memory": "Store results and intermediate values on GPU. Useful if VRAM > RAM",
    "out_shard_size": "Number of parameters per output shard  [default: 5B]",
    "copy_tokenizer": "Copy a tokenizer to the output",
    "clone_tensors": "Clone tensors before saving, to allow multiple occurrences of the same layer",
    "trust_remote_code": "Trust remote code from huggingface repos (danger)",
    "random_seed": "Seed for reproducible use of randomized merge methods",
    "lazy_unpickle": "Experimental lazy unpickler for lower memory usage",
    "write_model_card": "Output README.md containing details of the merge",
    "safe_serialization": "Save output in safetensors. Do this, don't poison the world with more pickled models.",
    "quiet": "Suppress progress bars and other non-essential output",
    "read_to_gpu": "Read model weights directly to GPU",
    "multi_gpu": "Use multi-gpu parallel graph execution engine",
    "num_threads": "Number of threads to use for parallel CPU operations",
    "verbosity": "Verbose logging (repeat for more verbosity)",
    "gpu_rich": "Alias for --cuda --low-cpu-memory --read-to-gpu --multi-gpu",
}

OPTION_CATEGORIES = {
    "lora_merge_cache": "Storage",
    "transformers_cache": "Storage",
    "out_shard_size": "Output Settings",
    "copy_tokenizer": "Output Settings",
    "clone_tensors": "Output Settings",
    "write_model_card": "Output Settings",
    "safe_serialization": "Output Settings",
    "lazy_unpickle": "Performance",
    "cuda": "Performance",
    "low_cpu_memory": "Performance",
    "read_to_gpu": "Performance",
    "multi_gpu": "Performance",
    "num_threads": "Performance",
    "gpu_rich": "Performance",
    "trust_remote_code": "Dangerous Options",
    "allow_crimes": "Dangerous Options",
    "random_seed": "Miscellaneous",
    "verbosity": "Miscellaneous",
    "quiet": "Miscellaneous",
    "lora_merge_dtype": "Miscellaneous",
}


class ShardSizeParamType(click.ParamType):
    name = "size"

    def convert(
        self, value: Any, param: Optional[Parameter], ctx: Optional[Context]
    ) -> int:
        return parse_kmb(value)


def add_merge_options(f: Callable) -> Callable:
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        arg_dict = {}
        for field_name in MergeOptions.model_fields:
            if field_name in kwargs:
                arg_dict[field_name] = kwargs.pop(field_name)

        kwargs["merge_options"] = MergeOptions(**arg_dict)
        f(*args, **kwargs)

    for field_name, info in reversed(MergeOptions.model_fields.items()):
        origin = typing.get_origin(info.annotation)
        if origin is Union:
            ty, prob_none = typing.get_args(info.annotation)
            assert prob_none is type(None)
            field_type = ty
        else:
            field_type = info.annotation

        if field_name == "out_shard_size":
            field_type = ShardSizeParamType()

        arg_name = field_name.replace("_", "-")
        if field_type == bool:
            arg_str = f"--{arg_name}/--no-{arg_name}"
        else:
            arg_str = f"--{arg_name}"
        param_decls = [arg_str]
        kwargs = {}
        if field_name == "verbosity":
            param_decls = ["-v", "verbosity"]
            kwargs["count"] = True
        if field_name == "num_threads":
            param_decls = ["--num-threads", "-j"]
        if field_name == "gpu_rich":
            param_decls = ["--gpu-rich"]
            kwargs["is_flag"] = True

        help_str = OPTION_HELP.get(field_name, None)
        wrapper = click.option(
            *param_decls,
            type=field_type,
            default=info.default,
            help=help_str,
            show_default=field_name not in ("out_shard_size", "verbosity"),
            **kwargs,
        )(wrapper)

    return wrapper


class PrettyPrintHelp(click.Command):
    def format_options(self, ctx: Context, formatter: HelpFormatter) -> None:
        categories: dict[str, list[Parameter]] = {None: []}
        for param in ctx.command.params:
            if param.name in OPTION_CATEGORIES:
                category = OPTION_CATEGORIES[param.name]
                if category not in categories:
                    categories[category] = []
                categories[category].append(param)
            else:
                categories[None].append(param)

        for category, params in categories.items():
            title = category or "Script Options"
            opts = [p.get_help_record(ctx) for p in params]
            opts = [opt for opt in opts if opt is not None]
            if opts:
                with formatter.section(title):
                    formatter.write_dl(opts)
