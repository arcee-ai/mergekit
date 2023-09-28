import logging
from typing import Optional

import torch
import typer
import yaml
from typing_extensions import Annotated

import common
import merge_methods
from config import MergeConfiguration
from common import LLAMA_INFO, ModelReference
from graph import Executor, RuleSet
from plan import plan


def main(
    config_file: Annotated[str, typer.Argument(help="YAML configuration file")],
    out_path: Annotated[str, typer.Argument(help="Path to write result model")],
    lora_merge_cache: Annotated[
        Optional[str], typer.Option(help="Path to store merged LORA models")
    ] = None,
    cuda: Annotated[
        bool, typer.Option(help="Perform matrix arithmetic on GPU")
    ] = False,
    gpu_shard_buffer: Annotated[
        bool,
        typer.Option(
            help="Store results on GPU until shard is written. Useful if VRAM > RAM"
        ),
    ] = False,
    copy_tokenizer: Annotated[
        bool, typer.Option("Copy a tokenizer to the output")
    ] = True,
):
    with open(config_file, "r", encoding="utf-8") as file:
        data = yaml.load(file, yaml.SafeLoader)

    config = MergeConfiguration.parse_obj(data)
    (targets, static_rules) = plan(config, LLAMA_INFO)

    dtype: Optional[torch.dtype] = {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.dtype]

    # if models to merge are specified instead of output slices, compute them
    if config.models:
        if config.slices:
            raise RuntimeError("Must specify either models to merge or output slices")
        
        config.slices = []
        for model in config.models:
            model_cfg = ModelReference.parse(model.model)


    if not config.slices:
        raise RuntimeError("No output requested")

    method = merge_methods.get(config.merge_method)

    rules = RuleSet(static_rules)
    exec = Executor(
        config.referenced_models(),
        targets,
        rules,
        {"merge": method},
        cache_dir=lora_merge_cache,
        dtype=dtype,
        cuda=cuda,
        gpu_shard_buffer=gpu_shard_buffer,
    )
    exec.run(out_path)

    method.model_out_config(config).save_pretrained(out_path)
    if copy_tokenizer:
        try:
            method.model_tokenizer(config).save_pretrained(
                out_path, safe_serialization=True
            )
        except Exception as e:
            logging.error(
                "Failed to save tokenizer. The merge was still successful, just copy it from somewhere else.",
                e,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(main)
