from typing import Optional

import torch
import typer
import yaml
from typing_extensions import Annotated

import common
import merge_methods
from config import MergeConfiguration
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
):
    with open(config_file, "r", encoding="utf-8") as file:
        data = yaml.load(file, yaml.SafeLoader)

    config = MergeConfiguration.parse_obj(data)
    (targets, static_rules) = plan(config, common.LLAMA_INFO)

    dtype: Optional[torch.dtype] = {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.dtype]

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


if __name__ == "__main__":
    typer.run(main)
