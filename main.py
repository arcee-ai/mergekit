import logging
from typing import Optional

import torch
import typer
import yaml
from typing_extensions import Annotated

import merge_methods
from architecture import get_architecture_info
from common import parse_kmb
from config import MergeConfiguration
from graph import Executor, RuleSet
from plan import plan


def main(
    config_file: Annotated[str, typer.Argument(help="YAML configuration file")],
    out_path: Annotated[str, typer.Argument(help="Path to write result model")],
    lora_merge_cache: Annotated[
        Optional[str],
        typer.Option(help="Path to store merged LORA models", metavar="PATH"),
    ] = None,
    cuda: Annotated[
        bool, typer.Option(help="Perform matrix arithmetic on GPU")
    ] = False,
    low_cpu_memory: Annotated[
        bool,
        typer.Option(
            help="Store results and intermediate values on GPU. Useful if VRAM > RAM"
        ),
    ] = False,
    copy_tokenizer: Annotated[
        bool, typer.Option(help="Copy a tokenizer to the output")
    ] = True,
    allow_crimes: Annotated[
        bool, typer.Option(help="Allow mixing architectures")
    ] = False,
    out_shard_size: Annotated[
        Optional[int],
        typer.Option(
            help="Number of parameters per output shard  [default: 5B]",
            parser=parse_kmb,
            show_default=False,
            metavar="NUM",
        ),
    ] = parse_kmb("5B"),
    verbose: Annotated[bool, typer.Option("-v")] = False,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    with open(config_file, "r", encoding="utf-8") as file:
        data = yaml.load(file, yaml.SafeLoader)

    merge_config: MergeConfiguration = MergeConfiguration.model_validate(data)

    dtype: Optional[torch.dtype] = {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[merge_config.dtype]

    if not merge_config.models and not merge_config.slices:
        raise RuntimeError("No output requested")

    method = merge_methods.get(merge_config.merge_method)
    model_arch_info = [
        get_architecture_info(m.config()) for m in merge_config.referenced_models()
    ]
    if not allow_crimes:
        if not all(a == model_arch_info[0] for a in model_arch_info[1:]):
            raise RuntimeError(
                "Must specify --allow-crimes to mix different architectures"
            )
    arch_info = model_arch_info[0]

    (targets, static_rules) = plan(merge_config, arch_info)

    rules = RuleSet(static_rules)
    exec = Executor(
        merge_config.referenced_models(),
        targets,
        rules,
        {"merge": method},
        cache_dir=lora_merge_cache,
        dtype=dtype,
        cuda=cuda,
        low_cpu_memory=low_cpu_memory,
    )
    exec.run(out_path, max_shard_size=out_shard_size)

    method.model_out_config(merge_config).save_pretrained(out_path)
    if copy_tokenizer:
        try:
            method.model_tokenizer(merge_config).save_pretrained(
                out_path, safe_serialization=True
            )
        except Exception as e:
            logging.error(
                "Failed to save tokenizer. The merge was still successful, just copy it from somewhere else.",
                e,
            )


if __name__ == "__main__":
    typer.run(main)
