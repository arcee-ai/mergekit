# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import Optional

import click
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, add_merge_options


@click.command("mergekit-yaml")
@click.argument("config_file")
@click.argument("out_path")
@click.option(
    "--verbose", "-v", type=bool, default=False, is_flag=True, help="Verbose logging"
)
@click.option(
    "--num-threads",
    type=int,
    help="Number of threads to use for parallel CPU operations",
    default=None,
)
@add_merge_options
def main(
    merge_options: MergeOptions,
    config_file: str,
    out_path: str,
    verbose: bool,
    num_threads: Optional[int],
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    with open(config_file, "r", encoding="utf-8") as file:
        config_source = file.read()

    merge_config: MergeConfiguration = MergeConfiguration.model_validate(
        yaml.safe_load(config_source)
    )
    run_merge(
        merge_config,
        out_path,
        options=merge_options,
        config_source=config_source,
    )


if __name__ == "__main__":
    main()
