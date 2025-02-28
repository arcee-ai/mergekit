# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1


import click
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options


@click.command("mergekit-yaml", cls=PrettyPrintHelp)
@click.argument("config_file")
@click.argument("out_path")
@add_merge_options
def main(
    merge_options: MergeOptions,
    config_file: str,
    out_path: str,
):
    merge_options.apply_global_options()

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
