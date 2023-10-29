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
from typing import Optional

import typer
import yaml
from typing_extensions import Annotated

from mergekit.common import parse_kmb
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge


def main(
    config_file: Annotated[str, typer.Argument(help="YAML configuration file")],
    out_path: Annotated[str, typer.Argument(help="Path to write result model")],
    lora_merge_cache: Annotated[
        Optional[str],
        typer.Option(help="Path to store merged LORA models", metavar="PATH"),
    ] = None,
    transformers_cache: Annotated[
        Optional[str],
        typer.Option(
            help="Override storage path for downloaded models", metavar="PATH"
        ),
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
    run_merge(
        merge_config,
        out_path,
        options=MergeOptions(
            lora_merge_cache=lora_merge_cache,
            transformers_cache=transformers_cache,
            cuda=cuda,
            low_cpu_memory=low_cpu_memory,
            copy_tokenizer=copy_tokenizer,
            allow_crimes=allow_crimes,
            out_shard_size=out_shard_size,
        ),
    )


def _main():
    # just a wee li'l stub for setuptools
    typer.run(main)


if __name__ == "__main__":
    _main()
