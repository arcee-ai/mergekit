# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import List, Optional

import click
import yaml
from pydantic import BaseModel

from mergekit.common import MergeOptions
from mergekit.config import (
    ConditionalParameter,
    InputSliceDefinition,
    MergeConfiguration,
)
from mergekit.merge import run_merge


class LayerSlice(BaseModel):
    model: str
    start: int
    end: int
    scale: Optional[float] = None


class BakllamaConfig(BaseModel):
    layer_slices: List[LayerSlice]
    embedding_source: Optional[str] = None
    lm_head_source: Optional[str] = None


@click.command("bakllama")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=str)
@click.option(
    "--clone-tensors/--no-clone-tensors",
    type=bool,
    is_flag=True,
    help="Clone tensors before saving, to allow multiple occurrences of the same layer",
    default=False,
)
@click.option("--fp16/--no-fp16", type=bool, default=False)
def main(
    config_path: str,
    out_path: str,
    clone_tensors: bool,
    fp16: bool,
):
    """Wrapper for using legacy bakllama configuration files."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = BakllamaConfig.model_validate(yaml.safe_load(file))

    slices = []
    for s in config.layer_slices:
        parameters = {}
        if s.scale is not None:
            parameters["scale"] = ConditionalParameter(
                value=s.scale, filter="down_proj"
            )
        slices.append(
            InputSliceDefinition(
                model=s.model, layer_range=(s.start, s.end), parameters=parameters
            )
        )

    merge_config = MergeConfiguration(
        merge_method="passthrough", slices=slices, dtype="float16" if fp16 else None
    )
    run_merge(merge_config, out_path, MergeOptions(clone_tensors=clone_tensors))


if __name__ == "__main__":
    main()
