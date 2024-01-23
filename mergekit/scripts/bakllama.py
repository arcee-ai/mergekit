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
