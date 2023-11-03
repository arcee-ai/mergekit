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

import random
from typing import List, Optional

import typer
import yaml
from typing_extensions import Annotated

from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference
from mergekit.config import (
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from mergekit.merge import MergeOptions, run_merge


def main(
    out_path: Annotated[
        str, typer.Argument(help="Output path for merged model", metavar="PATH")
    ],
    model: Annotated[
        List[str], typer.Option(help="Add a model to the merge", metavar="MODEL")
    ],
    weight: Annotated[
        List[float],
        typer.Option(
            help="Weighting for a model",
            default_factory=list,
            show_default=False,
        ),
    ],
    print_yaml: Annotated[
        bool, typer.Option(help="Print YAML merge config for resulting model")
    ] = False,
    write_yaml: Annotated[
        Optional[str], typer.Option(help="Path to write YAML merge config to")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Generate a config but do not run the merge")
    ] = False,
    fp16: bool = False,
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
    copy_tokenizer: Annotated[
        bool, typer.Option(help="Copy a tokenizer to the output")
    ] = True,
    full_random: Annotated[
        bool, typer.Option(help="Randomize layer index as well as source model")
    ] = False,
):
    models = [ModelReference.parse(m) for m in model]

    m0_cfg = models[0].config()
    arch_info = get_architecture_info(m0_cfg)
    total_num_layers = arch_info.num_layers(m0_cfg)

    out_slices: List[OutputSliceDefinition] = []

    if full_random:
        for model, frac in zip(models, weight):
            cfg = model.config()
            num_layers = int(arch_info.num_layers(cfg) * frac)
            for _ in range(num_layers):
                src_idx = random.randrange(0, num_layers)
                out_slices.append(
                    OutputSliceDefinition(
                        sources=[
                            InputSliceDefinition(
                                model=str(model),
                                layer_range=(src_idx, src_idx + 1),
                            )
                        ]
                    )
                )
        random.shuffle(out_slices)
    else:
        for layer_idx in range(total_num_layers):
            src_model = random.choices(models, weights=weight, k=1)[0]
            if out_slices and out_slices[-1].sources[0].model == str(src_model):
                out_slices[-1].sources[0].layer_range = (
                    out_slices[-1].sources[0].layer_range[0],
                    layer_idx + 1,
                )
            else:
                out_slices.append(
                    OutputSliceDefinition(
                        sources=[
                            InputSliceDefinition(
                                model=str(src_model),
                                layer_range=(layer_idx, layer_idx + 1),
                            )
                        ]
                    )
                )
    merge_config = MergeConfiguration(
        merge_method="passthrough", slices=out_slices, dtype="float16" if fp16 else None
    )

    if print_yaml or write_yaml:
        yaml_str = yaml.dump(merge_config.model_dump(exclude_none=True, mode="json"))

        if print_yaml:
            print(yaml_str)
        if write_yaml:
            with open(write_yaml, "w", encoding="utf-8") as file:
                file.write(yaml_str)

    if dry_run:
        return

    run_merge(
        merge_config,
        out_path,
        MergeOptions(
            lora_merge_cache=lora_merge_cache,
            transformers_cache=transformers_cache,
            copy_tokenizer=copy_tokenizer,
        ),
    )


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
