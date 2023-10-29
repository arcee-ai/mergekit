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

from typing import List, Optional

import typer
import yaml
from typing_extensions import Annotated

from mergekit.config import InputModelDefinition, MergeConfiguration
from mergekit.merge import MergeOptions, run_merge


def main(
    out_path: Annotated[str, typer.Argument(help="Output directory for final model")],
    merge: Annotated[
        List[str], typer.Option(help="Add a model to the merge", metavar="MODEL")
    ],
    density: Annotated[
        List[float],
        typer.Option(
            help="Fraction of weights to keep for each model (ties only)",
            default_factory=list,
            show_default=False,
        ),
    ],
    weight: Annotated[
        List[float],
        typer.Option(
            help="Weighting for a model (default 1.0 for all models if not specified)",
            default_factory=list,
            show_default=False,
        ),
    ],
    method: Annotated[str, typer.Option(help="Method used to merge models")] = "ties",
    base_model: Annotated[
        Optional[str], typer.Option(help="Base model for merge")
    ] = None,
    normalize: Annotated[
        bool,
        typer.Option(
            help="Divide merged parameters by the sum of weights",
        ),
    ] = True,
    merged_cache_dir: Annotated[
        Optional[str], typer.Option(help="Storage path for merged LoRA models")
    ] = None,
    cache_dir: Annotated[
        Optional[str], typer.Option(help="Override storage path for downloaded models")
    ] = None,
    cuda: bool = False,
    int8_mask: Annotated[
        bool, typer.Option(help="Store intermediate masks in int8 to save memory")
    ] = False,
    bf16: Annotated[bool, typer.Option(help="Use bfloat16")] = True,
    naive_count: Annotated[
        bool, typer.Option(help="Use naive sign count instead of weight (ties only)")
    ] = False,
    copy_tokenizer: Annotated[
        bool, typer.Option(help="Copy base model tokenizer into output")
    ] = True,
    print_yaml: Annotated[
        bool, typer.Option(help="Print generated YAML configuration")
    ] = False,
):
    """Wrapper for using a subset of legacy-style script arguments."""
    models = [InputModelDefinition(model=model, parameters={}) for model in merge]
    if base_model and base_model not in merge:
        models.append(InputModelDefinition(model=base_model, parameters={}))

    parameters = {}

    if density:
        if len(density) == 1:
            density = [density[0]] * len(models)
        for idx, d in enumerate(density):
            models[idx].parameters["density"] = d

    if method == "slerp":
        assert len(weight) == 1, "Must specify exactly one weight for SLERP"
        parameters["t"] = weight[0]
    else:
        if weight:
            if len(weight) == 1:
                weight = [weight[0]] * len(models)
            for idx, w in enumerate(weight):
                models[idx].parameters["weight"] = w

    if int8_mask:
        parameters["int8_mask"] = True
    if naive_count:
        parameters["consensus_method"] = "count"
    parameters["normalize"] = normalize

    merge_config = MergeConfiguration(
        merge_method=method,
        models=models,
        parameters=parameters,
        base_model=base_model,
        dtype="bfloat16" if bf16 else None,
    )

    if print_yaml:
        print(yaml.dump(merge_config.model_dump(mode="json", exclude_none=True)))

    run_merge(
        merge_config,
        out_path,
        options=MergeOptions(
            lora_merge_cache=merged_cache_dir,
            transformers_cache=cache_dir,
            cuda=cuda,
            copy_tokenizer=copy_tokenizer,
        ),
    )


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
