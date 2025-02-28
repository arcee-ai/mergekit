# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import List, Optional

import click
import yaml

from mergekit.config import InputModelDefinition, MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options


@click.command("mergekit-legacy", cls=PrettyPrintHelp)
@click.argument("out_path", type=str)
@click.option(
    "--merge", "merge", type=str, multiple=True, help="Add a model to the merge"
)
@click.option(
    "--density",
    "density",
    type=float,
    multiple=True,
    default=[],
    help="Fraction of weights to keep for each model (ties only)",
)
@click.option(
    "--weight",
    "weight",
    type=float,
    multiple=True,
    default=[],
    help="Weighting for a model (default 1.0 for all models if not specified)",
)
@click.option(
    "--method", "method", type=str, default="ties", help="Method used to merge models"
)
@click.option(
    "--base-model", "base_model", type=str, default=None, help="Base model for merge"
)
@click.option(
    "--normalize/--no-normalize",
    "normalize",
    is_flag=True,
    default=True,
    help="Divide merged parameters by the sum of weights",
)
@click.option(
    "--int8-mask/--no-int8-mask",
    "int8_mask",
    is_flag=True,
    help="Store intermediate masks in int8 to save memory",
)
@click.option("--bf16/--no-bf16", "bf16", is_flag=True, help="Use bfloat16")
@click.option(
    "--naive-count/--no-naive-count",
    "naive_count",
    is_flag=True,
    help="Use naive sign count instead of weight (ties only)",
)
@click.option(
    "--print-yaml/--no-print-yaml",
    "print_yaml",
    is_flag=True,
    help="Print generated YAML configuration",
)
@add_merge_options
def main(
    out_path: str,
    merge: List[str],
    density: List[float],
    weight: List[float],
    method: str,
    base_model: Optional[str],
    normalize: bool,
    int8_mask: bool,
    bf16: bool,
    naive_count: bool,
    print_yaml: bool,
    merge_options: MergeOptions,
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
        options=merge_options,
    )


if __name__ == "__main__":
    main()
