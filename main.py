#!/usr/bin/env python3
# Charles O. Goddard
# 8/20/230
"""Merge a set of task-specific models into a base model using the methodology
of \"Resolving Interference When Merging Models\" (https://arxiv.org/abs/2306.01708)."""

import json
import logging
from typing import Dict, List, Optional

import transformers
import typer
from typing_extensions import Annotated

from common import LLAMA_LAYER_MEMBERS, ModelReference, gradient_weights
from merge_methods import MergeMethod
from merger import MergeConfig, ModelMerger


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
    layer_gradient: Annotated[
        Optional[str],
        typer.Option(help="List of weight values to interpolate between per-layer"),
    ] = None,
    method: Annotated[
        MergeMethod, typer.Option(help="Method used to merge models")
    ] = MergeMethod.ties,
    base_model: Annotated[
        Optional[str], typer.Option(help="Base model for merge")
    ] = None,
    normalize: Annotated[
        bool,
        typer.Option(
            help="Divide merged parameters by the sum of weights",
        ),
    ] = True,
    cache_dir: Annotated[
        Optional[str], typer.Option(help="Override storage path for downloaded models")
    ] = None,
    merged_cache_dir: Annotated[
        Optional[str], typer.Option(help="Storage path for merged LoRA models")
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
):
    """Merge a set of models with a shared base model by resolving sign differences."""
    if merged_cache_dir is None:
        merged_cache_dir = cache_dir

    models = [ModelReference.parse(m) for m in merge]

    merge_options = {
        "normalize": normalize,
    }

    if method == MergeMethod.ties:
        if density:
            if len(density) == 1:
                density = [density[0]] * len(models)
            merge_options["density"] = dict(zip(models, density))

        merge_options.update(
            {
                "int8_mask": int8_mask,
                "consensus_method": "count" if naive_count else "sum",
            }
        )

    if method == MergeMethod.linear:
        if (not weight) and (not layer_gradient):
            raise RuntimeError("Must specify weight for linear merge")

        if base_model:
            logging.warning(
                "Linear merge mode does not use base model - will not be included"
            )

    if method == MergeMethod.slerp:
        if len(weight) > 1:
            raise RuntimeError("Too many weights for slerp")

        if not layer_gradient and len(weight) != 1:
            raise RuntimeError(
                "Slerp merge needs either exactly one weight or gradient"
            )

        merge_options["t"] = weight[0] if weight else 0

        if not base_model:
            base_model = str(models[0])
    else:
        if weight:
            if len(weight) == 1:
                weight = [weight[0]] * len(models)
            elif len(weight) != len(models):
                raise RuntimeError("Must specify one weight per model")
            merge_options["weight"] = dict(zip(models, weight))

    if base_model:
        base_model = ModelReference.parse(base_model)
        merge_options["base_model"] = base_model

    if base_model and base_model not in models:
        models = [base_model] + models

    if method == MergeMethod.slerp and len(models) != 2:
        raise RuntimeError("Slerp expects exactly two models")

    overrides: Dict[str, Dict] = {
        "model.embed_tokens.weight": {"dtype": "float32"},
        "lm_head.weight": {"dtype": "float32"},
    }

    if layer_gradient:
        if len(models) != 2:
            raise RuntimeError("Gradient can only be used with exactly two models")

        layer_gradient = json.loads(layer_gradient)
        num_layers = transformers.AutoConfig.from_pretrained(
            models[0].path
        ).num_hidden_layers

        def override_weight(name: str, weight: float):
            if name not in overrides:
                overrides[name] = {}

            if method == MergeMethod.slerp:
                overrides[name]["t"] = weight
            else:
                overrides[name]["weight"] = {
                    models[0]: (1 - weight),
                    models[1]: weight,
                }

        layer_weights: List[float] = gradient_weights(layer_gradient, num_layers + 2)
        embedding_weight = layer_weights.pop(0)
        lm_head_weight = layer_weights.pop(-1)

        override_weight("model.embed_tokens.weight", embedding_weight)
        for layer_idx in range(num_layers):
            for name in LLAMA_LAYER_MEMBERS:
                override_weight(
                    f"model.layers.{layer_idx}.{name}.weight", layer_weights[layer_idx]
                )
        override_weight("model.norm.weight", lm_head_weight)
        override_weight("model.lm_head.weight", lm_head_weight)

    config = MergeConfig(
        models=models,
        out_path=out_path,
        cuda=cuda,
        dtype="bfloat16" if bf16 else None,
        merge_method=method,
        merge_cache=merged_cache_dir,
        transformers_cache=cache_dir,
        options=merge_options,
        overrides=overrides,
    )
    merger = ModelMerger(config)
    merger.run()

    cfg_donor = base_model if base_model else models[0]

    try:
        cfg = transformers.AutoConfig.from_pretrained(cfg_donor.path)
        cfg.save_pretrained(out_path)
    except Exception as e:
        logging.warning("Failed to copy config from base model", exc_info=e)
        logging.warning(
            "The merge was still successful. "
            "Just copy config.json from one of the models, it's fine."
        )

    if copy_tokenizer:
        tok = transformers.AutoTokenizer.from_pretrained(cfg_donor.path)
        tok.save_pretrained(out_path)

    logging.info("Merge complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(main)
