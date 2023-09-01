#!/usr/bin/env python3
# Charles O. Goddard
# 8/20/230
"""Merge a set of task-specific models into a base model using the methodology
of \"Resolving Interference When Merging Models\" (https://arxiv.org/abs/2306.01708)."""

import logging
from typing import List, Optional

import transformers
import typer
from typing_extensions import Annotated

from common import ModelReference
from merge_methods import TiesMergeOptions
from merger import MergeConfig, ModelMerger


def main(
    base_model: Annotated[str, typer.Argument(help="Base model for merge")],
    out_path: Annotated[str, typer.Argument(help="Output directory for final model")],
    merge: Annotated[
        List[str], typer.Option(help="Add a model to the merge", metavar="MODEL")
    ],
    density: Annotated[
        List[float],
        typer.Option(
            help="Fraction of weights to keep for each model (default 0.33)",
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
        bool, typer.Option(help="Use naive sign count instead of weight")
    ] = False,
    copy_tokenizer: Annotated[
        bool, typer.Option(help="Copy base model tokenizer into output")
    ] = True,
):
    """Merge a set of models with a shared base model by resolving sign differences."""
    if merged_cache_dir is None:
        merged_cache_dir = cache_dir

    if not density:
        density = [0.33] * len(merge)
    elif len(density) == 1:
        density = [density[0]] * len(merge)
    elif len(density) != len(merge):
        raise RuntimeError(
            "Must specify either one single density or exactly one per model"
        )

    if not weight:
        weight = None
    elif len(weight) != len(merge):
        raise RuntimeError("Must specify one weight per merged model")

    models = [ModelReference.parse(m) for m in ([base_model] + merge)]
    ties_options = TiesMergeOptions(
        base_model=models[0],
        density=dict(zip(models[1:], density)),
        weight=None if not weight else dict(zip(models[1:], weight)),
        int8_mask=int8_mask,
        dtype="bfloat16" if bf16 else None,
        consensus_method="count" if naive_count else "sum",
        normalize=normalize,
    )
    config = MergeConfig(
        models=models,
        out_path=out_path,
        cuda=cuda,
        dtype="bfloat16" if bf16 else None,
        merge_method="ties",
        merge_cache=merged_cache_dir,
        model_cache=cache_dir,
        options=ties_options.dict(exclude_none=True),
        overrides={
            "model.embed_tokens.weight": {"dtype": "float32"},
            "lm_head.weight": {"dtype": "float32"},
        },
    )
    merger = ModelMerger(config)
    merger.run()

    try:
        cfg = transformers.AutoConfig.from_pretrained(base_model.path)
        cfg.save_pretrained(out_path)
    except Exception as e:
        logging.warning("Failed to copy config from base model", exc_info=e)
        logging.warning(
            "The merge was still successful. "
            "Just copy config.json from one of the models, it's fine."
        )

    if copy_tokenizer:
        tok = transformers.AutoTokenizer.from_pretrained(base_model.path)
        tok.save_pretrained(out_path)

    logging.info("Merge complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(main)
