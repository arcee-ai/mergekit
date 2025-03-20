# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import os
import sys
from typing import List

import click
import transformers
import yaml

from mergekit.merge import MergeOptions
from mergekit.moe import ALL_OUTPUT_ARCHITECTURES, MoEOutputArchitecture
from mergekit.moe.config import MoEMergeConfig, is_bad_config
from mergekit.moe.router import get_gate_params, warn_degenerate_gates
from mergekit.options import PrettyPrintHelp, add_merge_options


def build(
    config: MoEMergeConfig,
    out_path: str,
    merge_options: MergeOptions,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device: str = "auto",
    allow_all_same: bool = False,
    verbose: bool = False,
):
    if is_bad_config(config, allow_all_same=allow_all_same):
        sys.exit(1)

    base_model = config.base_model
    out_arch = select_output_arch(config, merge_options, verbose=verbose)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model.model.path, revision=base_model.model.revision
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Getting gate parameters...")
    need_gates = list(config.experts)
    if config.shared_experts:
        has_prompts = any(e.positive_prompts for e in config.shared_experts)
        assert all(
            bool(e.positive_prompts) == has_prompts for e in config.shared_experts
        ), "Must specify prompts for all shared experts or none, not a mix"
        if has_prompts or config.gate_mode in ("random", "uniform_random"):
            need_gates.extend(config.shared_experts)

    gate_vecs = get_gate_params(
        base_model,
        tokenizer,
        need_gates,
        mode=config.gate_mode,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        lazy_unpickle=merge_options.lazy_unpickle,
        trust_remote_code=merge_options.trust_remote_code,
        device=device,
    )
    # gate_vecs: (num_layers, num_experts, hidden_size)
    router_weights = gate_vecs[:, : len(config.experts), :]
    shared_router_weights = gate_vecs[:, len(config.experts) :, :]
    warn_degenerate_gates(gate_vecs)

    out_arch.write_model(
        out_path,
        config,
        merge_options,
        router_weights=[router_weights[i, ...] for i in range(router_weights.shape[0])],
        shared_router_weights=[
            shared_router_weights[i, ...] for i in range(router_weights.shape[0])
        ],
    )

    if merge_options.copy_tokenizer:
        logging.info("Saving tokenizer...")
        tokenizer.save_pretrained(out_path, safe_serialization=True)

    logging.info("Done.")


def select_output_arch(
    config: MoEMergeConfig,
    merge_options: MergeOptions,
    verbose: bool = False,
) -> MoEOutputArchitecture:
    candidates_in = ALL_OUTPUT_ARCHITECTURES
    if config.architecture:
        candidates_in = [
            a
            for a in candidates_in
            if a.name().lower().startswith(config.architecture.lower())
        ]
    if not candidates_in:
        logging.error(
            f"No output architecture found that matches the given architecture: {config.architecture}"
        )
        logging.error("All supported output architectures:")
        for arch in ALL_OUTPUT_ARCHITECTURES:
            logging.error(f"  * {arch.name()}")
        sys.exit(1)

    candidates: List[MoEOutputArchitecture] = []
    for arch in candidates_in:
        if arch.supports_config(
            config, explain=verbose, trust_remote_code=merge_options.trust_remote_code
        ):
            candidates.append(arch)
        else:
            logging.info(f"Output architecture {arch.name()} does not support config.")

    if not candidates:
        logging.error(
            "No output architecture found that is compatible with the given models."
        )

        logging.error("All supported output architectures:")
        for arch in ALL_OUTPUT_ARCHITECTURES:
            logging.error(f"  * {arch.name()}")
        sys.exit(1)

    # for compatibility with older configs, default to Mixtral if available
    for arch in candidates:
        if arch.name() == "Mixtral":
            return arch

    if len(candidates) > 1:
        logging.warning(
            "Multiple output architectures found that are compatible with the given models."
        )
        logging.warning(f"Defaulting to {candidates[0].name()}")
    else:
        logging.info(f"Selected output architecture: {candidates[0].name()}")
    return candidates[0]


@click.command("mergekit-moe", cls=PrettyPrintHelp)
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path())
@click.option(
    "--load-in-4bit",
    is_flag=True,
    type=bool,
    default=False,
    help="Load model in 4bit for computing hidden states",
)
@click.option(
    "--load-in-8bit",
    is_flag=True,
    type=bool,
    default=False,
    help="Load model in 8bit for computing hidden states",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to use to compute embeddings",
    show_default=True,
)
@click.option(
    "--i-understand-this-is-not-useful-without-training",
    type=bool,
    default=False,
    is_flag=True,
    help="Really make the questionable model you want.",
)
@add_merge_options
def main(
    config_path: str,
    out_path: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    device: str,
    i_understand_this_is_not_useful_without_training: bool,
    merge_options: MergeOptions,
):
    """Create a Mixture of Experts model by combining the pretrained weights of multiple models."""
    merge_options.apply_global_options()

    if merge_options.cuda:
        logging.warning(
            '--cuda is a no-op for mergekit-moe, use "--device cuda" instead'
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config_source = file.read()

    config = MoEMergeConfig.model_validate(yaml.safe_load(config_source))
    build(
        config,
        out_path=out_path,
        merge_options=merge_options,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device=device,
        allow_all_same=i_understand_this_is_not_useful_without_training,
        verbose=merge_options.verbosity > 0,
    )

    if merge_options.write_model_card:
        # TODO: generate a README.md as well
        with open(
            os.path.join(out_path, "mergekit_moe_config.yml"), "w", encoding="utf-8"
        ) as fp:
            fp.write(config_source)


if __name__ == "__main__":
    main()
