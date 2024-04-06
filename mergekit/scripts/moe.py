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

import logging
import os
import sys

import click
import transformers
import yaml

from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference
from mergekit.merge import MergeOptions
from mergekit.moe.arch import ALL_OUTPUT_ARCHITECTURES, MoEOutputArchitecture
from mergekit.moe.config import MoEMergeConfig, is_bad_config
from mergekit.moe.router import get_gate_params, warn_degenerate_gates
from mergekit.options import add_merge_options


def build(
    config: MoEMergeConfig,
    out_path: str,
    merge_options: MergeOptions,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device: str = "auto",
    allow_all_same: bool = False,
):
    if is_bad_config(config, allow_all_same=allow_all_same):
        sys.exit(1)

    base_model = ModelReference.parse(config.base_model)
    base_cfg = base_model.config(trust_remote_code=merge_options.trust_remote_code)
    out_arch = select_output_arch(config, merge_options, base_cfg)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model.model.path, revision=base_model.model.revision
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Getting gate parameters...")
    gate_vecs = get_gate_params(
        base_model,
        tokenizer,
        config.experts,
        mode=config.gate_mode,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        lazy_unpickle=merge_options.lazy_unpickle,
        trust_remote_code=merge_options.trust_remote_code,
        device=device,
    )
    # gate_vecs: (num_layers, num_experts, hidden_size)
    warn_degenerate_gates(gate_vecs)

    out_arch.write_model(out_path, config, merge_options, out_dtype=gate_vecs.dtype)

    if merge_options.copy_tokenizer:
        logging.info("Saving tokenizer...")
        tokenizer.save_pretrained(out_path, safe_serialization=True)

    logging.info("Done.")


def select_output_arch(
    config: MoEMergeConfig,
    merge_options: MergeOptions,
    base_cfg: transformers.PretrainedConfig,
) -> MoEOutputArchitecture:
    expert_cfgs = [
        ModelReference.parse(e.model_ref).config(
            trust_remote_code=merge_options.trust_remote_code
        )
        for e in config.experts
    ]

    out_arch = None
    for arch in ALL_OUTPUT_ARCHITECTURES:
        if arch.arch_is_compatible(get_architecture_info(base_cfg)) and all(
            arch.arch_is_compatible(get_architecture_info(e)) for e in expert_cfgs
        ):
            out_arch = arch
            break
    if out_arch is None:
        logging.error(
            "No output architecture found that is compatible with the given models."
        )
        logging.error(f"Base architecture: {base_cfg.model_type}")
        logging.error(
            f"Expert donor architectures: {', '.join(e.model_type for e in expert_cfgs)}"
        )

        logging.error("Supported output architectures:")
        for arch in ALL_OUTPUT_ARCHITECTURES:
            logging.error(f"  * {arch.name()}")
        sys.exit(1)
    return out_arch


@click.command("mergekit-moe")
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
    "--verbose", "-v", type=bool, default=False, is_flag=True, help="Verbose logging"
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
    merge_options: MergeOptions,
    verbose: bool,
    i_understand_this_is_not_useful_without_training: bool,
):
    """Create a Mixture of Experts model by combining the pretrained weights of multiple models."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

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
    )

    if merge_options.write_model_card:
        # TODO: generate a README.md as well
        with open(
            os.path.join(out_path, "mergekit_moe_config.yml"), "w", encoding="utf-8"
        ) as fp:
            fp.write(config_source)


if __name__ == "__main__":
    main()
