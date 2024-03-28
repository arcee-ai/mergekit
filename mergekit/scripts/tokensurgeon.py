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
import sys
from typing import Dict, List, Tuple

import click
import torch
import transformers

from mergekit.architecture import (
    ConfiguredArchitectureInfo,
    WeightInfo,
    get_architecture_info,
)
from mergekit.common import ModelReference
from mergekit.io import TensorWriter
from mergekit.io.tasks import LoaderCache
from mergekit.options import MergeOptions, add_merge_options


@click.command("mergekit-tokensurgeon")
@click.argument("model", type=str)
@click.argument("donor", type=str)
@click.argument("out_path", type=str)
@add_merge_options
def main(model: str, donor: str, out_path: str, options: MergeOptions):
    model_ref = ModelReference.model_validate(model)
    donor_ref = ModelReference.model_validate(donor)
    replace_tokenizer(model_ref, donor_ref, out_path, options)


def replace_tokenizer(
    model: ModelReference, donor: ModelReference, out_path: str, options: MergeOptions
):
    cache = LoaderCache()
    cache.setup(options=options)

    arch_info = validate_architecture(model, donor, options)
    embed_info, lm_head_info = get_embedding_info(model, options)

    _, old_vocab = load_tokenizer(model, options)
    tokenizer, new_vocab = load_tokenizer(donor, options)
    common_tokens = list(set(old_vocab.keys()) & set(new_vocab.keys()))

    old_embed = cache.get(model).get_tensor(embed_info.name, aliases=embed_info.aliases)
    donor_embed = cache.get(donor).get_tensor(
        embed_info.name, aliases=embed_info.aliases
    )

    (_, hidden_size_0) = old_embed.shape
    (_, hidden_size_1) = donor_embed.shape
    if hidden_size_1 != hidden_size_0:
        report_issue(
            f"Embedding sizes do not match: {hidden_size_0} vs {hidden_size_1}",
            error=not options.allow_crimes,
        )

    min_overlap = max(hidden_size_0, hidden_size_1)
    if len(common_tokens) < min_overlap:
        report_issue(
            f"Common tokens ({len(common_tokens)}) less than embedding size ({min_overlap})",
            error=not options.allow_crimes,
        )

    logging.info(
        f"Computing new embeddings for {len(new_vocab) - len(common_tokens)} tokens"
    )
    new_embed = lstsq_embeddings(
        old_embed, donor_embed, old_vocab, new_vocab, common_tokens
    )
    new_lm_head = lstsq_embeddings(
        cache.get(model).get_tensor(lm_head_info.name, aliases=lm_head_info.aliases),
        cache.get(donor).get_tensor(lm_head_info.name, aliases=lm_head_info.aliases),
        old_vocab,
        new_vocab,
        common_tokens,
    )

    # Save out the new model
    logging.info(f"Saving new model to {out_path}")
    writer = TensorWriter(
        out_path,
        max_shard_size=options.out_shard_size,
        safe_serialization=options.safe_serialization,
    )
    for weight_info in arch_info.all_weights():
        if weight_info.name == embed_info.name:
            tensor = new_embed
        elif weight_info.name == lm_head_info.name:
            tensor = new_lm_head
        else:
            tensor = cache.get(model).get_tensor(
                weight_info.name, aliases=weight_info.aliases
            )
        writer.save_tensor(weight_info.name, tensor, clone=options.clone_tensors)
    writer.finalize()

    tokenizer.save_pretrained(out_path)
    arch_info.config.save_pretrained(out_path)


def normalize_token(token: str) -> str:
    if token.startswith("Ġ"):
        return "▁" + token[1:]
    return token


def get_embedding_info(
    model: ModelReference, options: MergeOptions
) -> Tuple[WeightInfo, WeightInfo]:
    cfg = model.config(trust_remote_code=options.trust_remote_code)
    arch_info = get_architecture_info(cfg)

    embed, lm_head = None, None
    for weight_info in arch_info.pre_weights(cfg):
        if weight_info.is_embed:
            if embed is not None:
                raise RuntimeError("Multiple input embeddings found")
            embed = weight_info

    for weight_info in arch_info.post_weights(cfg):
        if weight_info.is_embed:
            if lm_head is not None:
                raise RuntimeError("Multiple output embeddings found")
            lm_head = weight_info

    return embed, lm_head


def report_issue(message: str, error: bool = False):
    if error:
        logging.error(message)
        sys.exit(1)
    else:
        logging.warning(message)


def lstsq_embeddings(
    embed_0: torch.Tensor,
    embed_1: torch.Tensor,
    vocab_0: Dict[str, int],
    vocab_1: Dict[str, int],
    common_tokens: List[str],
) -> torch.Tensor:
    hidden_size_0 = embed_0.shape[1]
    hidden_size_1 = embed_1.shape[1]

    e_0 = torch.empty(hidden_size_0, len(common_tokens))
    e_1 = torch.empty(hidden_size_1, len(common_tokens))
    for i, token in enumerate(common_tokens):
        e_0[:, i] = embed_0[vocab_0[token]]
        e_1[:, i] = embed_1[vocab_1[token]]

    M = torch.linalg.lstsq(e_1, e_0).solution
    new_embed = embed_1 @ M
    for token in common_tokens:
        new_embed[vocab_1[token], :] = embed_0[vocab_0[token], :]

    return new_embed


def load_tokenizer(
    model: ModelReference, options: MergeOptions
) -> Tuple[transformers.PreTrainedTokenizerBase, Dict[str, int]]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.model.path,
        revision=model.model.revision,
        trust_remote_code=options.trust_remote_code,
    )
    return tokenizer, {
        normalize_token(token): i for token, i in tokenizer.get_vocab().items()
    }


def validate_architecture(
    model: ModelReference, donor: ModelReference, options: MergeOptions
) -> ConfiguredArchitectureInfo:
    model_cfg = model.config(trust_remote_code=options.trust_remote_code)
    model_arch_info = get_architecture_info(model_cfg)
    donor_arch_info = get_architecture_info(
        donor.config(trust_remote_code=options.trust_remote_code)
    )
    if donor_arch_info != model_arch_info:
        report_issue(
            f"Model architectures do not match: {model_arch_info} vs {donor_arch_info}",
            error=not options.allow_crimes,
        )

    return ConfiguredArchitectureInfo(info=model_arch_info, config=model_cfg)


if __name__ == "__main__":
    main()
