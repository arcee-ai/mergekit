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
from enum import Enum
from typing import List, Optional

import click
import torch
import tqdm
import transformers
import yaml
from pydantic import BaseModel

from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference
from mergekit.io import LazyTensorLoader, TensorWriter
from mergekit.merge import MergeOptions
from mergekit.options import add_merge_options
from mergekit.tokenizer import (
    build_intersection_tokenizer,
    build_union_tokenizer,
    filter_tokenizer,
    get_stripped_tokenizer,
)


class TokenizerMode(Enum):
    UNION = "union"
    INTERSECTION = "intersection"
    BASE = "base"


class AddedTokenEmbedSource(Enum):
    ZERO = "zero"
    RANDOM = "random"
    AVERAGE = "average"


class AddedTokenDef(BaseModel):
    content: str
    embed_source: AddedTokenEmbedSource
    normalized: Optional[bool] = True
    special: Optional[bool] = False


class TokenizerDefinition(BaseModel):
    model: str
    mode: TokenizerMode
    input_tokenizers: List[str]
    remove_tokens: Optional[List[str]] = None
    add_tokens: Optional[List[AddedTokenDef]] = None
    always_keep_tokens: Optional[List[str]] = None


@click.command("mergekit-tokensurgeon")
@click.argument("config_file")
@click.argument("out_path")
@click.option(
    "--verbose", "-v", type=bool, default=False, is_flag=True, help="Verbose logging"
)
@add_merge_options
def main(
    merge_options: MergeOptions,
    config_file: str,
    out_path: str,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    with open(config_file, "r", encoding="utf-8") as file:
        config = TokenizerDefinition.model_validate(yaml.safe_load(file))

    base_tok = get_stripped_tokenizer(
        config.model, trust_remote_code=merge_options.trust_remote_code
    )
    tokenizers = {
        ModelReference(path=p): get_stripped_tokenizer(
            p, trust_remote_code=merge_options.trust_remote_code
        )
        for p in config.input_tokenizers
    }

    if config.mode == TokenizerMode.UNION:
        tokenizer = build_union_tokenizer(
            base_tok, tokenizers, trust_remote_code=merge_options.trust_remote_code
        )
    elif config.mode == TokenizerMode.INTERSECTION:
        tokenizer = build_intersection_tokenizer(
            base_tok, tokenizers, always_keep=config.always_keep_tokens
        )
    elif config.mode == TokenizerMode.BASE:
        tokenizer = base_tok
    else:
        raise NotImplementedError(config.mode)

    for tok_def in config.add_tokens or []:
        tokenizer.add_tokens(
            transformers.AddedToken(
                content=tok_def.content,
                special=tok_def.special,
                normalized=tok_def.normalized,
            )
        )

    if config.remove_tokens:
        to_remove = set(config.remove_tokens)

        def _keep_token(token: str, **kwargs) -> bool:
            return token not in to_remove

        tokenizer = filter_tokenizer(
            tokenizer, _keep_token, trust_remote_code=merge_options.trust_remote_code
        )

    tokenizer.save_pretrained(out_path)

    new_vocab = tokenizer.get_vocab()
    old_vocab = base_tok.get_vocab()
    added_tokens = {tok_def.content: tok_def for tok_def in (config.add_tokens or [])}
    vocabs = {mr: tokenizers[mr].get_vocab() for mr in tokenizers}
    if config.mode == TokenizerMode.UNION:
        loaders = {
            mr: LazyTensorLoader(
                index=mr.tensor_index(cache_dir=merge_options.transformers_cache)
            )
            for mr in tokenizers
        }
    else:
        loaders = {}

    model_ref = ModelReference(path=config.model)
    model_cfg = model_ref.config(trust_remote_code=merge_options.trust_remote_code)
    model_cfg.vocab_size = len(tokenizer.get_vocab())
    model_cfg.save_pretrained(out_path)

    embed_names = get_architecture_info(model_cfg).embed_weights()
    loader = LazyTensorLoader(
        model_ref.tensor_index(cache_dir=merge_options.transformers_cache)
    )
    writer = TensorWriter(
        out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )
    for tensor_name in tqdm.tqdm(loader.index.tensor_paths):
        tensor = loader.get_tensor(tensor_name)

        if tensor_name in embed_names:
            # honk honk honk

            new_tensor = torch.zeros(
                len(new_vocab), tensor.shape[1], dtype=tensor.dtype
            )

            donor_tensors = {mr: loaders[mr].get_tensor(tensor_name) for mr in loaders}

            for tok, new_idx in new_vocab.items():
                if tok in old_vocab:
                    old_idx = old_vocab[tok]
                    new_tensor[new_idx, :] = tensor[old_idx, :]
                elif tok in added_tokens:
                    new_tensor[new_idx, :] = get_added_embed(
                        added_tokens[tok], base_tok, embed_tensor=tensor
                    )
                else:
                    sample_ct = 0
                    for model_ref, vocab in vocabs:
                        if tok in vocab:
                            donor_tensor = donor_tensors[model_ref]
                            if donor_tensor.shape[-1] != tensor.shape[-1]:
                                continue

                            donor_idx = vocab[tok]
                            new_tensor[new_idx, :] += donor_tensor[donor_idx, :]
                            sample_ct += 1
                    if sample_ct > 0:
                        new_tensor[new_idx, :] /= sample_ct
                    else:
                        logging.warning(
                            f"No input models have an embedding for {repr(tok)} - setting to zero"
                        )
            tensor = new_tensor

        writer.save_tensor(tensor_name, tensor)

    writer.finalize()


def get_added_embed(
    tok_def: AddedTokenDef,
    base_tok: transformers.PreTrainedTokenizerBase,
    embed_tensor: torch.Tensor,
) -> torch.FloatTensor:
    if tok_def.embed_source == AddedTokenEmbedSource.RANDOM:
        return torch.randn((1, embed_tensor.shape[1]))
    elif tok_def.embed_source == AddedTokenEmbedSource.ZERO:
        return torch.zeros((1, embed_tensor.shape[1]))
    elif tok_def.embed_source == AddedTokenEmbedSource.AVERAGE:
        token_ids = base_tok(
            tok_def.content, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        onehot = torch.nn.functional.one_hot(
            token_ids,
            num_classes=embed_tensor.shape[0],
        )  # (1, seq_len, 32000)
        h = onehot.float() @ embed_tensor.float()  # (1, seq_len, hidden_size)
        return h.sum(dim=1).squeeze()
    else:
        raise RuntimeError(f"Unimplemented embed source {tok_def.embed_source}")
