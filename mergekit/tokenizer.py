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

import json
import logging
from typing import Dict, Optional, Tuple

import tokenizers
import tokenizers.models
import torch
import tqdm
import transformers

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration


def get_vocab_size(model_path: str, trust_remote_code: bool) -> Optional[int]:
    try:
        cfg = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        return cfg.vocab_size
    except Exception as e:
        logging.warning(f"Unable to get vocab size for {model_path}", exc_info=e)

    return None


def get_stripped_tokenizer(
    path: str, trust_remote_code: bool = False
) -> transformers.PreTrainedTokenizerFast:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path, trust_remote_code=trust_remote_code, use_fast=True
    )
    vocab_size = get_vocab_size(path) or len(tokenizer.get_vocab())

    unused_toks = [
        tok for tok, idx in tokenizer.get_vocab().items() if idx >= vocab_size
    ]
    if not unused_toks:
        return tokenizer

    if not tokenizer.is_fast:
        raise RuntimeError(
            f"Model {path} has unused tokens and does not support fast "
            "tokenizer - can not be used in tokenizer merge"
        )

    tok_dict = json.loads(tokenizer._tokenizer.to_str())
    if tok_dict["model"]["type"] != "BPE":
        raise RuntimeError(
            f"Tokenizer for {path} has type {tok_dict['model']['type']}, "
            "but only BPE is currently supported for tokenizer merge"
        )

    tok_dict["added_tokens"] = [
        e for e in tok_dict["added_tokens"] if e["id"] < vocab_size
    ]

    for tok in unused_toks:
        if tok in tok_dict["model"]["vocab"]:
            del tok_dict["model"]["vocab"][tok]

    def _keep_merge(m):
        toks = m.split(" ")
        for tok in toks:
            if tok in unused_toks:
                return False
        return True

    tok_dict["model"]["merges"] = [
        e for e in tok_dict["model"]["merges"] if _keep_merge(e)
    ]
    tokenizer._tokenizer = tokenizers.Tokenizer.from_str(json.dumps(tok_dict))
    return tokenizer


def build_union_tokenizer(
    base_tok: transformers.PreTrainedTokenizerBase,
    tokenizers: Dict[ModelReference, transformers.PreTrainedTokenizerBase],
) -> transformers.PreTrainedTokenizerBase:
    out_added_tokens = {}
    out_vocab = {}

    for model, tokenizer in tokenizers.items():
        vocab_size = get_vocab_size(model) or tokenizer.vocab_size
        added_tokens = tokenizer.added_tokens_decoder

        vocab = tokenizer.get_vocab()
        for tok, idx in vocab.items():
            if idx >= vocab_size:
                logging.warning(
                    f"Token {repr(tok)} present in {model.path} tokenizer but >= vocab_size"
                )
                continue
            if tok in added_tokens:
                # deal with later
                continue

            if tok not in out_vocab:
                out_vocab[tok] = len(out_vocab)

        for tok, info in tokenizer.added_tokens_decoder.items():
            if tok in out_added_tokens:
                if out_added_tokens[tok] != info:
                    logging.warning(
                        f"Token '{tok}' added with multiple different settings, using first"
                    )
                continue
            out_added_tokens[tok] = info

    res = base_tok
    orig_base_vocab = base_tok.get_vocab()
    for tok in out_vocab:
        if tok in out_added_tokens:
            continue

        if tok not in orig_base_vocab:
            res.add_tokens(tok)

    for info in out_added_tokens.values():
        res.add_tokens(info)
    return res


def build_tokenizer(
    config: MergeConfiguration,
    trust_remote_code: bool,
) -> Tuple[transformers.PreTrainedTokenizer, Dict[ModelReference, torch.IntTensor]]:
    base_model = None
    if config.base_model:
        base_model = ModelReference.parse(config.base_model)
    if base_model is None:
        base_model = config.referenced_models()[0]
    if base_model is None:
        raise RuntimeError("No models referenced")

    tokenizer_out = get_stripped_tokenizer(
        base_model.path, trust_remote_code=trust_remote_code
    )

    # load all tokenizers
    logging.info("Loading tokenizers")
    tokenizers = {base_model: tokenizer_out}
    for model in config.referenced_models():
        if model == base_model:
            continue

        try:
            model_tok = get_stripped_tokenizer(
                model.path, trust_remote_code=trust_remote_code
            )
        except Exception:
            logging.warning(
                f"Unable to load tokenizer for {model}. Assuming same as {base_model}."
            )
            continue
        tokenizers[model] = model_tok

    logging.info("Building output tokenizer")
    # build final vocabulary
    if config.tokenizer_source == "base":
        # it done
        pass
    elif config.tokenizer_source == "union":
        tokenizer_out = build_union_tokenizer(tokenizer_out, tokenizers)
    elif config.tokenizer_source.startswith("model:"):
        tokenizer_out = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_source.removeprefix("model:"),
            trust_remote_code=trust_remote_code,
        )
    else:
        raise RuntimeError(f"Unimplemented tokenizer source: {config.tokenizer_source}")

    vocab_out = tokenizer_out.get_vocab()

    logging.info("Building permutations")
    permutations = {}
    for model in tqdm.tqdm(config.referenced_models()):
        if model in tokenizers:
            model_vocab = tokenizers[model].get_vocab()
        else:
            model_vocab = tokenizers[base_model].get_vocab()

        vocab_size = get_vocab_size(model, trust_remote_code=trust_remote_code)
        if vocab_size is None:
            vocab_size = len(model_vocab)

        p = torch.zeros(len(vocab_out), vocab_size, dtype=torch.int32)
        for tok in model_vocab:
            if tok not in vocab_out:
                continue

            orig_idx = model_vocab[tok]
            if orig_idx >= vocab_size:
                logging.warning(
                    f"{model} token {repr(tok)} has index {orig_idx}>{vocab_size-1} (padding?)"
                )
                continue

            new_idx = vocab_out[tok]
            p[new_idx, orig_idx] = 1
        permutations[model] = p

    return tokenizer_out, permutations
