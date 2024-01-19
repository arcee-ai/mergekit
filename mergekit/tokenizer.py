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
import tempfile
from typing import Callable, Dict, Optional, Set, Tuple

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


def _copy_tokenizer(
    tokenizer: transformers.PreTrainedTokenizerBase, trust_remote_code: bool = False
) -> transformers.PreTrainedTokenizerBase:
    # HACK: save tokenizer to temp dir and reload
    with tempfile.TemporaryDirectory() as p:
        tokenizer.save_pretrained(p, legacy_format=False, safe_serialization=True)
        return transformers.AutoTokenizer.from_pretrained(
            p, use_fast=True, trust_remote_code=trust_remote_code
        )


def filter_tokenizer(
    tokenizer: transformers.PreTrainedTokenizerBase,
    keep_token: Callable,
    renumber: bool = False,
    trust_remote_code: bool = False,
) -> transformers.PreTrainedTokenizerBase:
    vocab = tokenizer.get_vocab()
    removed_toks = set()
    for tok, index in vocab.items():
        if not keep_token(token=tok, index=index):
            removed_toks.add(tok)

    if not removed_toks:
        return tokenizer

    if not isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
        raise RuntimeError("Need a fast tokenizer to be able to remove tokens")

    tok_dict = json.loads(tokenizer._tokenizer.to_str())
    if tok_dict["model"]["type"] != "BPE":
        raise RuntimeError(
            f"Tokenizer has type {tok_dict['model']['type']}, but only "
            "BPE is currently supported for tokenizer merge"
        )

    new_added_tokens = []
    for added_tok in tok_dict["added_tokens"]:
        if keep_token(token=added_tok["content"], index=added_tok["id"]):
            new_added_tokens.append(added_tok)
        else:
            removed_toks.add(added_tok["content"])

    for tok in removed_toks:
        if tok in tok_dict["model"]["vocab"]:
            del tok_dict["model"]["vocab"][tok]

    if renumber:
        new_vocab = dict(
            zip(
                tok_dict["model"]["vocab"].keys(),
                range(len(tok_dict["model"]["vocab"])),
            )
        )
        next_idx = len(new_vocab)
        for added_tok in new_added_tokens:
            added_tok["id"] = next_idx
            next_idx += 1
        tok_dict["model"]["vocab"] = new_vocab

    tok_dict["added_tokens"] = new_added_tokens

    def _keep_merge(m):
        toks = m.split(" ")
        for tok in toks:
            if tok in removed_toks:
                return False
        return True

    tok_dict["model"]["merges"] = [
        e for e in tok_dict["model"]["merges"] if _keep_merge(e)
    ]
    res = _copy_tokenizer(tokenizer, trust_remote_code=trust_remote_code)
    res._tokenizer = tokenizers.Tokenizer.from_str(json.dumps(tok_dict))
    return res


def get_stripped_tokenizer(
    path: str,
    trust_remote_code: bool = False,
) -> transformers.PreTrainedTokenizerFast:
    """
    Return a tokenizer for a model that only contains used tokens.

    Strips any tokens with indices >= model.vocab_size.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path, trust_remote_code=trust_remote_code, use_fast=True
    )
    vocab_size = get_vocab_size(path, trust_remote_code=trust_remote_code) or len(
        tokenizer.get_vocab()
    )

    def _keep_tok(index: int, **_kwargs):
        return index < vocab_size

    return filter_tokenizer(
        tokenizer, keep_token=_keep_tok, trust_remote_code=trust_remote_code
    )


def build_intersection_tokenizer(
    base_tok: transformers.PreTrainedTokenizerBase,
    tokenizers: Dict[ModelReference, transformers.PreTrainedTokenizerBase],
    trust_remote_code: bool = False,
    always_keep: Optional[Set[str]] = None,
) -> transformers.PreTrainedTokenizerBase:
    base_vocab = set(base_tok.get_vocab())
    out_vocab = set(base_vocab)
    for tok in tokenizers.values():
        out_vocab = out_vocab.intersection(set(tok.get_vocab()))

    if out_vocab == base_vocab:
        return base_tok

    if always_keep:
        for tok in always_keep:
            if tok in base_vocab:
                out_vocab.add(tok)

    if not isinstance(base_tok, transformers.PreTrainedTokenizerFast):
        raise RuntimeError(
            "Can't strip tokens from slow tokenizer - need fast tokenizer"
        )

    def _keep_token(token: str, **kwargs) -> bool:
        return token in out_vocab

    return filter_tokenizer(
        base_tok._tokenizer,
        _keep_token,
        renumber=True,
        trust_remote_code=trust_remote_code,
    )


def build_union_tokenizer(
    base_tok: transformers.PreTrainedTokenizerBase,
    tokenizers: Dict[ModelReference, transformers.PreTrainedTokenizerBase],
    trust_remote_code: bool = False,
) -> transformers.PreTrainedTokenizerBase:
    out_added_tokens = {}
    out_vocab = {}

    warned_added_tokens = set()

    for model, tokenizer in tokenizers.items():
        vocab_size = (
            get_vocab_size(model, trust_remote_code=trust_remote_code)
            or tokenizer.vocab_size
        )
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

        for tok_idx, info in tokenizer.added_tokens_decoder.items():
            tok = info.content
            if tok_idx >= vocab_size:
                continue

            if tok in out_added_tokens:
                if (out_added_tokens[tok] != info) and tok not in warned_added_tokens:
                    logging.warning(
                        f"Token '{tok}' added with multiple different settings, using first"
                    )
                    warned_added_tokens.add(tok)

                continue
            out_added_tokens[tok] = info

    res = _copy_tokenizer(base_tok, trust_remote_code=trust_remote_code)

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

    #
    tokenizer_base = get_stripped_tokenizer(
        base_model.path, trust_remote_code=trust_remote_code
    )

    # load all tokenizers
    logging.info("Loading tokenizers")
    tokenizers = {base_model: tokenizer_base}
    for model in config.referenced_models():
        if model == base_model:
            continue

        try:
            model_tok = transformers.AutoTokenizer.from_pretrained(
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
        tokenizer_out = tokenizer_base
    elif config.tokenizer_source == "union":
        tokenizer_out = build_union_tokenizer(
            tokenizer_base, tokenizers, trust_remote_code=trust_remote_code
        )
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

        p = {}
        for tok in vocab_out:
            new_idx = vocab_out[tok]
            if tok not in model_vocab:
                p[new_idx] = -1
                continue

            orig_idx = model_vocab[tok]
            if orig_idx >= vocab_size:
                logging.warning(
                    f"{model} token {repr(tok)} has index {orig_idx}>{vocab_size-1} (padding?)"
                )
                continue

            p[new_idx] = orig_idx

        permutations[model] = p

    return tokenizer_out, permutations
