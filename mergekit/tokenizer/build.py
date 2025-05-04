# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import json
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import tokenizers
import tokenizers.models
import tqdm
import transformers
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.architecture import arch_info_for_config
from mergekit.common import ModelPath, ModelReference, get_config_value
from mergekit.graph import Task

LOG = logging.getLogger(__name__)


def get_vocab_size(model_path: ModelPath, trust_remote_code: bool) -> Optional[int]:
    try:
        cfg = transformers.AutoConfig.from_pretrained(
            model_path.path,
            revision=model_path.revision,
            trust_remote_code=trust_remote_code,
        )
        arch_info = arch_info_for_config(cfg)
        key = "vocab_size"
        if arch_info is not None:
            key = arch_info.vocab_size_config_key or "vocab_size"
        return get_config_value(cfg, key)
    except Exception as e:
        LOG.warning(f"Unable to get vocab size for {model_path}", exc_info=e)

    return None


def get_stripped_tokenizer(
    path: ModelPath, trust_remote_code: bool = False
) -> transformers.PreTrainedTokenizerFast:
    """
    Return a tokenizer for a model that only contains used tokens.

    Strips any tokens with indices >= model.vocab_size.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path.path,
        revision=path.revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    vocab_size = get_vocab_size(path, trust_remote_code=trust_remote_code) or len(
        tokenizer.get_vocab()
    )

    unused_toks = [
        tok for tok, idx in tokenizer.get_vocab().items() if idx >= vocab_size
    ]
    if not unused_toks:
        # we're good, ship it
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
        if isinstance(m, str) and m.count(" ") == 1:
            toks = m.split(" ")
        elif isinstance(m, list):
            toks = m
        else:
            raise RuntimeError(f"Unexpected merge format: {repr(m)} ({type(m)})")
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
    trust_remote_code: bool = False,
) -> transformers.PreTrainedTokenizerBase:
    out_added_tokens = {}
    out_vocab = {}

    warned_added_tokens = set()

    for model, tokenizer in tokenizers.items():
        vocab_size = (
            get_vocab_size(model.model, trust_remote_code=trust_remote_code)
            or tokenizer.vocab_size
        )
        added_tokens = tokenizer.added_tokens_decoder

        vocab = tokenizer.get_vocab()
        for tok, idx in vocab.items():
            if idx >= vocab_size:
                LOG.warning(
                    f"Token {repr(tok)} present in {str(model)} tokenizer but >= vocab_size"
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
                    LOG.warning(
                        f"Token '{tok}' added with multiple different settings, using first"
                    )
                    warned_added_tokens.add(tok)

                continue
            out_added_tokens[tok] = info

    # HACK: save base tokenizer to temp dir and reload to avoid mutating base_tok
    with tempfile.TemporaryDirectory() as p:
        base_tok.save_pretrained(p, legacy_format=False, safe_serialization=True)
        res = transformers.AutoTokenizer.from_pretrained(
            p, use_fast=True, trust_remote_code=trust_remote_code
        )

    orig_base_vocab = base_tok.get_vocab()
    for tok in out_vocab:
        if tok in out_added_tokens:
            continue

        if tok not in orig_base_vocab:
            res.add_tokens(tok)

    for info in out_added_tokens.values():
        res.add_tokens(info)
    return res


class TokenizerInfo(BaseModel, arbitrary_types_allowed=True):
    tokenizer: transformers.PreTrainedTokenizerBase
    permutations: Dict[ModelReference, Dict[int, int]]
    original_vocabs: Dict[ModelReference, Dict[str, int]]


def build_tokenizer(
    base_model: Optional[ModelReference],
    referenced_models: List[ModelReference],
    tokenizer_source: Union[Literal["union"], Literal["base"], ModelReference],
    trust_remote_code: bool,
    add_tokens: Optional[List[str]] = None,
) -> TokenizerInfo:
    if base_model is None:
        base_model = referenced_models[0]
    if base_model is None:
        raise RuntimeError("No models referenced")

    #
    tokenizer_base = get_stripped_tokenizer(
        base_model.model, trust_remote_code=trust_remote_code
    )

    # load all tokenizers
    LOG.info("Loading tokenizers")
    tokenizers = {base_model: tokenizer_base}
    for model in referenced_models:
        if model == base_model:
            continue

        try:
            model_tok = transformers.AutoTokenizer.from_pretrained(
                model.model.path,
                revision=model.model.revision,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            LOG.error(e)
            LOG.warning(
                f"Unable to load tokenizer for {model}. Assuming same as {base_model}."
            )
            continue
        tokenizers[model] = model_tok

    LOG.info("Building output tokenizer")
    # build final vocabulary
    if isinstance(tokenizer_source, ModelReference):
        tokenizer_out = transformers.AutoTokenizer.from_pretrained(
            tokenizer_source.model.path,
            revision=tokenizer_source.model.revision,
            trust_remote_code=trust_remote_code,
        )
    elif tokenizer_source == "base":
        # it done
        tokenizer_out = tokenizer_base
    elif tokenizer_source == "union":
        tokenizer_out = build_union_tokenizer(
            tokenizer_base, tokenizers, trust_remote_code=trust_remote_code
        )
    else:
        raise RuntimeError(f"Unimplemented tokenizer source: {tokenizer_source}")

    for tok in add_tokens:
        tokenizer_out.add_tokens(tok)

    vocab_out = tokenizer_out.get_vocab()

    LOG.info("Building permutations")
    permutations = {}
    for model in (
        pbar := tqdm.tqdm(referenced_models, desc="Building tokenizer permutations")
    ):
        if model in tokenizers:
            model_vocab = tokenizers[model].get_vocab()
        else:
            model_vocab = tokenizers[base_model].get_vocab()

        vocab_size = get_vocab_size(model.model, trust_remote_code=trust_remote_code)
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
                LOG.warning(
                    f"{model} token {repr(tok)} has index {orig_idx}>{vocab_size - 1} (padding?)"
                )
                continue

            p[new_idx] = orig_idx

        permutations[model] = p

    del pbar

    return TokenizerInfo(
        tokenizer=tokenizer_out,
        permutations=permutations,
        original_vocabs={model: tok.get_vocab() for model, tok in tokenizers.items()},
    )


class BuildTokenizer(Task[TokenizerInfo]):
    base_model: Optional[ModelReference]
    referenced_models: Tuple[ModelReference, ...]
    tokenizer_source: Union[Literal["union"], Literal["base"], ModelReference]
    add_tokens: Optional[Tuple[str, ...]]
    trust_remote_code: bool = False

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **_kwargs) -> TokenizerInfo:
        return build_tokenizer(
            base_model=self.base_model,
            referenced_models=self.referenced_models,
            tokenizer_source=self.tokenizer_source,
            trust_remote_code=self.trust_remote_code,
            add_tokens=self.add_tokens,
        )
