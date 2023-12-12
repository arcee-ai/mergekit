import logging
from typing import Dict, List, Optional, Tuple

import torch
import tqdm
import transformers
from pydantic import BaseModel

from mergekit.common import ModelReference
from mergekit.graph import Task


def get_vocab_size(model_path: str, trust_remote_code: bool) -> Optional[int]:
    try:
        cfg = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        return cfg.vocab_size
    except Exception as e:
        logging.warning(f"Unable to get vocab size for {model_path}", exc_info=e)

    return None


def build_tokenizer(
    base_model: Optional[ModelReference],
    referenced_models: List[ModelReference],
    tokenizer_source: str,
    trust_remote_code: bool,
) -> Tuple[transformers.PreTrainedTokenizer, Dict[ModelReference, torch.IntTensor]]:
    if base_model is None:
        base_model = referenced_models()[0]
    if base_model is None:
        raise RuntimeError("No models referenced")

    tokenizer_out = transformers.AutoTokenizer.from_pretrained(
        base_model.path, trust_remote_code=trust_remote_code
    )

    # load all tokenizers
    logging.info("Loading tokenizers")
    vocabularies = {base_model: tokenizer_out.get_vocab()}
    for model in referenced_models:
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
        vocabularies[model] = model_tok.get_vocab()

    logging.info("Building output tokenizer")
    # build final vocabulary
    if tokenizer_source == "base":
        # it done
        pass
    elif tokenizer_source == "union":
        added = set(tokenizer_out.get_vocab().keys())

        for model_vocab in tqdm.tqdm(vocabularies.values(), total=len(vocabularies)):
            for tok in tqdm.tqdm(model_vocab, leave=False):
                if tok not in added:
                    tokenizer_out.add_tokens(tok)
                    added.add(tok)

        del added
    elif tokenizer_source.startswith("model:"):
        tokenizer_out = transformers.AutoTokenizer.from_pretrained(
            tokenizer_source.removeprefix("model:"),
            trust_remote_code=trust_remote_code,
        )
    else:
        raise RuntimeError(f"Unimplemented tokenizer source: {tokenizer_source}")

    vocab_out = tokenizer_out.get_vocab()

    logging.info("Building permutations")
    permutations = {}
    for model in tqdm.tqdm(referenced_models):
        if model in vocabularies:
            model_vocab = vocabularies[model]
        else:
            model_vocab = vocabularies[base_model]

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


class TokenizerInfo(BaseModel, arbitrary_types_allowed=True):
    tokenizer: transformers.PreTrainedTokenizerBase
    permutations: Optional[Dict[ModelReference, torch.Tensor]]


class BuildTokenizer(Task[TokenizerInfo]):
    base_model: Optional[ModelReference]
    referenced_models: Tuple[ModelReference, ...]
    tokenizer_source: str
    trust_remote_code: bool = False

    def arguments(self) -> Dict[str, Task]:
        return {}

    def execute(self, **_kwargs) -> TokenizerInfo:
        tokenizer, permutations = build_tokenizer(
            self.base_model,
            self.referenced_models,
            self.tokenizer_source,
            self.trust_remote_code,
        )
        return TokenizerInfo(tokenizer=tokenizer, permutations=permutations)
