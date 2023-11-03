import logging
from typing import Dict, Tuple

import torch
import transformers

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration


def build_tokenizer(
    config: MergeConfiguration,
) -> Tuple[transformers.PreTrainedTokenizer, Dict[ModelReference, torch.IntTensor]]:
    base_model = None
    if config.base_model:
        base_model = ModelReference.parse(config.base_model)
    if base_model is None:
        base_model = config.referenced_models()[0]
    if base_model is None:
        raise RuntimeError("No models referenced")

    tokenizer_out = transformers.AutoTokenizer.from_pretrained(base_model.path)

    # load all tokenizers
    vocabularies = {base_model: tokenizer_out.vocab}
    for model in config.referenced_models():
        if model == base_model:
            continue

        try:
            model_tok = transformers.AutoTokenizer.from_pretrained(model.path)
        except Exception:
            logging.warning(
                f"Unable to load tokenizer for {model}. Assuming same as {base_model}."
            )
            continue
        vocabularies[model] = model_tok.vocab

    # build final vocabulary
    if config.tokenizer_source == "base":
        # it done
        pass
    elif config.tokenizer_source == "union":
        for model_vocab in vocabularies.values():
            for tok in model_vocab:
                if tok not in tokenizer_out.vocab:
                    tokenizer_out.add_tokens(tok)
    elif config.tokenizer_source.startswith("model:"):
        tokenizer_out = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_source.removeprefix("model:")
        )
    else:
        raise RuntimeError(f"Unimplemented tokenizer source: {config.tokenizer_source}")

    vocab_out = tokenizer_out.vocab

    permutations = {}
    for model in config.referenced_models():
        if model in vocabularies:
            model_vocab = vocabularies[model]
        else:
            model_vocab = vocabularies[base_model]

        p = torch.zeros(len(vocab_out), len(model_vocab), dtype=torch.int32)
        for tok in model_vocab:
            if tok not in vocab_out:
                continue

            orig_idx = model_vocab[tok]
            new_idx = vocab_out[tok]
            p[new_idx, orig_idx] = 1
        permutations[model] = p

    return tokenizer_out, permutations
