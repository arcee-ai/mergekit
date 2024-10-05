import json
import os
import tempfile
from typing import Dict, List, Optional, Union

import pytest
import tokenizers
import torch
from common import make_picollama, run_and_check_merge
from transformers import LlamaTokenizerFast, PreTrainedTokenizerBase

from mergekit.config import InputModelDefinition, MergeConfiguration
from mergekit.io import LazyTensorLoader
from mergekit.tokenizer import TokenizerConfig


@pytest.fixture(scope="session")
def model_base(tmp_path_factory):
    model_path = make_picollama(tmp_path_factory.mktemp("model_base"), vocab_size=64)
    make_tokenizer(vocab_size=64, added_tokens=[]).save_pretrained(model_path)
    return model_path


@pytest.fixture(scope="session")
def model_chatml(tmp_path_factory):
    model_path = make_picollama(tmp_path_factory.mktemp("model_chatml"), vocab_size=66)
    make_tokenizer(
        vocab_size=64, added_tokens=["<|im_start|>", "<|im_end|>"]
    ).save_pretrained(model_path)
    return model_path


@pytest.fixture(scope="session")
def model_padded(tmp_path_factory):
    model_path = make_picollama(tmp_path_factory.mktemp("model_padded"), vocab_size=64)
    make_tokenizer(
        vocab_size=64,
        added_tokens=["<UNUSED_0>", "<UNUSED_1>", "<UNUSED_2>", "<UNUSED_3>"],
    ).save_pretrained(model_path)
    return model_path


def make_tokenizer(
    vocab_size: int, added_tokens: List[Union[str, tokenizers.AddedToken]]
) -> PreTrainedTokenizerBase:
    tokens = ["<unk>", "<s>", "</s>"] + [f"_tok_{idx}" for idx in range(3, vocab_size)]
    tokens = tokens[:vocab_size]
    tok_data = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": dict(zip(tokens, range(vocab_size))),
            "merges": [],
        },
        "added_tokens": [],
    }
    tok = tokenizers.Tokenizer.from_str(json.dumps(tok_data))
    with tempfile.TemporaryDirectory() as p:
        tok_path = os.path.join(p, "tokenizer.json")
        tok.save(tok_path)
        res = LlamaTokenizerFast(tokenizer_file=tok_path)

    res.add_tokens(added_tokens)
    return res


def check_tokenizer(
    expected_size: int,
    expected_added_ct: Optional[int] = None,
    must_contain: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None,
):
    def _cb(model_path: str):
        tok: LlamaTokenizerFast = LlamaTokenizerFast.from_pretrained(model_path)
        vocab = tok.get_vocab()
        print(vocab)
        assert len(vocab) == expected_size

        if expected_added_ct is not None:
            assert len(tok.added_tokens_decoder) == expected_added_ct

        if must_contain:
            for tok in must_contain:
                assert tok in vocab

        if must_not_contain:
            for tok in must_not_contain:
                assert tok not in vocab

    return _cb


class ModelEmbeddings:
    embed_tokens: torch.Tensor
    vocab: Dict[str, int]

    def __init__(self, model_path: str):
        tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
        loader = LazyTensorLoader.from_disk(model_path)
        self.embed_tokens = loader.get_tensor("model.embed_tokens.weight")
        self.vocab = tokenizer.get_vocab()

    def token_embedding(self, token: str) -> Optional[torch.Tensor]:
        idx = self.vocab.get(token)
        if idx is None:
            return None
        return self.embed_tokens[idx, :]


class TestTokenizerMerges:
    def test_legacy_mode(self, model_base: str, model_padded: str, model_chatml: str):
        config = self.make_config(
            [model_base, model_padded, model_chatml], base_model=model_base
        )
        # when no tokenizer_source is set, expect output tokenizer to be from base_model
        run_and_check_merge(
            config, validate=check_tokenizer(expected_size=64, expected_added_ct=3)
        )

    def test_source_base(self, model_base: str, model_padded: str, model_chatml: str):
        config = self.make_config(
            [model_base, model_padded, model_chatml],
            base_model=model_base,
            tokenizer_source="base",
        )
        # expect the same output but it's a different code path
        run_and_check_merge(
            config, validate=check_tokenizer(expected_size=64, expected_added_ct=3)
        )

    def test_source_union(self, model_base: str, model_padded: str, model_chatml: str):
        config = self.make_config(
            [model_base, model_padded, model_chatml],
            base_model=model_base,
            tokenizer_source="union",
        )

        def _check_embed(model_path: str):
            # output should have all tokens used by any model
            # but not include any unused tokens
            check_tokenizer(
                expected_size=66,
                expected_added_ct=5,
                must_contain=["<|im_start|>", "<|im_end|>"],
                must_not_contain=[f"<UNUSED_{idx}>" for idx in range(4)],
            )(model_path)
            emb_out = ModelEmbeddings(model_path)
            emb_chatml = ModelEmbeddings(model_chatml)

            assert torch.allclose(
                emb_out.token_embedding("<|im_start|>"),
                emb_chatml.token_embedding("<|im_start|>"),
            ), "Token <|im_start|> should be from model_chatml"
            assert torch.allclose(
                emb_out.token_embedding("<|im_end|>"),
                emb_chatml.token_embedding("<|im_end|>"),
                atol=1e-3,
                rtol=1e-4,
            ), "Token <|im_end|> should be from model_chatml"

        run_and_check_merge(
            config,
            validate=_check_embed,
        )

    def test_source_model(self, model_base: str, model_padded: str, model_chatml: str):
        config = self.make_config(
            [model_base, model_padded, model_chatml],
            base_model=model_base,
            tokenizer_source=model_chatml,
        )
        # tokenizer should match model_chatml
        run_and_check_merge(
            config,
            validate=check_tokenizer(
                expected_size=66, must_contain=["<|im_start|>", "<|im_end|>"]
            ),
        )

    def test_slerp_union(self, model_base: str, model_chatml: str):
        config = self.make_config(
            [model_base, model_chatml],
            base_model=model_base,
            tokenizer_source="union",
            merge_method="slerp",
            t=0.5,
        )

        run_and_check_merge(
            config,
            validate=check_tokenizer(
                expected_size=66,
                must_contain=["<|im_start|>", "<|im_end|>"],
            ),
        )

    def test_force_token(self, model_base: str, model_chatml: str):
        config = self.make_config(
            [model_base, model_chatml],
            base_model=model_base,
            merge_method="linear",
            tokenizer_config=TokenizerConfig(
                source="union",
                tokens={
                    "_tok_10": {"source": model_chatml, "force": True},
                    "_tok_11": {"source": model_base, "force": True},
                },
            ),
        )

        def _check_embed(model_path: str):
            check_tokenizer(
                expected_size=66, must_contain=["<|im_start|>", "<|im_end|>"]
            )(model_path)
            emb_out = ModelEmbeddings(model_path)
            emb_base = ModelEmbeddings(model_base)
            emb_chatml = ModelEmbeddings(model_chatml)

            assert torch.allclose(
                emb_out.token_embedding("_tok_10"),
                emb_chatml.token_embedding("_tok_10"),
            ), "Token _tok_10 should be from model_chatml"
            assert torch.allclose(
                emb_out.token_embedding("_tok_11"),
                emb_base.token_embedding("_tok_11"),
            ), "Token _tok_11 should be from model_base"

        run_and_check_merge(config, validate=_check_embed)

    def test_model_token_id(self, model_base: str, model_chatml: str):
        config = self.make_config(
            [model_base, model_chatml],
            base_model=model_base,
            merge_method="linear",
            tokenizer_config=TokenizerConfig(
                source="base",
                tokens={
                    "_tok_20": {
                        "source": {
                            "kind": "model_token",
                            "model": model_chatml,
                            "token_id": 64,
                        },
                        "force": True,
                    },
                    "_tok_21": {
                        "source": {
                            "kind": "model_token",
                            "model": model_base,
                            "token": "<s>",
                        },
                        "force": True,
                    },
                },
            ),
        )

        def _check_embed(model_path: str):
            check_tokenizer(expected_size=64, must_contain=["_tok_10"])(model_path)
            emb_out = ModelEmbeddings(model_path)
            emb_base = ModelEmbeddings(model_base)
            emb_chatml = ModelEmbeddings(model_chatml)

            assert torch.allclose(
                emb_out.token_embedding("_tok_20"), emb_chatml.embed_tokens[64, :]
            ), "Token _tok_20 should be == model_chatml token 64"
            assert torch.allclose(
                emb_out.token_embedding("_tok_21"), emb_base.token_embedding("<s>")
            ), "Token _tok_21 should be == model_base <s>"

        run_and_check_merge(config, validate=_check_embed)

    def make_config(
        self,
        models: List[str],
        base_model: Optional[str] = None,
        merge_method: str = "linear",
        tokenizer_source: Optional[str] = None,
        t: Optional[float] = None,
        tokenizer_config: Optional[TokenizerConfig] = None,
    ):
        parameters = {"t": t} if t is not None else {}
        config = MergeConfiguration(
            merge_method=merge_method,
            base_model=base_model,
            models=[
                InputModelDefinition(
                    model=m,
                    parameters={"weight": 1.0},
                )
                for m in models
            ],
            dtype="float32",
            tokenizer_source=tokenizer_source,
            parameters=parameters,
            tokenizer=tokenizer_config,
        )
        return config
