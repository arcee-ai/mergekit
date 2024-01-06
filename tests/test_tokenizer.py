import json
import os
import tempfile
from typing import List, Optional, Union

import pytest
import tokenizers
from common import make_picollama, run_and_check_merge
from transformers import LlamaTokenizerFast, PreTrainedTokenizerBase

from mergekit.config import InputModelDefinition, MergeConfiguration, ParameterSetting


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

        # output should have all tokens used by any model
        # but not include any unused tokens
        run_and_check_merge(
            config,
            validate=check_tokenizer(
                expected_size=66,
                expected_added_ct=5,
                must_contain=["<|im_start|>", "<|im_end|>"],
                must_not_contain=[f"<UNUSED_{idx}>" for idx in range(4)],
            ),
        )

    def test_source_model(self, model_base: str, model_padded: str, model_chatml: str):
        config = self.make_config(
            [model_base, model_padded, model_chatml],
            base_model=model_base,
            tokenizer_source="model:" + model_chatml,
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
            embed_slerp=True,
            t="0.5",
        )

        run_and_check_merge(
            config,
            validate=check_tokenizer(
                expected_size=66,
                must_contain=["<|im_start|>", "<|im_end|>"],
            ),
        )

    def make_config(
        self,
        models: List[str],
        base_model: Optional[str] = None,
        merge_method: str = "linear",
        tokenizer_source: Optional[str] = None,
        embed_slerp: bool = False,
        t: Optional[ParameterSetting] = None,
    ):
        parameters = {"embed_slerp": embed_slerp}
        if t is not None:
            parameters["t"] = t

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
            dtype="bfloat16",
            tokenizer_source=tokenizer_source,
            parameters=parameters,
        )
        return config
