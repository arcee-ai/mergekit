import json
import logging

import pytest
import torch
from click.testing import CliRunner
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer

from mergekit import merge_methods
from mergekit.config import MergeConfiguration
from mergekit.io import LazyTensorLoader
from mergekit.merge import run_merge
from mergekit.merge_methods import TensorGroup
from mergekit.merge_methods.preprocessing import truncate_vocabulary
from mergekit.options import MergeOptions
from mergekit.scripts.run_yaml import main
from tests.common import make_picollama
from tests.test_tokenizer import make_tokenizer


def test_explicit_preprocessing_preserves_inputs_and_identity(caplog):
    a = torch.arange(24.0).reshape(6, 4).requires_grad_()
    b = torch.ones(4, 4, requires_grad=True)
    group = TensorGroup.from_tensors(
        [a, b], ids=["a", "b"], base_index=1, name="embed", vocabulary_axis=0
    )
    with pytest.raises(ValueError, match="Tensor size mismatch"):
        merge_methods.get("linear")((group,), parameters={"weight": [1, 1]})
    with caplog.at_level(logging.WARNING):
        aligned = truncate_vocabulary(group)
    assert aligned.metadata == group.metadata
    assert [entry.id for entry in aligned.entries] == ["a", "b"]
    assert aligned.base.id == "b"
    assert a.shape == (6, 4)
    assert aligned.entries[0].tensor.data_ptr() == a.data_ptr()
    assert "(6, 4)" in caplog.text and "(4, 4)" in caplog.text
    assert "identical token IDs" in caplog.text and "tokenizer:" in caplog.text
    (result,) = merge_methods.get("linear")((aligned,), parameters={"weight": [1, 1]})
    torch.testing.assert_close(result, (a[:4] + b) / 2)
    result.sum().backward()
    assert torch.count_nonzero(a.grad[4:]) == 0
    torch.testing.assert_close(a.grad[:4], torch.full((4, 4), 0.5))


@pytest.mark.parametrize(
    "shapes, message",
    [
        ([(6, 4), (4, 3)], "Non-vocabulary dimensions must match"),
        ([(6,), (4, 2)], "Non-vocabulary dimensions must match"),
        ([(6, 4, 2), (4, 4, 3)], "Non-vocabulary dimensions must match"),
        ([(0, 4), (4, 4)], "Empty vocabulary"),
        ([(), ()], "Invalid vocabulary_axis"),
    ],
)
def test_invalid_embedding_shapes(shapes, message):
    group = TensorGroup.from_tensors([torch.ones(s) for s in shapes], vocabulary_axis=0)
    with pytest.raises(ValueError, match=message):
        truncate_vocabulary(group)


def test_noop_and_non_embeddings(caplog):
    matched = TensorGroup.from_tensors([torch.ones(4, 3)] * 2, vocabulary_axis=0)
    assert truncate_vocabulary(matched) is matched
    unmarked = TensorGroup.from_tensors([torch.ones(4, 3), torch.ones(6, 3)])
    assert truncate_vocabulary(unmarked) is unmarked
    with pytest.raises(ValueError, match="Tensor size mismatch"):
        unmarked.validate_tensors()
    empty = TensorGroup.from_tensors([], vocabulary_axis=0)
    assert truncate_vocabulary(empty) is empty
    assert not caplog.records


@pytest.fixture
def models(tmp_path):
    small = make_picollama(tmp_path / "small", vocab_size=8)
    large = make_picollama(tmp_path / "large", vocab_size=10)
    # Extra rows in the larger model can be padding rather than tokenizer entries.
    for path in [small, large]:
        make_tokenizer(8, []).save_pretrained(path)
    return small, large


def config_for(models, method="linear", **kwargs):
    small, large = models
    return MergeConfiguration(
        merge_method=method,
        base_model=large,
        models=[{"model": large}, {"model": small}],
        parameters={"weight": 0.5, "t": 0.5},
        **kwargs,
    )


@pytest.mark.parametrize("method", ["linear", "slerp", "task_arithmetic"])
def test_opt_in_merge_truncates_without_repairing_config(
    models, tmp_path, method, caplog
):
    # Preserve the unsafe legacy escape hatch, not a loadable-model contract:
    # prefix truncation intentionally leaves the donor config unchanged. Do not
    # "repair" this mismatch here or generalize it to normal tokenizer alignment.
    output = tmp_path / "output"
    run_merge(
        config_for(models, method),
        str(output),
        MergeOptions(unsafe_truncate_embeddings=True, quiet=True),
    )
    assert AutoConfig.from_pretrained(output).vocab_size == 10
    loader = LazyTensorLoader.from_disk(str(output))
    assert loader.get_tensor("model.embed_tokens.weight").shape == (8, 32)
    assert loader.get_tensor("lm_head.weight").shape == (8, 32)
    assert "UNSAFE vocabulary truncation" in caplog.text
    if method == "linear":
        a, b = [LazyTensorLoader.from_disk(path) for path in models]
        for name in ["model.embed_tokens.weight", "lm_head.weight"]:
            expected = (a.get_tensor(name) + b.get_tensor(name)[:8]) / 2
            actual = loader.get_tensor(name)
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("allow_crimes", [False, True])
def test_default_and_allow_crimes_do_not_truncate(models, tmp_path, allow_crimes):
    with pytest.raises(ValueError, match="Tensor size mismatch"):
        run_merge(
            config_for(models),
            str(tmp_path / "output"),
            MergeOptions(allow_crimes=allow_crimes, quiet=True),
        )


@pytest.mark.parametrize(
    "kwargs", [{"tokenizer": {"source": "base"}}, {"tokenizer_source": "base"}]
)
def test_tokenizer_alignment_takes_precedence(models, tmp_path, kwargs, caplog):
    make_tokenizer(10, []).save_pretrained(models[1])
    output = tmp_path / "output"
    run_merge(
        config_for(models, **kwargs),
        str(output),
        MergeOptions(unsafe_truncate_embeddings=True, quiet=True),
    )
    assert AutoConfig.from_pretrained(output).vocab_size == 10
    loader = LazyTensorLoader.from_disk(str(output))
    assert loader.get_tensor("model.embed_tokens.weight").shape == (10, 32)
    assert loader.get_tensor("lm_head.weight").shape == (10, 32)
    assert "UNSAFE vocabulary truncation" not in caplog.text


@pytest.mark.parametrize("copy_tokenizer", [False, True])
def test_out_of_range_metadata_is_not_validated_or_repaired(
    models, tmp_path, copy_tokenizer
):
    # Legacy truncation leaves normal metadata copying untouched, even when token
    # IDs fall outside the retained tensor rows. Validation/repair would change
    # this opt-in behavior; these expectations are not a policy for normal merges.
    make_tokenizer(10, []).save_pretrained(models[1])
    donor_config = AutoConfig.from_pretrained(models[1])
    donor_config.eos_token_id = 9
    donor_config.save_pretrained(models[1])
    generation_config = {"eos_token_id": [2, 9], "sequence_bias": [[[9], 1.0]]}
    with open(f"{models[1]}/generation_config.json", "w") as file:
        json.dump(generation_config, file)
    output = tmp_path / "output"
    run_merge(
        config_for(models),
        str(output),
        MergeOptions(
            unsafe_truncate_embeddings=True, copy_tokenizer=copy_tokenizer, quiet=True
        ),
    )
    config = AutoConfig.from_pretrained(output)
    assert config.vocab_size == 10
    assert config.eos_token_id == 9
    assert (output / "tokenizer.json").exists() == copy_tokenizer
    if copy_tokenizer:
        assert max(AutoTokenizer.from_pretrained(output).get_vocab().values()) == 9
        assert (
            json.loads((output / "generation_config.json").read_text())
            == generation_config
        )


def test_embedding_and_head_are_truncated_independently(models, tmp_path):
    # Legacy truncation is per tensor, not a model-wide vocabulary-size decision.
    # The inconsistent result is intentional here, not a valid-model example.
    path = f"{models[0]}/model.safetensors"
    weights = load_file(path)
    weights["lm_head.weight"] = weights["lm_head.weight"][:7].clone()
    save_file(weights, path)
    output = tmp_path / "output"
    run_merge(
        config_for(models),
        str(output),
        MergeOptions(unsafe_truncate_embeddings=True, quiet=True),
    )
    loader = LazyTensorLoader.from_disk(str(output))
    assert loader.get_tensor("model.embed_tokens.weight").shape == (8, 32)
    assert loader.get_tensor("lm_head.weight").shape == (7, 32)


def test_missing_tokenizer_does_not_prevent_truncation(tmp_path):
    models = (
        make_picollama(tmp_path / "small", vocab_size=8),
        make_picollama(tmp_path / "large", vocab_size=10),
    )
    output = tmp_path / "output"
    run_merge(
        config_for(models),
        str(output),
        MergeOptions(unsafe_truncate_embeddings=True, quiet=True),
    )
    assert (output / "config.json").exists()
    assert not (output / "tokenizer.json").exists()


def test_cli_exposes_dangerous_opt_in():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    dangerous_options = result.output.split("Dangerous Options:")[1]
    assert "--unsafe-truncate-embeddings" in dangerous_options
    assert "UNSAFE" in dangerous_options
    assert MergeOptions().unsafe_truncate_embeddings is False
