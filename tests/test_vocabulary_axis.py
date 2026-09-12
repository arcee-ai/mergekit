import json

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
    PhiConfig,
    PhiForCausalLM,
    T5Config,
)

from mergekit.architecture import (
    ConfiguredModelArchitecture,
    WeightInfo,
    arch_info_for_config,
)
from mergekit.architecture.auto import get_transformers_info, infer_vocabulary_axes
from mergekit.architecture.json_definitions import NAME_TO_ARCH
from mergekit.common import ImmutableMap, ModelReference
from mergekit.config import MergeConfiguration
from mergekit.io.tasks import GatherTensors
from mergekit.merge import run_merge
from mergekit.merge_methods import TensorGroup
from mergekit.merge_methods.base import TensorMetadata
from mergekit.merge_methods.preprocessing import truncate_vocabulary
from mergekit.options import MergeOptions
from mergekit.scripts.tokensurgeon import get_embedding_info, remap_auxiliary_vocabulary
from mergekit.tokenizer import BuildTokenizer, PermutedVocabulary, TokenizerInfo
from mergekit.tokenizer.config import ModelTokenEmbedding, TokenEmbeddingConfig
from tests.test_tokenizer import make_tokenizer


def permutation_task(axis, *, base_model=None, pad_to_multiple_of=None):
    model = ModelReference.parse("test-model")
    task = PermutedVocabulary(
        gather_tensors=GatherTensors(weight_info=ImmutableMap({})),
        tokenizer_task=BuildTokenizer(
            base_model=None,
            referenced_models=(model,),
            tokenizer_source="union",
            add_tokens=None,
        ),
        tokens=None,
        base_model=base_model,
        pad_to_multiple_of=pad_to_multiple_of,
        vocabulary_axis=axis,
    )
    info = TokenizerInfo(
        tokenizer=make_tokenizer(5, []),
        permutations={model: dict(enumerate([2, 0, 1, 3, -1]))},
        original_vocabs={model: {}},
    )
    return model, task, info


@pytest.mark.parametrize(
    "shape,axis", [((4,), 0), ((4, 2), 0), ((2, 4), 1), ((2, 3, 4), 2)]
)
def test_alignment_preserves_layout_and_gradients(shape, axis):
    model, task, info = permutation_task(axis, pad_to_multiple_of=8)
    source = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    source = source.reshape(shape).requires_grad_()
    result = task.execute(info, {model: source})[model]
    front = source.movedim(axis, 0)
    expected = torch.cat([front[[2, 0, 1, 3]], front.new_zeros((1, *front.shape[1:]))])
    expected = torch.cat(
        [expected, expected.mean(dim=0, keepdim=True).expand(3, *front.shape[1:])]
    )
    torch.testing.assert_close(result, expected.movedim(0, axis))
    result.sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, 1.6))


def test_alignment_allows_optional_weight_missing_from_base():
    model, task, info = permutation_task(
        0, base_model=ModelReference.parse("absent-base")
    )
    result = task.execute(info, {model: torch.arange(4.0)})
    torch.testing.assert_close(result[model], torch.tensor([2.0, 0.0, 1.0, 3.0, 0.0]))
    assert task.execute(info, {}) == {}


@pytest.mark.parametrize("explicit_token", [False, True])
@pytest.mark.parametrize("need_default", ["unused", "missing_token", "forced"])
def test_alignment_resolves_absent_sources_only_when_needed(
    explicit_token, need_default
):
    base = ModelReference.parse("absent-base")
    model, task, info = permutation_task(0, base_model=base)
    token = "_tok_3"
    source = (
        ModelTokenEmbedding(kind="model_token", model=base, token_id=3)
        if explicit_token
        else base
    )
    task = task.model_copy(
        update={
            "tokens": ImmutableMap(
                {
                    token: TokenEmbeddingConfig(
                        source=source, force=need_default == "forced"
                    )
                }
            )
        }
    )
    info.permutations[base] = dict(enumerate(range(5)))
    if need_default == "missing_token":
        info.permutations[model][3] = -1
    tensors = {model: torch.arange(4.0)}
    if need_default == "unused":
        result = task.execute(info, tensors)
        torch.testing.assert_close(
            result[model], torch.tensor([2.0, 0.0, 1.0, 3.0, 0.0])
        )
    else:
        with pytest.raises(ValueError, match="_tok_3.*absent-base.*weight is missing"):
            task.execute(info, tensors)


@pytest.mark.parametrize(
    "shape,axis", [((6,), 0), ((6, 2), 0), ((2, 6), 1), ((2, 3, 6), 2)]
)
def test_truncation_borrows_slices_on_vocabulary_axis(shape, axis):
    source = torch.randn(shape, requires_grad=True)
    smaller = source.detach().narrow(axis, 0, 4).clone()
    group = TensorGroup.from_tensors(
        [source, smaller], vocabulary_axis=axis, base_index=0
    )
    result = truncate_vocabulary(group)
    assert result.metadata == group.metadata
    assert result.base.id == group.base.id
    assert result.entries[0].tensor.data_ptr() == source.data_ptr()
    torch.testing.assert_close(result.entries[0].tensor, source.narrow(axis, 0, 4))
    result.entries[0].tensor.sum().backward()
    assert torch.count_nonzero(source.grad.narrow(axis, 4, 2)) == 0


@pytest.mark.parametrize("axis", [-1, True, 1.5])
def test_metadata_rejects_invalid_axes(axis):
    with pytest.raises(ValueError):
        WeightInfo(name="test", vocabulary_axis=axis)
    with pytest.raises(ValueError):
        TensorMetadata(vocabulary_axis=axis)


@pytest.mark.parametrize("extra", [{"is_embed": True}, {"vocab_axis": 0}])
def test_unknown_weight_metadata_is_rejected_instead_of_silently_ignored(extra):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        WeightInfo(name="test", **extra)


@pytest.mark.parametrize("transform", ["align", "truncate"])
def test_transform_rejects_axis_outside_tensor_rank(transform):
    if transform == "align":
        model, task, info = permutation_task(1)
        with pytest.raises(ValueError, match="Invalid vocabulary_axis"):
            task.execute(info, {model: torch.ones(4)})
    else:
        with pytest.raises(ValueError, match="Invalid vocabulary_axis"):
            truncate_vocabulary(
                TensorGroup.from_tensors([torch.ones(4)], vocabulary_axis=1)
            )


class ModelWithOtherEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = torch.nn.Embedding(8, 4)
        self.positions = torch.nn.Embedding(8, 4)
        self.token_types = torch.nn.Embedding(2, 4)
        self.output = torch.nn.Linear(4, 8)
        self.output.weight = self.tokens.weight
        self.bias_alias = self.output.bias
        self.norm = torch.nn.LayerNorm(4)
        self.norm_alias = self.norm

    def get_input_embeddings(self):
        return self.tokens

    def get_output_embeddings(self):
        return self.output


def test_inference_marks_only_vocabulary_parameters_and_their_actual_aliases():
    with torch.device("meta"):
        model = ModelWithOtherEmbeddings()
    assert infer_vocabulary_axes(model) == {
        "tokens.weight": 0,
        "output.weight": 0,
        "output.bias": 0,
        "bias_alias": 0,
    }


@pytest.mark.parametrize(
    "config,expected_names",
    [
        pytest.param(
            T5Config(
                architectures=["T5ForConditionalGeneration"],
                vocab_size=8,
                d_model=4,
                d_ff=8,
                num_layers=1,
                num_heads=1,
            ),
            {
                "shared.weight",
                "encoder.embed_tokens.weight",
                "decoder.embed_tokens.weight",
                "lm_head.weight",
            },
            id="t5-embedding-aliases",
        ),
        pytest.param(
            BertConfig(
                architectures=["BertForMaskedLM"],
                vocab_size=8,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=1,
                num_attention_heads=1,
            ),
            {
                "bert.embeddings.word_embeddings.weight",
                "cls.predictions.bias",
                "cls.predictions.decoder.weight",
                "cls.predictions.decoder.bias",
            },
            id="bert-bias-aliases",
        ),
    ],
)
def test_transformers_inference_detects_tied_vocabulary_aliases(
    tmp_path, config, expected_names
):
    # Config-only checkpoints exercise the production meta/no-init construction.
    config.save_pretrained(tmp_path)
    _, _, axes, _ = get_transformers_info(
        ModelReference.parse(str(tmp_path)), MergeOptions()
    )
    assert axes == {name: 0 for name in expected_names}


@pytest.mark.parametrize(
    "arch_name",
    [
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Lfm2ForCausalLM",
    ],
)
def test_static_metadata_excludes_vision_and_normalization(arch_name):
    for arch in NAME_TO_ARCH[arch_name]:
        for module in arch.modules.values():
            definition = module.architecture.definition
            weights = (
                definition.pre_weights
                + definition.post_weights
                + definition.layer_templates.weights
            )
            for weight in weights:
                if "visual." in weight.name or "embedding_norm" in weight.name:
                    assert weight.vocabulary_axis is None


def make_phi(path, size, *, swapped=False):
    model = PhiForCausalLM(
        PhiConfig(
            vocab_size=size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    with torch.no_grad():
        model.lm_head.bias.copy_(torch.arange(size, dtype=torch.float32))
    model.save_pretrained(path)
    make_tokenizer(size, []).save_pretrained(path)
    if swapped:
        tokenizer_path = path / "tokenizer.json"
        data = json.loads(tokenizer_path.read_text())
        vocab = data["model"]["vocab"]
        vocab["_tok_3"], vocab["_tok_4"] = vocab["_tok_4"], vocab["_tok_3"]
        tokenizer_path.write_text(json.dumps(data))
    return model


@pytest.mark.parametrize(
    "source,other_size", [("base", 8), ("base", 10), ("union", 10)]
)
def test_phi_alignment_reorders_and_resizes_biases(tmp_path, source, other_size):
    paths = [tmp_path / "base", tmp_path / "other"]
    originals = [make_phi(paths[0], 8), make_phi(paths[1], other_size, swapped=True)]
    config = MergeConfiguration(
        merge_method="linear",
        base_model=str(paths[0]),
        models=[{"model": str(p)} for p in paths],
        parameters={"weight": 0.5},
        tokenizer_source=source,
    )
    output = tmp_path / "output"
    run_merge(config, str(output), MergeOptions(quiet=True))
    merged = AutoModelForCausalLM.from_pretrained(output)
    vocabs = [AutoTokenizer.from_pretrained(p).get_vocab() for p in paths]
    target = AutoTokenizer.from_pretrained(output).get_vocab()
    assert merged.config.vocab_size == len(target)
    for name in ["model.embed_tokens.weight", "lm_head.weight", "lm_head.bias"]:
        actual = merged.state_dict()[name]
        assert actual.shape[0] == len(target)
        for token, idx in target.items():
            values = [
                m.state_dict()[name][v[token]]
                for m, v in zip(originals, vocabs)
                if token in v
            ]
            torch.testing.assert_close(actual[idx], torch.stack(values).mean(dim=0))
    torch.testing.assert_close(merged.lm_head.bias[3:5], torch.tensor([3.5, 3.5]))
    with torch.no_grad():
        assert merged(torch.tensor([[1, 3, 2]])).logits.shape[-1] == len(target)


def test_phi_truncation_handles_output_bias(tmp_path):
    paths = [tmp_path / "small", tmp_path / "large"]
    make_phi(paths[0], 8)
    make_phi(paths[1], 10)
    config = MergeConfiguration(
        merge_method="linear",
        base_model=str(paths[0]),
        models=[{"model": str(p)} for p in paths],
        parameters={"weight": 0.5},
    )
    output = tmp_path / "output"
    run_merge(
        config, str(output), MergeOptions(quiet=True, unsafe_truncate_embeddings=True)
    )
    merged = AutoModelForCausalLM.from_pretrained(output)
    torch.testing.assert_close(merged.lm_head.bias, torch.arange(8.0))
    assert merged.config.vocab_size == 8


def test_tokensurgeon_distinguishes_matrices_from_bias_and_remaps_auxiliaries():
    cfg = PhiConfig(architectures=["PhiForCausalLM"])
    arch = ConfiguredModelArchitecture(info=arch_info_for_config(cfg), config=cfg)
    embed, head = get_embedding_info(arch)
    assert embed.name == "model.embed_tokens.weight"
    assert head.name == "lm_head.weight"
    result = remap_auxiliary_vocabulary(
        torch.tensor([[1.0, 2.0, 3.0]]),
        WeightInfo(name="bias", vocabulary_axis=1),
        {"a": 0, "b": 1, "c": 2},
        {"c": 0, "a": 1, "new": 2},
    )
    torch.testing.assert_close(result, torch.tensor([[3.0, 1.0, 0.0]]))
