from __future__ import annotations

import pytest
import torch

from mergekit.config import MergeConfiguration
from mergekit.merge_methods import (
    BasePolicy,
    InputContract,
    MergeBatch,
    PerGroupValues,
    PerInput,
    Shared,
    TensorEntry,
    TensorGroup,
    merge_state_dicts,
)
from mergekit.merge_methods.base import method_from_function
from mergekit.scripts.merge_raw_pytorch import (
    InputModelDefinition as RawInputModelDefinition,
)
from mergekit.scripts.merge_raw_pytorch import (
    RawPyTorchMergeConfig,
    construct_param_dicts,
)


def test_signature_is_parameter_ssot_and_supports_shared_lists():
    def kernel(
        group: TensorGroup,
        weight: PerInput[float],
        offsets: Shared[list[float]],
        normalize: Shared[bool] = True,
    ) -> torch.Tensor:
        values = weight.values_for(group.entries)
        result = sum(
            entry.tensor * value for entry, value in zip(group.entries, values)
        )
        if normalize:
            result /= sum(values)
        return result + torch.tensor(offsets)

    method = method_from_function(kernel, name="test_method")
    assert [parameter.name for parameter in method.parameters()] == [
        "offsets",
        "normalize",
    ]
    assert [parameter.name for parameter in method.tensor_parameters()] == ["weight"]

    batch = MergeBatch.from_tensors(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], ids=["a", "b"]
    )
    result = method(
        batch,
        weight={"b": 0.75, "a": 0.25},
        offsets=[10.0, 20.0],
    ).one()
    assert torch.equal(result, torch.tensor([12.5, 23.5]))


def test_contract_validates_entire_batch_before_math_runs():
    calls = []

    def kernel(group: TensorGroup) -> torch.Tensor:
        calls.append(group)
        return group.entries[0].tensor

    method = method_from_function(
        kernel,
        name="requires_base",
        contract=InputContract(
            base=BasePolicy.REQUIRED,
            min_inputs=2,
            max_inputs=2,
        ),
    )
    valid = TensorGroup(
        entries=(
            TensorEntry("base", torch.tensor(1.0), is_base=True),
            TensorEntry("other", torch.tensor(2.0)),
        )
    )
    invalid = TensorGroup(
        entries=(
            TensorEntry("a", torch.tensor(1.0)),
            TensorEntry("b", torch.tensor(2.0)),
        )
    )

    with pytest.raises(ValueError, match="requires a base input"):
        method(MergeBatch(groups=(valid, invalid)))
    assert calls == []


def test_per_input_values_are_ordered_and_shape_checked():
    def kernel(group: TensorGroup, value: PerInput[int]) -> torch.Tensor:
        return torch.tensor(value.values_for(group.entries))

    method = method_from_function(kernel, name="ordered")
    batch = MergeBatch.from_tensors(
        [torch.tensor(0), torch.tensor(0)], ids=["second", "first"]
    )
    assert torch.equal(
        method(batch, value={"first": 1, "second": 2}).one(),
        torch.tensor([2, 1]),
    )
    with pytest.raises(ValueError, match="expects 2 values"):
        method(batch, value=[1])


def test_registered_method_can_be_called_directly():
    from mergekit import merge_methods

    batch = MergeBatch.from_tensors(
        [torch.tensor([1.0]), torch.tensor([3.0])], ids=["a", "b"]
    )
    result = merge_methods.get("linear")(batch, weight={"a": 0.25, "b": 0.75}).one()
    assert torch.equal(result, torch.tensor([2.5]))


def test_explicit_per_group_parameter_values():
    def kernel(group: TensorGroup, scale: Shared[float]) -> torch.Tensor:
        return group.entries[0].tensor * scale

    method = method_from_function(kernel, name="per_group")
    batch = MergeBatch(
        groups=(
            TensorGroup(entries=(TensorEntry("a", torch.tensor(2.0)),)),
            TensorGroup(entries=(TensorEntry("a", torch.tensor(3.0)),)),
        )
    )
    result = method(batch, scale=PerGroupValues([5.0, 7.0]))
    assert result.tensors == (torch.tensor(10.0), torch.tensor(21.0))


def test_merge_state_dicts_programmatic_api():
    result = merge_state_dicts(
        {
            "a": {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])},
            "b": {"x": torch.tensor([3.0]), "y": torch.tensor([6.0])},
        },
        "linear",
        parameters={"weight": {"a": 0.25, "b": 0.75}},
    )
    assert torch.equal(result["x"], torch.tensor([2.5]))
    assert torch.equal(result["y"], torch.tensor([5.0]))


def test_merge_state_dicts_validates_keys_before_merging():
    with pytest.raises(ValueError, match="different tensor keys"):
        merge_state_dicts(
            {
                "a": {"x": torch.tensor([1.0])},
                "b": {"y": torch.tensor([2.0])},
            },
            "linear",
            parameters={"weight": [0.5, 0.5]},
        )


def test_config_parameter_carrier_preserves_scalar_types():
    config = MergeConfiguration.model_validate(
        {
            "merge_method": "test",
            "models": [{"model": "model_a"}],
            "parameters": {"enabled": True, "count": 2, "mode": "sum"},
        }
    )
    assert config.parameters == {"enabled": True, "count": 2, "mode": "sum"}


def test_raw_parameter_binding_preserves_zero_values():
    from mergekit import merge_methods

    config = RawPyTorchMergeConfig(
        merge_method="linear",
        models=[
            RawInputModelDefinition(model="a", parameters={"weight": 0.0}),
            RawInputModelDefinition(model="b", parameters={"weight": 1.0}),
        ],
    )
    _, input_parameters = construct_param_dicts(
        config, merge_methods.get("linear"), "weight"
    )
    assert sorted(values["weight"] for values in input_parameters.values()) == [
        0.0,
        1.0,
    ]
