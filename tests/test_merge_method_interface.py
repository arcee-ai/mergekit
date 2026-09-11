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
from mergekit.merge_methods.easy_define import from_group_kernel
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

    method = from_group_kernel(kernel, name="test_method")
    assert [parameter.name for parameter in method.spec.shared_parameters] == [
        "offsets",
        "normalize",
    ]
    assert [parameter.name for parameter in method.spec.input_parameters] == ["weight"]

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

    method = from_group_kernel(
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

    method = from_group_kernel(kernel, name="ordered")
    batch = MergeBatch.from_tensors(
        [torch.zeros(2), torch.zeros(2)], ids=["second", "first"]
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

    method = from_group_kernel(kernel, name="per_group")
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


@pytest.mark.parametrize("failure", ["shape", "dtype", "device"])
def test_group_adapter_validates_every_group_before_execution(failure):
    calls = []

    def kernel(group: TensorGroup) -> torch.Tensor:
        calls.append(group)
        return group.entries[0].tensor

    method = from_group_kernel(kernel, name="validated")
    good = MergeBatch.from_tensors([torch.ones(2, 1), torch.ones(2, 1)]).groups[0]
    bad_tensor = {
        "shape": lambda: torch.ones(1, 2),
        "dtype": lambda: torch.ones(2, 1, dtype=torch.float64),
        "device": lambda: torch.ones(2, 1, device="meta"),
    }[failure]()
    bad = MergeBatch.from_tensors([torch.ones(2, 1), bad_tensor]).groups[0]
    with pytest.raises(ValueError, match="size mismatch|same dtype and device"):
        method(MergeBatch((good, bad)))
    assert calls == []


def test_state_dict_merge_rejects_broadcastable_weights():
    with pytest.raises(ValueError, match="Tensor size mismatch for w"):
        merge_state_dicts(
            {"base": {"w": torch.ones(2, 1)}, "other": {"w": torch.ones(1, 2)}},
            "nearswap",
            base="base",
            parameters={"t": 0.5},
        )


@pytest.mark.parametrize(
    "method_name,count",
    [
        ("linear", 2),
        ("slerp", 2),
        ("nuslerp", 3),
        ("arcee_fusion", 2),
        ("model_stock", 3),
        ("karcher", 2),
        ("task_arithmetic", 2),
        ("ties", 2),
        ("nearswap", 2),
        ("multislerp", 2),
        ("sce", 2),
        ("ram", 2),
        ("ramplus_tl", 2),
    ],
)
@pytest.mark.parametrize("shape", [(3, 2), (2, 3)])
def test_no_method_crops_embeddings(method_name, count, shape):
    from mergekit import merge_methods
    from mergekit.merge_methods import TensorMetadata

    entries = tuple(
        TensorEntry(i, torch.ones((2, 2) if i == 0 else shape), is_base=i == 0)
        for i in range(count)
    )
    group = TensorGroup(
        entries, TensorMetadata(name="embed_tokens.weight", is_embed=True)
    )
    with pytest.raises(ValueError, match="Tensor size mismatch.*tokenizer"):
        merge_methods.get(method_name)(MergeBatch((group,)))


def test_group_kernel_must_preserve_weight_shape():
    def kernel(group: TensorGroup) -> torch.Tensor:
        return group.entries[0].tensor.sum()

    method = from_group_kernel(kernel, name="invalid_reduction")
    with pytest.raises(TypeError, match="must return a tensor of shape"):
        method(MergeBatch.from_tensors([torch.ones(2)]))


@pytest.mark.parametrize("filter_wise", [False, True])
def test_model_stock_singularity_retains_base_with_finite_gradients(filter_wise):
    from mergekit import merge_methods

    base = torch.tensor([[2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[3.0, 3.0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[1.0, 3.0]], dtype=torch.float64, requires_grad=True)
    result = merge_methods.get("model_stock")(
        MergeBatch.from_tensors([base, a, b], base_index=0), filter_wise=filter_wise
    ).one()
    torch.testing.assert_close(result, base)
    result.sum().backward()
    torch.testing.assert_close(base.grad, torch.ones_like(base))
    for tensor in (a, b):
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
def test_sce_zero_variance_preserves_shape(shape, density):
    from mergekit import merge_methods

    base = torch.ones(shape)
    other = torch.full(shape, 3.0)
    result = merge_methods.get("sce")(
        MergeBatch.from_tensors([base, other, other], base_index=0), select_topk=density
    ).one()
    torch.testing.assert_close(result, other if density == 1 else base)
