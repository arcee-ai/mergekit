from __future__ import annotations

import inspect
from typing import Annotated, Literal

import pytest
import torch
from pydantic import Field, ValidationError

from mergekit.config import MergeConfiguration
from mergekit.merge_methods import (
    BasePolicy,
    BatchParameter,
    InputContract,
    MergeBatch,
    PerGroupValues,
    PerInput,
    PerInputValues,
    TensorBatch,
    TensorEntry,
    TensorGroup,
    merge_state_dicts,
    merge_tensors,
)
from mergekit.merge_methods.easy_define import merge_method
from mergekit.scripts.merge_raw_pytorch import (
    InputModelDefinition as RawInputModelDefinition,
)
from mergekit.scripts.merge_raw_pytorch import (
    RawPyTorchMergeConfig,
    construct_param_dicts,
)


@pytest.mark.parametrize("weight", [[0.25, 0.75], {"b": 0.75, "a": 0.25}])
def test_merge_tensors_aligns_parameters_and_preserves_gradients(weight):
    a = torch.tensor([1.0, 2.0], dtype=torch.float16, requires_grad=True)
    b = torch.tensor([3.0, 6.0], dtype=torch.bfloat16, requires_grad=True)
    result = merge_tensors(
        [a, b],
        "linear",
        ids=["a", "b"],
        parameters={"weight": weight},
        dtype=torch.float64,
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(result, torch.tensor([2.5, 5.0]))
    result.sum().backward()
    torch.testing.assert_close(a.grad, torch.full_like(a, 0.25))
    torch.testing.assert_close(b.grad, torch.full_like(b, 0.75))


def test_merge_tensors_base_index_refers_to_input_order():
    result = merge_tensors(
        [torch.tensor([3.0]), torch.tensor([1.0])],
        "task_arithmetic",
        ids=["other", "base"],
        base_index=1,
        parameters={"weight": {"other": 0.25}},
    )
    torch.testing.assert_close(result, torch.tensor([1.5]))
    with pytest.raises(ValueError, match="requires a base input"):
        merge_tensors([torch.ones(1)], "task_arithmetic", parameters={"weight": 1.0})
    with pytest.raises(ValueError, match="size mismatch for projection"):
        merge_tensors(
            [torch.ones(2), torch.ones(3)],
            "linear",
            name="projection",
            parameters={"weight": 1.0},
        )


@pytest.mark.parametrize("input_type", [TensorGroup, TensorBatch])
def test_plain_annotations_preserve_constraints_defaults_and_requiredness(input_type):
    def kernel(
        inputs,
        scale: Annotated[float, Field(gt=0)],
        mode: Literal["add", "multiply"] = "multiply",
    ) -> torch.Tensor:
        return (
            inputs.tensors[0] * scale
            if mode == "multiply"
            else inputs.tensors[0] + scale
        )

    kernel.__annotations__["inputs"] = input_type
    method = merge_method(kernel, name="plain_parameters")
    source = torch.ones(2)
    torch.testing.assert_close(
        merge_tensors([source], method, parameters={"scale": 3}), torch.full((2,), 3.0)
    )
    torch.testing.assert_close(
        merge_tensors([source], method, parameters={"scale": 3, "mode": "add"}),
        torch.full((2,), 4.0),
    )
    with pytest.raises(TypeError, match="Missing required parameter scale"):
        merge_tensors([source], method)
    for parameters in ({"scale": 0}, {"scale": 1, "mode": "unknown"}):
        with pytest.raises(ValidationError):
            merge_tensors([source], method, parameters=parameters)


@pytest.mark.parametrize(
    "input_type,annotation,message",
    [
        (TensorGroup, inspect.Parameter.empty, "must have a type annotation"),
        (TensorBatch, torch.Tensor, "must be annotated with BatchParameter"),
        (TensorGroup, PerInputValues[float], "must use PerInput"),
        (TensorGroup, Annotated[torch.Tensor, BatchParameter(float)], "Group kernels"),
        (
            TensorBatch,
            Annotated[torch.Tensor, BatchParameter(float), BatchParameter(int)],
            "at most one scope",
        ),
    ],
)
def test_signature_rejects_missing_or_ambiguous_parameter_types(
    input_type, annotation, message
):
    def kernel(inputs, value) -> torch.Tensor:
        return inputs.tensors[0]

    kernel.__annotations__["inputs"] = input_type
    if annotation is not inspect.Parameter.empty:
        kernel.__annotations__["value"] = annotation
    with pytest.raises(TypeError, match=message):
        merge_method(kernel, name="invalid_signature")


def test_signature_is_parameter_ssot_and_supports_shared_lists():
    def kernel(
        group: TensorGroup,
        weight: PerInput[float],
        offsets: list[float],
        normalize: bool = True,
    ) -> torch.Tensor:
        values = weight.values_for(group.entries)
        result = sum(
            entry.tensor * value for entry, value in zip(group.entries, values)
        )
        if normalize:
            result /= sum(values)
        return result + torch.tensor(offsets)

    method = merge_method(kernel, name="test_method")
    assert [parameter.name for parameter in method.spec.shared_parameters] == [
        "offsets",
        "normalize",
    ]
    assert [parameter.name for parameter in method.spec.input_parameters] == ["weight"]

    batch = MergeBatch.from_tensors(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], ids=["a", "b"]
    )
    (result,) = method(
        batch, parameters={"weight": {"b": 0.75, "a": 0.25}, "offsets": [10.0, 20.0]}
    )
    assert torch.equal(result, torch.tensor([12.5, 23.5]))


def test_contract_validates_entire_batch_before_math_runs():
    calls = []

    def kernel(group: TensorGroup) -> torch.Tensor:
        calls.append(group)
        return group.entries[0].tensor

    method = merge_method(
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


@pytest.mark.parametrize("values_type", [dict, PerInputValues])
def test_per_input_values_are_ordered_and_shape_checked(values_type):
    def kernel(group: TensorGroup, value: PerInput[int]) -> torch.Tensor:
        assert list(value) == ["second", "first"]
        return torch.tensor(value.values_for(group.entries))

    method = merge_method(kernel, name="ordered")
    batch = MergeBatch.from_tensors(
        [torch.zeros(2), torch.zeros(2)], ids=["second", "first"]
    )
    (result,) = method(
        batch, parameters={"value": values_type([("first", 1), ("second", 2)])}
    )
    assert torch.equal(result, torch.tensor([2, 1]))
    with pytest.raises(ValueError, match="Missing value for input"):
        method(batch, parameters={"value": values_type([("first", 1)])})
    with pytest.raises(ValueError, match="expects 2 values"):
        method(batch, parameters={"value": [1]})


def test_registered_method_can_be_called_directly():
    from mergekit import merge_methods

    batch = MergeBatch.from_tensors(
        [torch.tensor([1.0]), torch.tensor([3.0])], ids=["a", "b"]
    )
    (result,) = merge_methods.get("linear")(
        batch, parameters={"weight": {"a": 0.25, "b": 0.75}}
    )
    assert torch.equal(result, torch.tensor([2.5]))


def test_explicit_per_group_parameter_values():
    def kernel(group: TensorGroup, scale: float) -> torch.Tensor:
        return group.entries[0].tensor * scale

    method = merge_method(kernel, name="per_group")
    batch = MergeBatch(
        groups=(
            TensorGroup(entries=(TensorEntry("a", torch.tensor(2.0)),)),
            TensorGroup(entries=(TensorEntry("a", torch.tensor(3.0)),)),
        )
    )
    result = method(batch, parameters={"scale": PerGroupValues([5.0, 7.0])})
    assert result == (torch.tensor(10.0), torch.tensor(21.0))


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


@pytest.mark.parametrize("failure", ["shape", "device"])
def test_group_adapter_validates_every_group_before_execution(failure):
    calls = []

    def kernel(group: TensorGroup) -> torch.Tensor:
        calls.append(group)
        return group.entries[0].tensor

    method = merge_method(kernel, name="validated")
    good = MergeBatch.from_tensors([torch.ones(2, 1), torch.ones(2, 1)]).groups[0]
    bad_tensor = {
        "shape": lambda: torch.ones(1, 2),
        "device": lambda: torch.ones(2, 1, device="meta"),
    }[failure]()
    bad = MergeBatch.from_tensors([torch.ones(2, 1), bad_tensor]).groups[0]
    with pytest.raises(ValueError, match="size mismatch|same device"):
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

    method = merge_method(kernel, name="invalid_reduction")
    with pytest.raises(TypeError, match="must return a tensor of shape"):
        method(MergeBatch.from_tensors([torch.ones(2)]))


@pytest.mark.parametrize("filter_wise", [False, True])
def test_model_stock_singularity_retains_base_with_finite_gradients(filter_wise):
    from mergekit import merge_methods

    base = torch.tensor([[2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[3.0, 3.0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[1.0, 3.0]], dtype=torch.float64, requires_grad=True)
    (result,) = merge_methods.get("model_stock")(
        MergeBatch.from_tensors([base, a, b], base_index=0),
        parameters={"filter_wise": filter_wise},
    )
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
    (result,) = merge_methods.get("sce")(
        MergeBatch.from_tensors([base, other, other], base_index=0),
        parameters={"select_topk": density},
    )
    torch.testing.assert_close(result, other if density == 1 else base)


def test_none_input_ids_are_rejected_at_every_entry_point():
    tensor = torch.ones(2)
    with pytest.raises(ValueError, match="IDs cannot be None"):
        MergeBatch.from_tensors([tensor], ids=[None], base_index=0)
    with pytest.raises(ValueError, match="IDs cannot be None"):
        InputContract().validate_ids([None, "b"])
    with pytest.raises(ValueError, match="IDs cannot be None"):
        merge_state_dicts(
            {None: {"w": tensor}, "b": {"w": tensor}},
            "nuslerp",
            parameters={"weight": [0.5, 0.5]},
        )
    with pytest.raises(ValueError, match="IDs cannot be None"):
        merge_state_dicts({None: {"counter": torch.tensor(0)}}, "passthrough")


@pytest.mark.parametrize("scale", [None, 2.0])
def test_passthrough_graph_does_not_visit_math_device(tmp_path, scale):
    from safetensors.torch import load_file, save_file

    from mergekit.graph import Executor
    from mergekit.options import MergeOptions
    from mergekit.scripts.merge_raw_pytorch import plan_flat_merge

    source = tmp_path / "source.safetensors"
    output = tmp_path / "output"
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    save_file({"w": tensor}, source)
    config = RawPyTorchMergeConfig(
        merge_method="passthrough",
        models=[{"model": str(source)}],
        parameters={} if scale is None else {"scale": scale},
    )
    tasks = plan_flat_merge(config, str(output), False, False, MergeOptions())
    # A transfer to meta followed by a return to CPU would fail. This exercises
    # actual scheduling and saving without requiring an available accelerator.
    Executor(tasks, math_device="meta", storage_device="cpu").execute()
    actual = load_file(output / "model.safetensors")["w"]
    torch.testing.assert_close(actual, tensor if scale is None else tensor * scale)


def test_construction_and_registration_are_independent(monkeypatch):
    from mergekit.merge_methods import get, register, registered_methods, registry

    builtin = get("linear")

    @merge_method(name="linear")
    def custom(group: TensorGroup) -> torch.Tensor:
        return group.tensors[0]

    assert get("linear") is builtin
    batch = MergeBatch.from_tensors([torch.ones(2)])
    (result,) = custom(batch)
    torch.testing.assert_close(result, torch.ones(2))
    with pytest.raises(ValueError, match="already registered"):
        register(custom)
    assert get("linear") is builtin

    monkeypatch.setattr(registry, "_METHODS", {})
    register(custom)
    assert get("linear") is custom
    assert registered_methods() == (custom,)
    with pytest.raises(RuntimeError, match="Unimplemented merge method missing"):
        get("missing")


def test_execution_controls_do_not_reserve_algorithm_parameter_names():
    @merge_method(name="controls")
    def kernel(
        group: TensorGroup,
        dtype: float,
        out_dtype: float,
        batch_options: float,
        parameters: float,
    ) -> torch.Tensor:
        return group.tensors[0] * (dtype + out_dtype + batch_options + parameters)

    (result,) = kernel(
        MergeBatch.from_tensors([torch.ones(2)]),
        parameters={"dtype": 1, "out_dtype": 2, "batch_options": 3, "parameters": 4},
        dtype=torch.float64,
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(result, torch.full((2,), 10.0))


@pytest.mark.parametrize("batched", [False, True])
def test_kernel_wrappers_do_not_reserve_algorithm_parameter_names(batched):
    if batched:

        def kernel(
            inputs: TensorBatch,
            self: float,
            batch: float,
            group: float,
        ) -> torch.Tensor:
            return inputs.tensors[0] * (self + batch + group)

    else:

        def kernel(
            inputs: TensorGroup,
            self: float,
            batch: float,
            group: float,
        ) -> torch.Tensor:
            return inputs.tensors[0] * (self + batch + group)

    method = merge_method(kernel, name="wrapper_names")
    (result,) = method(
        MergeBatch.from_tensors([torch.ones(2)]),
        parameters={"self": 2, "batch": 3, "group": 5},
    )
    torch.testing.assert_close(result, torch.full((2,), 10.0))


def test_all_builtins_are_registered():
    from mergekit.merge_methods import get, registered_methods

    names = {
        "linear",
        "slerp",
        "nuslerp",
        "multislerp",
        "passthrough",
        "model_stock",
        "arcee_fusion",
        "karcher",
        "nearswap",
        "ram",
        "ramplus_tl",
        "sce",
        "task_arithmetic",
        "ties",
        "dare_ties",
        "dare_linear",
        "breadcrumbs",
        "breadcrumbs_ties",
        "della",
        "della_linear",
    }
    assert {method.spec.name for method in registered_methods()} == names
    for method in registered_methods():
        assert get(method.spec.name) is method
