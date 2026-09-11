from dataclasses import replace
from typing import Annotated

import pytest
import torch
import yaml
from pydantic import Field, TypeAdapter, ValidationError

from mergekit import merge_methods
from mergekit.config import ConfigReader, MergeConfiguration, evaluate_setting
from mergekit.merge_methods import PerInput, Shared, TensorGroup, merge_method
from mergekit.parameter_resolver import resolve_parameter, resolve_parameters
from mergekit.scripts.merge_raw_pytorch import (
    RawPyTorchMergeConfig,
    construct_param_dicts,
)


@pytest.mark.parametrize("level", ["global", "module", "slice", "input"])
def test_scientific_notation_gradients_at_every_config_level(level):
    setting = yaml.safe_load("[1e-5, 1e-3]")
    source = {"model": "a", "layer_range": [0, 3]}
    output_slice = {"sources": [source]}
    module = {"slices": [output_slice]}
    raw = {"merge_method": "linear", "modules": {"model": module}}
    {"global": raw, "module": module, "slice": output_slice, "input": source}[level][
        "parameters"
    ] = {"weight": [{"filter": "w", "value": setting}]}
    config = MergeConfiguration.model_validate(raw)
    module = config.modules["model"]
    output_slice = module.slices[0]
    reader = ConfigReader(config=config, module=module, slice_out=output_slice, t=0.5)
    parameter = merge_methods.get("linear").spec.input_parameters[0]
    assert resolve_parameter(
        parameter,
        reader.parameter_sources(output_slice.sources[0].model),
        tensor_name="w",
        t=reader.t,
    ) == pytest.approx(0.000505)


@pytest.mark.parametrize(
    "value_type,setting,expected",
    [
        (float, "[1e-5, 1e-3]", 0.000505),
        (float, "[1e-5, 0.001]", 0.000505),
        (str, "[1e-5, 1e-3]", "1e-5"),
        (bool, "[false, true]", False),
        (bool, "[0, 1]", False),
        (int, "[2, 4]", 3),
    ],
)
def test_gradient_interpretation_uses_declared_type(value_type, setting, expected):
    result = evaluate_setting(
        "w",
        yaml.safe_load(setting),
        t=0.5,
        validate=TypeAdapter(value_type).validate_python,
    )
    assert result == expected
    assert type(result) is type(expected)


def test_gradient_constraints_validate_endpoints():
    adapter = TypeAdapter(Annotated[float, Field(ge=0, le=1)])
    with pytest.raises(ValidationError):
        evaluate_setting("w", [-1.0, 2.0], t=0.5, validate=adapter.validate_python)


def test_single_conditional_parameter_and_filter_fallback():
    config = MergeConfiguration.model_validate(
        {
            "merge_method": "linear",
            "models": [{"model": "a"}],
            "parameters": {"weight": {"filter": "embed", "value": ["1e-5", "1e-3"]}},
        }
    )
    parameter = merge_methods.get("linear").spec.input_parameters[0]
    reader = ConfigReader(config=config, t=0.5)
    assert resolve_parameter(
        parameter, reader.parameter_sources(), tensor_name="embed.weight", t=reader.t
    ) == pytest.approx(0.000505)
    assert (
        resolve_parameter(
            replace(parameter, default=1.0),
            reader.parameter_sources(),
            tensor_name="other",
            t=reader.t,
        )
        == 1.0
    )


def test_raw_config_uses_declared_types_for_shared_and_input_parameters():
    config = RawPyTorchMergeConfig.model_validate(
        yaml.safe_load(
            """
merge_method: linear
models:
  - model: a
    parameters:
      weight: [1e-5, 1e-3]
  - model: b
parameters:
  weight: [1e-4, 1e-2]
  normalize: [0, 1]
"""
        )
    )
    shared, per_input = construct_param_dicts(config, merge_methods.get("linear"), "w")
    assert shared["normalize"] is False
    assert sorted(p["weight"] for p in per_input.values()) == [1e-5, 1e-4]

    def kernel(group: TensorGroup, mode: Shared[str]) -> torch.Tensor:
        return group.entries[0].tensor

    method = merge_method(kernel, name="string_option")
    config = RawPyTorchMergeConfig(
        merge_method="string_option",
        models=[{"model": "a"}],
        parameters={"mode": ["1e-5", "1e-3"]},
    )
    shared, _ = construct_param_dicts(config, method, "w")
    assert shared["mode"] == "1e-5"


@pytest.mark.parametrize("method_name", ["linear", "ties", "nuslerp"])
@pytest.mark.parametrize("explicit_base", [False, True])
def test_raw_and_yaml_parameter_resolution_agree(method_name, explicit_base):
    raw = RawPyTorchMergeConfig.model_validate(
        {
            "merge_method": method_name,
            "base_model": "base",
            "models": [
                {
                    "model": "a",
                    "parameters": {"weight": {"filter": "other", "value": 99}},
                },
                {"model": "b", "parameters": {"weight": 0.0}},
            ]
            + ([{"model": "base"}] if explicit_base else []),
            "parameters": {"weight": ["1e-5", "1e-3"], "normalize": False},
        }
    )
    # A NON_BASE parameter must not even validate the base's configured value.
    if explicit_base and method_name != "linear":
        raw.models[-1] = raw.models[-1].model_copy(
            update={"parameters": {"weight": "not-a-number"}}
        )
    sources = [{**model.model_dump(), "layer_range": [0, 3]} for model in raw.models]
    if not explicit_base:
        sources.append({"model": "base", "layer_range": [0, 3]})
    config = MergeConfiguration.model_validate(
        {
            "merge_method": method_name,
            "base_model": "base",
            "slices": [{"sources": sources}],
            "parameters": raw.parameters,
        }
    )
    reader = ConfigReader(config=config, slice_out=config.slices[0], t=0)
    method = merge_methods.get(method_name)
    expected = resolve_parameters(
        method.spec,
        tensor_name="w",
        inputs={source.model: "w" for source in config.slices[0].sources},
        sources=reader.parameter_sources,
        base_model=reader.base_model,
        t=reader.t,
    )
    actual = construct_param_dicts(raw, method, "w")
    assert actual == expected
    shared, per_input = actual
    base_values = per_input[config.base_model]
    assert base_values == ({"weight": 1e-5} if method_name == "linear" else {})
    assert per_input[config.slices[0].sources[0].model]["weight"] == 1e-5
    assert per_input[config.slices[0].sources[1].model]["weight"] == 0.0
    if method_name in ("linear", "ties"):
        assert shared["normalize"] is False
    if method_name == "ties":
        assert per_input[config.slices[0].sources[0].model]["density"] == 1.0


@pytest.mark.parametrize("first_matching_scope", range(5))
def test_precedence_and_distinct_input_output_tensor_names(first_matching_scope):
    def kernel(
        group: TensorGroup, weight: PerInput[float] = 7, gain: Shared[float] = 8
    ) -> torch.Tensor:
        return group.entries[0].tensor

    method = merge_method(kernel, name="precedence")
    source = {"model": "a", "layer_range": [0, 3]}
    output_slice = {"sources": [source]}
    module = {"slices": [output_slice]}
    raw = {"merge_method": "precedence", "modules": {"model": module}}
    for index, scope in enumerate([source, output_slice, module, raw]):
        scope["parameters"] = {
            "weight": {
                "filter": "input" if index >= first_matching_scope else "absent",
                "value": [index + 1, index + 3],
            },
            "gain": {"filter": "output", "value": 99 - index},
        }
    config = MergeConfiguration.model_validate(raw)
    module = config.modules["model"]
    output_slice = module.slices[0]
    reader = ConfigReader(config=config, module=module, slice_out=output_slice, t=0.5)
    model = output_slice.sources[0].model
    shared, per_input = resolve_parameters(
        method.spec,
        tensor_name="output.weight",
        inputs={model: "input.weight"},
        sources=reader.parameter_sources,
        t=reader.t,
    )
    assert shared == {"gain": 98.0}  # Input-level settings do not affect shared values.
    assert per_input[model]["weight"] == (
        first_matching_scope + 2 if first_matching_scope < 4 else 7.0
    )
    assert type(per_input[model]["weight"]) is float  # Defaults are validated too.


def test_missing_required_parameters_have_model_and_tensor_context():
    config = RawPyTorchMergeConfig(
        merge_method="linear",
        models=[{"model": "a", "parameters": {"weight": 1}}, {"model": "b"}],
    )
    with pytest.raises(RuntimeError, match="Missing required parameter weight for b.w"):
        construct_param_dicts(config, merge_methods.get("linear"), "w")


def test_invalid_defaults_fail_during_parameter_resolution():
    def kernel(group: TensorGroup, count: Shared[int] = 1.5) -> torch.Tensor:
        return group.entries[0].tensor

    method = merge_method(kernel, name="bad_default")
    config = RawPyTorchMergeConfig(merge_method="bad_default", models=[{"model": "a"}])
    with pytest.raises(ValidationError):
        construct_param_dicts(config, method, "w")
