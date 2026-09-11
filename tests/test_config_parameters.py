from typing import Annotated

import pytest
import torch
import yaml
from pydantic import Field, TypeAdapter, ValidationError

from mergekit import merge_methods
from mergekit.config import ConfigReader, MergeConfiguration, evaluate_setting
from mergekit.merge_methods import Shared, TensorGroup, from_group_kernel
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
    reader = ConfigReader(
        config=config, module=module, slice_out=output_slice, t=0.5, tensor_name="w"
    )
    parameter = merge_methods.get("linear").spec.input_parameters[0]
    assert reader.parameter(
        "weight", model=output_slice.sources[0].model, validate=parameter.validate
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
    reader = ConfigReader(config=config, t=0.5, tensor_name="embed.weight")
    assert reader.parameter("weight", validate=parameter.validate) == pytest.approx(
        0.000505
    )
    assert (
        reader.for_tensor("other").parameter(
            "weight", default=1.0, validate=parameter.validate
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

    method = from_group_kernel(kernel, name="string_option")
    config = RawPyTorchMergeConfig(
        merge_method="string_option",
        models=[{"model": "a"}],
        parameters={"mode": ["1e-5", "1e-3"]},
    )
    shared, _ = construct_param_dicts(config, method, "w")
    assert shared["mode"] == "1e-5"
