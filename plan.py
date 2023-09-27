from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from pydantic import BaseModel
from typing_extensions import TypeAlias

from common import ModelArchitectureInfo, ModelReference
from graph import Operation, OperationProtocol, ProceduralRule, TensorReference


class MergeMethodBase(ABC):
    @abstractmethod
    def function_name(self) -> str:
        ...

    @abstractmethod
    def parameters(self) -> List[str]:
        """List of scalar parameters"""
        ...

    @abstractmethod
    def model_parameters(self) -> List[str]:
        """List of per-model scalar parameters"""
        ...

    @abstractmethod
    def operations(self) -> Dict[str, OperationProtocol]:
        """Any operations necessary for rules"""
        ...

    @abstractmethod
    def rules(self, input_models: List[ModelReference]) -> List[ProceduralRule]:
        """Any rules necessary to produce dependency tensors"""
        ...


class LinearMerge(MergeMethodBase):
    def function_name(self) -> str:
        return "merge_linear"

    def parameters(self) -> List[str]:
        return []

    def model_parameters(self) -> List[str]:
        return ["weight"]

    def operations(self) -> Dict[str, OperationProtocol]:
        return {"merge_linear": LinearMerge.merge_linear}

    def rules(self, input_models: List[ModelReference]) -> List[ProceduralRule]:
        return []

    @staticmethod
    def merge_linear(
        _input_tensors: Dict[ModelReference, torch.Tensor],
        weight: Dict[ModelReference, float],
    ):
        print(weight)


def get_mm(name: str) -> MergeMethodBase:
    return LinearMerge()


ScalarOrGradient: TypeAlias = Union[float, List[float]]


class ConditionalParameter(BaseModel):
    value: ScalarOrGradient
    pattern: Optional[str] = None


ParameterSetting: TypeAlias = Union[
    ConditionalParameter, List[ConditionalParameter], ScalarOrGradient
]


def evaluate_setting(
    tensor_name: str, setting: ParameterSetting, t: float = 0
) -> float:
    if isinstance(setting, float):
        return setting
    elif isinstance(setting, list):
        if all(isinstance(e, float) for e in setting):
            scaled = t * (len(setting) - 1)
            i0 = int(scaled)
            i1 = min(len(setting - 1), i0 + 1)
            frac = scaled - i0

            return (1 - frac) * setting[i0] + frac * setting[i1]
        else:
            for cond in setting:
                if (
                    (cond.pattern is None)
                    or (cond.pattern == "*")
                    or cond.pattern in tensor_name
                ):
                    return evaluate_setting(tensor_name, cond.value, t)
    return None


class InputSliceDefinition(BaseModel):
    model: str
    layer_range: Tuple[int, int]
    parameters: Optional[Dict[str, ParameterSetting]] = None


class OutputSliceDefinition(BaseModel):
    sources: List[InputSliceDefinition]
    merge_method: Optional[str] = None
    base_model: Optional[str] = None
    residual_weight: Optional[float] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None


class MergeConfiguration(BaseModel):
    slices: List[OutputSliceDefinition]
    merge_method: str
    model_parameters: Dict[str, Dict[str, ParameterSetting]] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None
    base_model: Optional[str] = None


def resolve_model_parameter(
    model: str,
    tensor_name: str,
    parameter: str,
    t: float,
    slice_in: InputSliceDefinition,
    config: MergeConfiguration,
) -> Optional[float]:
    if (
        slice_in
        and slice_in.parameters
        and (model in slice_in.parameters)
        and (parameter in slice_in.parameters[model])
    ):
        return evaluate_setting(tensor_name, slice_in.parameters[model][parameter], t)
    elif config.model_parameters and model in config.model_parameters:
        return evaluate_setting(tensor_name, config.model_parameters[parameter], t)
    return None


def resolve_parameter(
    tensor_name: str,
    parameter: str,
    t: float,
    slice_out: Optional[OutputSliceDefinition],
    config: MergeConfiguration,
) -> Optional[float]:
    if slice_out and slice_out.parameters and (parameter in slice_out.parameters):
        return evaluate_setting(tensor_name, slice_out.parameters[parameter], t)
    elif config.parameters and (parameter in config.parameters):
        return evaluate_setting(tensor_name, config.parameters[parameter], t)


def plan(config: MergeConfiguration, arch_info: ModelArchitectureInfo):
    layer_idx = 0

    targets = []
    rules = {}

    method = get_mm(config.merge_method)

    for weight_name in arch_info.pre_weights:
        tr, op = make_operation(
            config, weight_name, config.slices[0].sources, t=0, method=method
        )
        targets.append(tr)
        rules[tr] = op

    for section in config.slices:
        (new_targets, new_rules, new_layers) = plan_slice(
            config, section, arch_info, layer_idx
        )

        targets.extend(new_targets)
        rules.update(new_rules)
        layer_idx += new_layers

    for weight_name in arch_info.post_weights:
        tr, op = make_operation(
            config, weight_name, config.slices[-1].sources, t=1, method=method
        )
        targets.append(tr)
        rules[tr] = op

    return (targets, rules)


def make_operation(
    config: MergeConfiguration,
    name_out: str,
    tensor_sources: List[InputSliceDefinition],
    t: float,
    method: MergeMethodBase,
    names_in: Optional[List[str]] = None,
    sdef: Optional[OutputSliceDefinition] = None,
):
    if names_in is None:
        names_in = [name_out] * len(tensor_sources)

    input_tensors = []
    params = {}

    for param in method.parameters():
        params[param] = resolve_parameter(name_out, param, t, sdef, config)
    for param in method.model_parameters():
        params[param] = {}

    for i, s in enumerate(tensor_sources):
        input_tensors.append(
            TensorReference(model=ModelReference.parse(s.model), key=names_in[i])
        )

        for param in method.model_parameters():
            params[param][s.model] = resolve_model_parameter(
                str(s.model), names_in[i], param, t, sdef, config
            )

    tr = TensorReference(model=None, key=name_out)
    op = Operation(function=method.function_name(), inputs=input_tensors, kwargs=params)
    return tr, op


def plan_slice(
    config: MergeConfiguration,
    definition: OutputSliceDefinition,
    arch_info: ModelArchitectureInfo,
    layer_base: int,
) -> Tuple[List[TensorReference], Dict[TensorReference, Operation], int]:
    slice_indices = get_slice_indices(definition)

    num_layers = len(slice_indices[0])

    method_name = (
        definition.merge_method if definition.merge_method else config.merge_method
    )
    method = get_mm(method_name)

    rules = {}
    targets = []
    for idx in range(num_layers):
        t = idx / (num_layers - 1)
        plan_layer(
            config,
            definition,
            arch_info,
            layer_base,
            slice_indices,
            method,
            rules,
            targets,
            idx,
            t,
        )

    return targets, rules, num_layers


def plan_layer(
    config: MergeConfiguration,
    definition: OutputSliceDefinition,
    arch_info: ModelArchitectureInfo,
    layer_base: int,
    slice_indices: List[List[int]],
    method: MergeMethodBase,
    rules: Dict[TensorReference, Operation],
    targets: List[TensorReference],
    idx: int,
    t: float,
):
    for suffix in arch_info.layer_weights:
        name_out = (
            arch_info.layer_prefix_format.format(idx=layer_base + idx) + "." + suffix
        )
        names_in = [
            arch_info.layer_prefix_format.format(idx=slice_indices[si][idx])
            + "."
            + suffix
            for (si, s) in enumerate(definition.sources)
        ]

        tr, op = make_operation(
            config,
            name_out,
            definition.sources,
            t,
            method,
            names_in=names_in,
            sdef=definition,
        )
        rules[tr] = op
        targets.append(tr)


def get_slice_indices(definition: OutputSliceDefinition):
    slice_indices = []
    for s in definition.sources:
        indices = list(range(s.layer_range[0], s.layer_range[1]))
        if slice_indices and len(indices) != len(slice_indices[-1]):
            raise RuntimeError(
                "All inputs to a slice must contain the same number of layers"
            )
        slice_indices.append(indices)
    return slice_indices
