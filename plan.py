from typing import Dict, List, Optional, Tuple

import merge_methods
from common import ModelArchitectureInfo, ModelReference
from config import (
    ConfigReader,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from graph import Operation, TensorReference
from merge_methods import MergeMethod


def plan(
    config: MergeConfiguration, arch_info: ModelArchitectureInfo
) -> Tuple[List[TensorReference], Dict[TensorReference, Operation]]:
    layer_idx = 0

    targets = []
    rules = {}

    method = merge_methods.get(config.merge_method)

    for weight_name in arch_info.pre_weights:
        tr, op = make_operation(config, weight_name, config.slices[0].sources, t=0)
        targets.append(tr)
        rules[tr] = op

    for section in config.slices:
        (new_targets, new_rules, new_layers) = plan_slice(
            config, section, arch_info, layer_idx, method
        )

        targets.extend(new_targets)
        rules.update(new_rules)
        layer_idx += new_layers

    for weight_name in arch_info.post_weights:
        tr, op = make_operation(config, weight_name, config.slices[-1].sources, t=1)
        targets.append(tr)
        rules[tr] = op

    return (targets, rules)


def make_operation(
    config: MergeConfiguration,
    name_out: str,
    tensor_sources: List[InputSliceDefinition],
    t: float,
    names_in: Optional[List[str]] = None,
    sdef: Optional[OutputSliceDefinition] = None,
    extra_dependencies: Optional[List[TensorReference]] = None,
):
    if names_in is None:
        names_in = [name_out] * len(tensor_sources)

    input_tensors = []
    kwargs = {
        "config": ConfigReader(
            config=config,
            tensor_name=name_out,
            t=t,
            slice_out=sdef,
            slices_in=tensor_sources,
        ),
        "parameter_name": name_out,
    }

    for i, s in enumerate(tensor_sources):
        input_tensors.append(
            TensorReference(model=ModelReference.parse(s.model), key=names_in[i])
        )

    if extra_dependencies:
        input_tensors.extend(extra_dependencies)

    tr = TensorReference(model=None, key=name_out)
    op = Operation(function="merge", inputs=input_tensors, kwargs=kwargs)
    return tr, op


def plan_slice(
    config: MergeConfiguration,
    definition: OutputSliceDefinition,
    arch_info: ModelArchitectureInfo,
    layer_base: int,
    method: MergeMethod,
) -> Tuple[List[TensorReference], Dict[TensorReference, Operation], int]:
    slice_indices = get_slice_indices(definition)

    num_layers = len(slice_indices[0])

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
    method: MergeMethod,
    rules: Dict[TensorReference, Operation],
    targets: List[TensorReference],
    idx: int,
    t: float,
):
    extra_dependencies = list(method.general_dependencies())
    for si, s in enumerate(definition.sources):
        source_layer_idx = slice_indices[si][idx]
        source_model = ModelReference.parse(s.model)
        extra_dependencies.extend(
            method.input_layer_dependencies(source_model, source_layer_idx)
        )

    for suffix in arch_info.layer_weights:
        name_out = (
            arch_info.layer_prefix_format.format(idx=layer_base + idx) + "." + suffix
        )
        names_in = [
            arch_info.layer_prefix_format.format(idx=slice_indices[si][idx])
            + "."
            + suffix
            for (si, _) in enumerate(definition.sources)
        ]

        tr, op = make_operation(
            config,
            name_out,
            definition.sources,
            t,
            names_in=names_in,
            sdef=definition,
            extra_dependencies=extra_dependencies,
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
