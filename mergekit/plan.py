# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

import mergekit.merge_methods as merge_methods
from mergekit.architecture import ArchitectureInfo
from mergekit.common import ModelReference
from mergekit.config import (
    ConfigReader,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from mergekit.graph import Operation, TensorReference
from mergekit.merge_methods import MergeMethod


def plan(
    merge_config: MergeConfiguration,
    arch_info: ArchitectureInfo,
    embed_permutations: Optional[Dict[ModelReference, torch.Tensor]] = None,
    trust_remote_code: bool = False,
) -> Tuple[List[TensorReference], Dict[TensorReference, Operation]]:
    layer_idx = 0

    targets = []
    rules = {}

    method = merge_methods.get(merge_config.merge_method)
    base_model = (
        ModelReference.parse(merge_config.base_model)
        if merge_config.base_model
        else None
    )

    # if models to merge are specified instead of output slices, compute them
    if merge_config.models:
        if merge_config.slices:
            raise RuntimeError("Must specify either models to merge or output slices")

        slices_in = []
        base_included = False

        for model_in in merge_config.models:
            mref = ModelReference.parse(model_in.model)

            if base_model and mref == base_model:
                base_included = True

            model_cfg = mref.config(trust_remote_code=trust_remote_code)
            num_layers = arch_info.num_layers(model_cfg)
            slices_in.append(
                InputSliceDefinition(
                    layer_range=[0, num_layers],
                    model=model_in.model,
                    parameters=model_in.parameters,
                )
            )

        if base_model and not base_included:
            logging.info("Base model specified but not in input models - adding")
            base_cfg = base_model.config(trust_remote_code=trust_remote_code)
            num_layers = arch_info.num_layers(base_cfg)
            slices_in.append(
                InputSliceDefinition(
                    layer_range=[0, num_layers],
                    model=str(base_model),
                )
            )

        merge_config.slices = [OutputSliceDefinition(sources=slices_in)]
        merge_config.models = None

    for weight_name in arch_info.pre_weights():
        is_embed = weight_name in arch_info.embed_weights()

        tr, op = make_operation(
            merge_config,
            weight_name,
            merge_config.slices[0].sources,
            t=0,
            extra_kwargs={"embed_permutations": embed_permutations},
            function="merge_embed" if (is_embed and embed_permutations) else "merge",
        )
        targets.append(tr)
        rules[tr] = op

    for section in merge_config.slices:
        (new_targets, new_rules, new_layers) = plan_slice(
            config=merge_config,
            definition=section,
            arch_info=arch_info,
            layer_base=layer_idx,
            method=method,
        )

        targets.extend(new_targets)
        rules.update(new_rules)
        layer_idx += new_layers

    for weight_name in arch_info.post_weights():
        is_embed = weight_name in arch_info.embed_weights()
        tr, op = make_operation(
            merge_config,
            weight_name,
            merge_config.slices[-1].sources,
            t=1,
            extra_kwargs={"embed_permutations": embed_permutations},
            function="merge_embed" if (is_embed and embed_permutations) else "merge",
        )
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
    extra_kwargs: Optional[Dict[str, Any]] = None,
    function: str = "merge",
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
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    for i, s in enumerate(tensor_sources):
        input_tensors.append(
            TensorReference(model=ModelReference.parse(s.model), key=names_in[i])
        )

    if extra_dependencies:
        input_tensors.extend(extra_dependencies)

    tr = TensorReference(model=None, key=name_out)
    op = Operation(function=function, inputs=input_tensors, kwargs=kwargs)
    return tr, op


def plan_slice(
    config: MergeConfiguration,
    definition: OutputSliceDefinition,
    arch_info: ArchitectureInfo,
    layer_base: int,
    method: MergeMethod,
) -> Tuple[List[TensorReference], Dict[TensorReference, Operation], int]:
    slice_indices = get_slice_indices(definition)

    num_layers = len(slice_indices[0])

    rules = {}
    targets = []
    for idx in range(num_layers):
        if num_layers > 1:
            t = idx / (num_layers - 1)
        else:
            t = 1

        plan_layer(
            config=config,
            definition=definition,
            arch_info=arch_info,
            layer_base=layer_base,
            slice_indices=slice_indices,
            method=method,
            rules=rules,
            targets=targets,
            idx=idx,
            t=t,
        )

    return targets, rules, num_layers


def plan_layer(
    config: MergeConfiguration,
    definition: OutputSliceDefinition,
    arch_info: ArchitectureInfo,
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

    for name_format in arch_info.layer_weight_formats():
        name_out = name_format.format(idx=layer_base + idx)
        names_in = [
            name_format.format(idx=slice_indices[si][idx])
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
