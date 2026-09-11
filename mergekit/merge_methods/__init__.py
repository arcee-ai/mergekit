# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from mergekit.merge_methods.api import merge_state_dicts
from mergekit.merge_methods.base import (
    BasePolicy,
    BatchedMergeMethod,
    BatchOptions,
    BatchParameter,
    GroupKernelAdapter,
    InputContract,
    InputParameterTarget,
    MergeBatch,
    MergedBatch,
    MergeMethod,
    MergeMethodSpec,
    Option,
    OptionalTensorPolicy,
    ParameterScope,
    PerGroupValues,
    PerInput,
    PerInputValues,
    PerNonBase,
    Shared,
    TensorBatch,
    TensorEntry,
    TensorGroup,
    TensorMetadata,
)
from mergekit.merge_methods.easy_define import (
    from_batch_kernel,
    from_group_kernel,
    group_merge_method,
    merge_method,
)
from mergekit.merge_methods.generalized_task_arithmetic import (
    GeneralizedTaskArithmeticMerge,
)
from mergekit.merge_methods.multislerp import multislerp as multislerp
from mergekit.merge_methods.nearswap import nearswap_merge as nearswap_merge
from mergekit.merge_methods.ram import ram_merge as ram_merge
from mergekit.merge_methods.ram import ramplus_tl_merge as ramplus_tl_merge
from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS
from mergekit.merge_methods.sce import sce_merge as sce_merge


def get(method: str) -> MergeMethod:
    if method in REGISTERED_MERGE_METHODS:
        return REGISTERED_MERGE_METHODS[method]
    raise RuntimeError(f"Unimplemented merge method {method}")


__all__ = [
    "BatchedMergeMethod",
    "BatchOptions",
    "BatchParameter",
    "TensorBatch",
    "GroupKernelAdapter",
    "Option",
    "OptionalTensorPolicy",
    "ParameterScope",
    "InputParameterTarget",
    "from_batch_kernel",
    "from_group_kernel",
    "group_merge_method",
    "merge_method",
    "MergeMethod",
    "MergeMethodSpec",
    "MergeBatch",
    "MergedBatch",
    "TensorGroup",
    "TensorEntry",
    "TensorMetadata",
    "merge_state_dicts",
    "InputContract",
    "BasePolicy",
    "Shared",
    "PerInput",
    "PerNonBase",
    "PerInputValues",
    "PerGroupValues",
    "multislerp",
    "nearswap_merge",
    "ram_merge",
    "ramplus_tl_merge",
    "get",
    "GeneralizedTaskArithmeticMerge",
    "REGISTERED_MERGE_METHODS",
    "sce_merge",
]
