# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import mergekit.merge_methods.multislerp
import mergekit.merge_methods.nearswap
import mergekit.merge_methods.sce
from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.generalized_task_arithmetic import (
    GeneralizedTaskArithmeticMerge,
)
from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS


def get(method: str) -> MergeMethod:
    if method in REGISTERED_MERGE_METHODS:
        return REGISTERED_MERGE_METHODS[method]
    raise RuntimeError(f"Unimplemented merge method {method}")


__all__ = [
    "MergeMethod",
    "get",
    "GeneralizedTaskArithmeticMerge",
    "REGISTERED_MERGE_METHODS",
]
