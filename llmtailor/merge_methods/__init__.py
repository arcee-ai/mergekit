# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import llmtailor.merge_methods.multislerp
import llmtailor.merge_methods.nearswap
import llmtailor.merge_methods.sce
from llmtailor.merge_methods.base import MergeMethod
from llmtailor.merge_methods.generalized_task_arithmetic import (
    GeneralizedTaskArithmeticMerge,
)
from llmtailor.merge_methods.registry import REGISTERED_MERGE_METHODS


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
