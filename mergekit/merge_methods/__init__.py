# Copyright (C) 2025 Arcee AI
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

import mergekit.merge_methods.multislerp
from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.generalized_task_arithmetic import (
    GeneralizedTaskArithmeticMerge,
)
from mergekit.merge_methods.linear import LinearMerge
from mergekit.merge_methods.passthrough import PassthroughMerge
from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS
from mergekit.merge_methods.slerp import SlerpMerge


def get(method: str) -> MergeMethod:
    if method in REGISTERED_MERGE_METHODS:
        return REGISTERED_MERGE_METHODS[method]
    raise RuntimeError(f"Unimplemented merge method {method}")


__all__ = [
    "MergeMethod",
    "get",
    "LinearMerge",
    "SCEMerge",
    "SlerpMerge",
    "PassthroughMerge",
    "GeneralizedTaskArithmeticMerge",
    "REGISTERED_MERGE_METHODS",
]
