# Copyright (C) 2023 Charles O. Goddard
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

from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.linear import LinearMerge
from mergekit.merge_methods.passthrough import PassthroughMerge
from mergekit.merge_methods.slerp import SlerpMerge
from mergekit.merge_methods.taskarithmetic import TaskArithmeticMerge
from mergekit.merge_methods.ties import TiesMerge


def get(method: str) -> MergeMethod:
    if method == "ties":
        return TiesMerge()
    elif method == "linear":
        return LinearMerge()
    elif method == "slerp":
        return SlerpMerge()
    elif method == "passthrough":
        return PassthroughMerge()
    elif method == "task_arithmetic":
        return TaskArithmeticMerge()
    raise RuntimeError(f"Unimplemented merge method {method}")


__all__ = [
    "MergeMethod",
    "get",
    "LinearMerge",
    "TiesMerge",
    "SlerpMerge",
    "PassthroughMerge",
    "TaskArithmeticMerge",
]
