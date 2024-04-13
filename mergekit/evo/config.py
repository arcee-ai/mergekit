# Copyright (C) 2024 Charles O. Goddard
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from typing import List, Optional

from pydantic import BaseModel, model_validator

from mergekit.evo.genome import ModelGenomeDefinition


class TaskConfiguration(BaseModel, frozen=True):
    name: str
    weight: float = 1.0
    metric: str = "acc,none"

    @model_validator(mode="before")
    def validate_string(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value


class EvolMergeConfiguration(BaseModel, frozen=True):
    genome: ModelGenomeDefinition
    tasks: List[TaskConfiguration]
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    shuffle: bool = False
    random_init: bool = False
