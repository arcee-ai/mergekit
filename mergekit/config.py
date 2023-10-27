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

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel
from typing_extensions import TypeAlias

from mergekit.common import ModelReference

ScalarOrGradient: TypeAlias = Union[float, List[float]]


class ConditionalParameter(BaseModel):
    value: ScalarOrGradient
    filter: Optional[str] = None


ParameterSetting: TypeAlias = Union[
    ConditionalParameter, List[ConditionalParameter], ScalarOrGradient
]


def evaluate_setting(
    tensor_name: str, setting: ParameterSetting, t: float = 0
) -> float:
    if isinstance(setting, (float, int, bool, str)):
        return setting
    elif isinstance(setting, list):
        if all(isinstance(e, (int, float)) for e in setting):
            scaled = t * (len(setting) - 1)
            i0 = int(scaled)
            i1 = min(len(setting) - 1, i0 + 1)
            frac = scaled - i0

            return (1 - frac) * setting[i0] + frac * setting[i1]
        elif all(isinstance(e, (float, int, bool, str)) for e in setting):
            return setting[int(t * (len(setting) - 1))]
        else:
            for cond in setting:
                if (
                    (cond.filter is None)
                    or (cond.filter == "*")
                    or cond.filter in tensor_name
                ):
                    res = evaluate_setting(tensor_name, cond.value, t)
                    return res
    else:
        raise RuntimeError(f"Unexpected setting value: {setting}")
    return None


class InputSliceDefinition(BaseModel):
    model: str
    layer_range: Tuple[int, int]
    parameters: Optional[Dict[str, ParameterSetting]] = None


class InputModelDefinition(BaseModel):
    model: str
    parameters: Optional[Dict[str, ParameterSetting]] = None


class OutputSliceDefinition(BaseModel):
    sources: List[InputSliceDefinition]
    base_model: Optional[str] = None
    residual_weight: Optional[float] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None


class MergeConfiguration(BaseModel):
    merge_method: str
    slices: Optional[List[OutputSliceDefinition]] = None
    models: Optional[List[InputModelDefinition]] = None
    input_model_parameters: Dict[str, Dict[str, ParameterSetting]] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None
    base_model: Optional[str] = None
    dtype: Optional[str] = None

    def referenced_models(self) -> List[ModelReference]:
        models = set()
        if self.base_model:
            models.add(ModelReference.parse(self.base_model))
        if self.input_model_parameters:
            for key in self.input_model_parameters:
                models.add(ModelReference.parse(key))
        if self.models:
            for model_in in self.models:
                models.add(ModelReference.parse(model_in.model))
        if self.slices:
            for s in self.slices:
                for src in s.sources:
                    models.add(ModelReference.parse(src.model))
        return list(models)

    def validate(self):
        if ((not self.slices) and (not self.models)) or (self.slices and self.models):
            raise RuntimeError("Must specify either output slices or models to merge")


class ConfigReader(BaseModel):
    config: MergeConfiguration
    tensor_name: str
    t: float
    slice_out: Optional[OutputSliceDefinition]
    slices_in: Optional[List[InputSliceDefinition]]

    @property
    def base_model(self) -> Optional[ModelReference]:
        if self.slice_out and self.slice_out.base_model:
            res = self.slice_out.base_model
        else:
            res = self.config.base_model

        if res:
            return ModelReference.parse(res)
        return None

    def parameter(
        self,
        name: str,
        model: Optional[ModelReference] = None,
        default: Any = None,
        required: bool = False,
    ) -> Any:
        if model and self.slices_in:
            for s in self.slices_in:
                if s.model == str(model) and s.parameters and name in s.parameters:
                    value = evaluate_setting(
                        self.tensor_name, s.parameters[name], self.t
                    )
                    if value is not None:
                        return value

        if self.slice_out:
            if self.slice_out.parameters and name in self.slice_out.parameters:
                value = evaluate_setting(
                    self.tensor_name, self.slice_out.parameters[name], self.t
                )
                if value is not None:
                    return value

        if (
            self.config.input_model_parameters
            and model
            and str(model) in self.config.input_model_parameters
        ):
            if name in self.config.input_model_parameters[self.slice_in.model]:
                value = evaluate_setting(
                    self.tensor_name,
                    self.config.input_model_parameters[str(model)][name],
                    self.t,
                )
                if value is not None:
                    return value

        if self.config.parameters and name in self.config.parameters:
            value = evaluate_setting(
                self.tensor_name,
                self.config.parameters[name],
                self.t,
            )
            if value is not None:
                return value

        if required:
            suffix = (
                f" for {str(model)}.{self.tensor_name}"
                if model
                else f" for {self.tensor_name}"
            )
            raise RuntimeError(f"Missing required parameter {name}{suffix}")
        return default
