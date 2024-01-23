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

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, model_validator
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
                    or (tensor_name and cond.filter in tensor_name)
                ):
                    res = evaluate_setting(tensor_name, cond.value, t)
                    return res
    else:
        raise RuntimeError(f"Unexpected setting value: {setting}")
    return None


class InputSliceDefinition(BaseModel):
    model: ModelReference
    layer_range: Tuple[int, int]
    parameters: Optional[Dict[str, ParameterSetting]] = None


class InputModelDefinition(BaseModel):
    model: ModelReference
    parameters: Optional[Dict[str, ParameterSetting]] = None


class OutputSliceDefinition(BaseModel):
    sources: List[InputSliceDefinition]
    base_model: Optional[ModelReference] = None
    residual_weight: Optional[float] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None


class MergeConfiguration(BaseModel):
    merge_method: str
    slices: Optional[List[OutputSliceDefinition]] = None
    models: Optional[List[InputModelDefinition]] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None
    base_model: Optional[ModelReference] = None
    dtype: Optional[str] = None
    tokenizer_source: Optional[str] = None

    def referenced_models(self) -> List[ModelReference]:
        models = set()
        if self.base_model:
            models.add(self.base_model)
        if self.models:
            for model_in in self.models:
                models.add(model_in.model)
        if self.slices:
            for s in self.slices:
                for src in s.sources:
                    models.add(src.model)
        return list(models)

    @model_validator(mode="after")
    def validate_inputs(self):
        if ((not self.slices) and (not self.models)) or (self.slices and self.models):
            raise RuntimeError("Must specify either output slices or models to merge")
        return self

    def to_yaml(self) -> str:
        return yaml.dump(
            self.model_dump(exclude_defaults=True, mode="json"),
            Dumper=ConfigYamlDumper,
        ).rstrip()


class ConfigReader(BaseModel):
    config: MergeConfiguration
    t: float
    tensor_name: Optional[str] = None
    slice_out: Optional[OutputSliceDefinition] = None

    @property
    def base_model(self) -> Optional[ModelReference]:
        if self.slice_out and self.slice_out.base_model:
            res = self.slice_out.base_model
        else:
            res = self.config.base_model

        return res

    def for_out_slice(self, slice: OutputSliceDefinition) -> "ConfigReader":
        return ConfigReader(
            config=self.config,
            t=self.t,
            tensor_name=self.tensor_name,
            slice_out=slice,
        )

    def for_tensor(self, tensor_name: str) -> "ConfigReader":
        return ConfigReader(
            config=self.config,
            t=self.t,
            tensor_name=tensor_name,
            slice_out=self.slice_out,
        )

    def with_t(self, t: float) -> "ConfigReader":
        return ConfigReader(
            config=self.config,
            t=t,
            tensor_name=self.tensor_name,
            slice_out=self.slice_out,
        )

    def parameter(
        self,
        name: str,
        model: Optional[ModelReference] = None,
        default: Any = None,
        required: bool = False,
    ) -> Any:
        if self.slice_out:
            if model:
                for s in self.slice_out.sources:
                    if s.model == model and s.parameters and name in s.parameters:
                        value = evaluate_setting(
                            self.tensor_name, s.parameters[name], self.t
                        )
                        if value is not None:
                            return value

            if self.slice_out.parameters and name in self.slice_out.parameters:
                value = evaluate_setting(
                    self.tensor_name, self.slice_out.parameters[name], self.t
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
            path_paths = [str(s) for s in [model, self.tensor_name] if s]
            p = ".".join(path_paths)
            suffix = f" for {p}" if p else ""
            raise RuntimeError(f"Missing required parameter {name}{suffix}")
        return default


class ConfigYamlDumper(yaml.Dumper):
    """Custom YAML dumper to format lists of numbers in flow style."""

    def represent_list(self, data: Iterable[Any]) -> yaml.SequenceNode:
        flow_style = all(isinstance(e, (int, float)) for e in data)
        return self.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=flow_style
        )


ConfigYamlDumper.add_representer(list, ConfigYamlDumper.represent_list)
