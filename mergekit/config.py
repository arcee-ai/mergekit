# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, model_validator
from typing_extensions import Literal, TypeAlias

from mergekit.common import ModelReference
from mergekit.tokenizer.config import TokenizerConfig

ScalarOrGradient: TypeAlias = Union[float, List[float]]


class ConditionalParameter(BaseModel):
    value: ScalarOrGradient
    filter: Optional[str] = None


ParameterSetting: TypeAlias = Union[
    ConditionalParameter, List[ConditionalParameter], ScalarOrGradient
]


def evaluate_setting(
    tensor_name: str, setting: ParameterSetting, t: float = 0
) -> Optional[float]:
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


class OutputModuleDefinition(BaseModel):
    slices: Optional[List[OutputSliceDefinition]] = None
    models: Optional[List[InputModelDefinition]] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None

    @model_validator(mode="after")
    def validate_inputs(self):
        if ((not self.slices) and (not self.models)) or (self.slices and self.models):
            raise RuntimeError("Must specify either output slices or models to merge")
        return self


class MergeConfiguration(BaseModel):
    modules: Optional[Dict[str, OutputModuleDefinition]] = None
    slices: Optional[List[OutputSliceDefinition]] = None
    models: Optional[List[InputModelDefinition]] = None

    merge_method: str
    base_model: Optional[ModelReference] = None
    dtype: Optional[str] = None
    tokenizer_source: Union[Literal["union"], Literal["base"], ModelReference, None] = (
        None
    )
    tokenizer: Optional[TokenizerConfig] = None
    chat_template: Optional[str] = None
    out_dtype: Optional[str] = None
    parameters: Optional[Dict[str, ParameterSetting]] = None

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
        if self.modules:
            for m in self.modules.values():
                if m.models:
                    for model_in in m.models:
                        models.add(model_in.model)
                if m.slices:
                    for s in m.slices:
                        for src in s.sources:
                            models.add(src.model)
        return list(models)

    @model_validator(mode="after")
    def validate_inputs(self):
        set_ct = 0
        if self.modules:
            set_ct += 1
        if self.slices:
            set_ct += 1
        if self.models:
            set_ct += 1

        if set_ct != 1:
            raise RuntimeError(
                "Exactly one of 'models', 'slices', or 'modules' must be present"
            )
        return self

    @model_validator(mode="after")
    def validate_tokenizer(self):
        if self.tokenizer_source and self.tokenizer:
            raise RuntimeError("Cannot specify both tokenizer_source and tokenizer")
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
    module: Optional[OutputModuleDefinition] = None

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
            module=self.module,
        )

    def for_tensor(self, tensor_name: str) -> "ConfigReader":
        return ConfigReader(
            config=self.config,
            t=self.t,
            tensor_name=tensor_name,
            slice_out=self.slice_out,
            module=self.module,
        )

    def with_t(self, t: float) -> "ConfigReader":
        return ConfigReader(
            config=self.config,
            t=t,
            tensor_name=self.tensor_name,
            slice_out=self.slice_out,
            module=self.module,
        )

    def for_module(self, module: OutputModuleDefinition) -> "ConfigReader":
        return ConfigReader(
            config=self.config,
            t=self.t,
            tensor_name=self.tensor_name,
            slice_out=self.slice_out,
            module=module,
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

        if self.module and self.module.parameters and name in self.module.parameters:
            value = evaluate_setting(
                self.tensor_name,
                self.module.parameters[name],
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
