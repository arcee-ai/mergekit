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
import os
from functools import lru_cache
from typing import Dict, List, Optional

from mergekit import merge_methods
from mergekit.architecture import ModelArchitecture, WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.config import (
    ConfigReader,
    InputSliceDefinition,
    MergeConfiguration,
    OutputModuleDefinition,
    OutputSliceDefinition,
)
from mergekit.graph import Task
from mergekit.io.tasks import FinalizeModel, GatherTensors, SaveTensor, TensorWriterTask
from mergekit.merge_methods import MergeMethod
from mergekit.merge_methods.tokenizer_permute import TokenizerPermutationMerge
from mergekit.options import MergeOptions
from mergekit.tokenizer import BuildTokenizer


class MergePlanner:
    config: MergeConfiguration
    arch_info: ModelArchitecture
    options: MergeOptions
    out_path: str
    _writer_task: TensorWriterTask
    _tensor_save_tasks: Dict[TensorWriterTask, List[SaveTensor]]
    _method: MergeMethod
    _tasks: List[Task] = []
    _current_module_layers: int = 0
    _tokenizer_task: Optional[BuildTokenizer] = None

    def __init__(
        self,
        config: MergeConfiguration,
        arch_info: ModelArchitecture,
        out_path: str,
        options: MergeOptions,
    ):
        self.config = config
        self.arch_info = arch_info
        self.options = options
        self.out_path = out_path
        self._method = merge_methods.get(config.merge_method)
        self._tensor_save_tasks = {}

        if config.tokenizer_source:
            self._tokenizer_task = BuildTokenizer(
                base_model=config.base_model,
                referenced_models=tuple(config.referenced_models()),
                tokenizer_source=config.tokenizer_source,
                trust_remote_code=options.trust_remote_code,
            )

    def normalize_config(self):
        base_model = self.config.base_model

        # models -> modules.models
        if self.config.models:
            self.config.modules = {}
            for module_name in self.arch_info.modules:
                self.config.modules[module_name] = OutputModuleDefinition(
                    name=module_name, models=self.config.models
                )
            self.config.models = None

        # slices -> modules.slices
        if self.config.slices:
            if len(self.arch_info.modules) != 1:
                raise RuntimeError(
                    "Model has multiple modules, must use modules: syntax"
                )
            module_name = list(self.arch_info.modules.keys())[0]
            self.config.modules = {
                module_name: OutputModuleDefinition(slices=self.config.slices)
            }
            self.config.slices = None

        # modules.models -> modules.slices
        for module_name in self.config.modules:
            module_out = self.config.modules[module_name]
            num_layers_key = (
                self.arch_info.modules[module_name].config_prefix or ""
            ) + self.arch_info.modules[module_name].architecture.num_layers_config_key()

            if module_out.models:
                slices_in = []
                base_included = False

                for model_in in module_out.models:
                    if base_model and model_in.model == base_model:
                        base_included = True

                    model_cfg = model_in.model.config(
                        trust_remote_code=self.options.trust_remote_code
                    )
                    num_layers = int(getattr(model_cfg, num_layers_key))
                    slices_in.append(
                        InputSliceDefinition(
                            layer_range=[0, num_layers],
                            model=model_in.model,
                            parameters=model_in.parameters,
                        )
                    )

                if base_model and not base_included:
                    logging.info(
                        "Base model specified but not in input models - adding"
                    )
                    base_cfg = base_model.config(
                        trust_remote_code=self.options.trust_remote_code
                    )
                    num_layers = int(getattr(base_cfg, num_layers_key))
                    slices_in.append(
                        InputSliceDefinition(
                            layer_range=[0, num_layers],
                            model=base_model,
                        )
                    )

                module_out.slices = [OutputSliceDefinition(sources=slices_in)]
                module_out.models = None

    @lru_cache
    def _tensor_writer(self, subfolder: Optional[str] = None):
        path = self.out_path
        if subfolder:
            path = os.path.join(path, subfolder)
        return TensorWriterTask(
            out_path=path,
            max_shard_size=self.options.out_shard_size,
            safe_serialization=self.options.safe_serialization,
        )

    def plan_tensor(
        self,
        name: str,
        names_in: List[str],
        models: List[ModelReference],
        cfg_reader: ConfigReader,
        tensor_writer: TensorWriterTask,
        is_embed: bool = False,
    ):
        tensor_merge_method = self._method
        if self._tokenizer_task and is_embed:
            tensor_merge_method = TokenizerPermutationMerge(
                tokenizer_task=self._tokenizer_task
            )

        cfg_g = cfg_reader.for_tensor(name)
        global_params = {}
        for p in tensor_merge_method.parameters():
            global_params[p.name] = cfg_g.parameter(
                p.name, model=None, required=p.required, default=p.default_value
            )

        tensor_params = {}
        for model, name_in in zip(models, names_in):
            is_base = model == cfg_reader.config.base_model
            tensor_params[model] = {}
            cfg_m = cfg_reader.for_tensor(name_in)
            for p in tensor_merge_method.tensor_parameters():
                tensor_params[model][p.name] = cfg_m.parameter(
                    p.name,
                    model=model,
                    required=p.required and not is_base,
                    default=p.default_value,
                )

        gather_tensors = GatherTensors(
            tensor_names=ImmutableMap(data=dict(zip(models, names_in))),
            dtype=self.config.dtype,
        )

        tensor_task = tensor_merge_method.make_task(
            output_tensor_name=name,
            tensors=gather_tensors,
            parameters=ImmutableMap(data=global_params),
            tensor_parameters=ImmutableMap(
                data={
                    key: ImmutableMap(data=tensor_params[key]) for key in tensor_params
                }
            ),
            base_model=self.config.base_model,
        )
        save_task = SaveTensor(
            tensor_name=name,
            tensor_task=tensor_task,
            writer_task=tensor_writer,
            clone=self.options.clone_tensors,
        )
        if tensor_writer not in self._tensor_save_tasks:
            self._tensor_save_tasks[tensor_writer] = []
        self._tensor_save_tasks[tensor_writer].append(save_task)
        self._tasks.append(save_task)

    def plan_layer(
        self,
        sources: List[InputSliceDefinition],
        layer_offset: int,
        t: float,
        cfg_reader: ConfigReader,
        module_name: str,
    ):
        module_arch_def = self.arch_info.modules[module_name]
        weights_out: List[WeightInfo] = module_arch_def.architecture.layer_weights(
            index=self._current_module_layers
        )
        weights_in: List[List[WeightInfo]] = [
            module_arch_def.architecture.layer_weights(
                index=s.layer_range[0] + layer_offset
            )
            for s in sources
        ]
        for idx, w_o in enumerate(weights_out):
            self.plan_tensor(
                name=w_o.prefixed_name(prefix=module_arch_def.weight_prefix),
                names_in=[
                    weights_in[j][idx].prefixed_name(
                        prefix=module_arch_def.weight_prefix
                    )
                    for j in range(len(weights_in))
                ],
                models=[s.model for s in sources],
                cfg_reader=cfg_reader.with_t(t),
                tensor_writer=self._tensor_writer(subfolder=module_arch_def.subfolder),
                is_embed=w_o.is_embed,
            )

        self._current_module_layers += 1

    def plan_slice(
        self,
        definition: OutputSliceDefinition,
        module_def: OutputModuleDefinition,
        module_name: str,
    ):
        slice_lengths = [
            s.layer_range[1] - s.layer_range[0] for s in definition.sources
        ]
        if not all(s == slice_lengths[0] for s in slice_lengths):
            raise RuntimeError(
                "All inputs to a slice must contain the same number of layers"
            )
        num_layers = slice_lengths[0]

        cfg_reader = ConfigReader(
            config=self.config, slice_out=definition, module=module_def, t=0
        )
        for idx in range(num_layers):
            # compute t for interpolated gradients
            if num_layers > 1:
                t = idx / (num_layers - 1)
            else:
                t = 1

            self.plan_layer(
                definition.sources,
                layer_offset=idx,
                t=t,
                cfg_reader=cfg_reader,
                module_name=module_name,
            )

    def plan_module(self, module_name: str, definition: OutputModuleDefinition):
        self._current_module_layers = 0

        module_arch_def = self.arch_info.modules[module_name]
        config_reader = ConfigReader(config=self.config, t=0, module=definition)

        for weight_info in module_arch_def.architecture.pre_weights():
            weight_name = weight_info.prefixed_name(
                prefix=module_arch_def.weight_prefix
            )
            self.plan_tensor(
                weight_name,
                [weight_name] * len(definition.slices[0].sources),
                [s.model for s in definition.slices[0].sources],
                config_reader.for_tensor(tensor_name=weight_name).for_out_slice(
                    definition.slices[0]
                ),
                tensor_writer=self._tensor_writer(subfolder=module_arch_def.subfolder),
                is_embed=weight_info.is_embed,
            )

        for out_slice in definition.slices:
            self.plan_slice(
                out_slice,
                module_def=definition,
                module_name=module_name,
            )

        for weight_info in module_arch_def.architecture.post_weights():
            weight_name = weight_info.prefixed_name(
                prefix=module_arch_def.weight_prefix
            )
            self.plan_tensor(
                weight_name,
                [weight_name] * len(definition.slices[-1].sources),
                [s.model for s in definition.slices[-1].sources],
                config_reader.for_tensor(tensor_name=weight_name).for_out_slice(
                    definition.slices[-1]
                ),
                tensor_writer=self._tensor_writer(subfolder=module_arch_def.subfolder),
                is_embed=weight_info.is_embed,
            )

    def plan(self):
        self.normalize_config()
        self._tasks = []

        for module_name in self.config.modules:
            self.plan_module(module_name, self.config.modules[module_name])

        for writer in self._tensor_save_tasks:
            self._tasks.append(
                FinalizeModel(
                    tensor_save_tasks=tuple(self._tensor_save_tasks[writer]),
                    writer_task=writer,
                )
            )
        res = list(self._tasks)
        if self._tokenizer_task:
            res.append(self._tokenizer_task)
        return res
