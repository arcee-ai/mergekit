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
from functools import lru_cache
from typing import Any, List, Optional, Tuple

from mergekit import merge_methods
from mergekit.architecture import (
    ArchitectureInfo,
    ConfiguredArchitectureInfo,
    WeightInfo,
)
from mergekit.common import ImmutableMap, ModelReference
from mergekit.config import (
    ConfigReader,
    InputSliceDefinition,
    MergeConfiguration,
    OutputSliceDefinition,
)
from mergekit.graph import Task
from mergekit.io.tasks import (
    FinalizeModel,
    GatherTensors,
    LoaderCache,
    ReturnTensor,
    SaveTensor,
    TensorWriterTask,
)
from mergekit.merge_methods import MergeMethod
from mergekit.options import MergeOptions
from mergekit.tokenizer import BuildTokenizer, PermutedEmbeddings


class MergePlanner:
    config: MergeConfiguration
    arch_info: ArchitectureInfo
    options: MergeOptions
    out_model_config: Any
    _method: MergeMethod
    _tensors: List[Tuple[WeightInfo, Task]]
    _current_layers: int = 0
    _tokenizer_task: Optional[BuildTokenizer] = None

    def __init__(
        self,
        config: MergeConfiguration,
        arch_info: ArchitectureInfo,
        options: MergeOptions,
        out_model_config: Any,
    ):
        self.config = config
        self.arch_info = arch_info
        self.options = options
        self.out_model_config = out_model_config
        self._method = merge_methods.get(config.merge_method)

        token_cfg = {}
        tokenizer_source = config.tokenizer_source
        if config.tokenizer is not None:
            token_cfg = config.tokenizer.tokens or {}
            tokenizer_source = config.tokenizer.source
        if tokenizer_source is not None:
            self._tokenizer_task = BuildTokenizer(
                base_model=config.base_model,
                referenced_models=tuple(config.referenced_models()),
                tokenizer_source=tokenizer_source,
                trust_remote_code=options.trust_remote_code,
                add_tokens=tuple(token_cfg.keys()),
            )

    @lru_cache
    def model_arch_info(self, model: ModelReference):
        return ConfiguredArchitectureInfo(
            info=self.arch_info,
            config=model.config(trust_remote_code=self.options.trust_remote_code),
        )

    def normalize_config(self):
        base_model = self.config.base_model

        # if models to merge are specified instead of output slices, compute them
        if self.config.models:
            if self.config.slices:
                raise RuntimeError(
                    "Must specify either models to merge or output slices"
                )

            slices_in = []
            base_included = False

            for model_in in self.config.models:
                if base_model and model_in.model == base_model:
                    base_included = True

                model_info = self.model_arch_info(model_in.model)
                slices_in.append(
                    InputSliceDefinition(
                        layer_range=[0, model_info.num_layers()],
                        model=model_in.model,
                        parameters=model_in.parameters,
                    )
                )

            if base_model and not base_included:
                logging.info("Base model specified but not in input models - adding")
                base_info = self.model_arch_info(base_model)
                slices_in.append(
                    InputSliceDefinition(
                        layer_range=[0, base_info.num_layers()],
                        model=base_model,
                    )
                )

            self.config.slices = [OutputSliceDefinition(sources=slices_in)]
            self.config.models = None

    def plan_tensor(
        self,
        weight: WeightInfo,
        weights_in: List[WeightInfo],
        models: List[ModelReference],
        cfg_reader: ConfigReader,
    ):
        if weight.optional:
            # check if any input weights are present
            any_weight = False
            for model, w_in in zip(models, weights_in):
                index = LoaderCache().get(model).index
                if w_in.name in index.tensor_paths:
                    any_weight = True
                    break

            if not any_weight:
                logging.info(f"Skipping optional weight {weight.name}")
                return

        tensor_merge_method = self._method
        cfg_g = cfg_reader.for_tensor(weight.name)
        global_params = {}
        for p in tensor_merge_method.parameters():
            global_params[p.name] = cfg_g.parameter(
                p.name, model=None, required=p.required, default=p.default_value
            )

        base_model = cfg_reader.base_model

        tensor_params = {}
        for model, weight_in in zip(models, weights_in):
            is_base = model == base_model
            tensor_params[model] = {}
            cfg_m = cfg_reader.for_tensor(weight_in.name)
            for p in tensor_merge_method.tensor_parameters():
                tensor_params[model][p.name] = cfg_m.parameter(
                    p.name,
                    model=model,
                    required=p.required and not is_base,
                    default=p.default_value,
                )

        gather_tensors = GatherTensors(
            weight_info=ImmutableMap(data=dict(zip(models, weights_in))),
            dtype=self.config.dtype,
            device="cuda" if self.options.read_to_gpu else None,
        )

        tensor_input_task = gather_tensors
        if self._tokenizer_task and weight.is_embed:
            token_cfg = {}
            if cfg_reader.config.tokenizer:
                token_cfg = cfg_reader.config.tokenizer.tokens
            tensor_input_task = PermutedEmbeddings(
                gather_tensors=gather_tensors,
                tokenizer_task=self._tokenizer_task,
                tokens=token_cfg,
                base_model=base_model,
            )

        tensor_task = tensor_merge_method.make_task(
            output_weight=weight,
            tensors=tensor_input_task,
            parameters=ImmutableMap(data=global_params),
            tensor_parameters=ImmutableMap(
                data={
                    key: ImmutableMap(data=tensor_params[key]) for key in tensor_params
                }
            ),
            base_model=base_model,
        )
        self._tensors.append((weight, tensor_task))

    def plan_layer(
        self,
        sources: List[InputSliceDefinition],
        layer_offset: int,
        t: float,
        cfg_reader: ConfigReader,
    ):
        weights_out: List[WeightInfo] = self.arch_info.layer_weights(
            index=self._current_layers,
            config=self.out_model_config,
        )
        weights_in: List[List[WeightInfo]] = [
            self.model_arch_info(s.model).layer_weights(
                index=s.layer_range[0] + layer_offset
            )
            for s in sources
        ]

        for idx, w_o in enumerate(weights_out):
            self.plan_tensor(
                weight=w_o,
                weights_in=[weights_in[j][idx] for j in range(len(weights_in))],
                models=[s.model for s in sources],
                cfg_reader=cfg_reader.with_t(t),
            )

        self._current_layers += 1

    def plan_slice(self, definition: OutputSliceDefinition):
        slice_lengths = [
            s.layer_range[1] - s.layer_range[0] for s in definition.sources
        ]
        if not all(s == slice_lengths[0] for s in slice_lengths):
            raise RuntimeError(
                "All inputs to a slice must contain the same number of layers"
            )
        num_layers = slice_lengths[0]

        cfg_reader = ConfigReader(config=self.config, slice_out=definition, t=0)
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
            )

    def plan_to_disk(self, out_path: str) -> List[Task]:
        """Plan the merge to be streamed to disk, returning a list of tasks."""
        self._plan()

        writer_task = TensorWriterTask(
            out_path=out_path,
            max_shard_size=self.options.out_shard_size,
            safe_serialization=self.options.safe_serialization,
        )
        save_tasks = []
        for weight, tensor_task in self._tensors:
            save_tasks.append(
                SaveTensor(
                    tensor_name=weight.name,
                    tensor_task=tensor_task,
                    writer_task=writer_task,
                    clone=self.options.clone_tensors,
                    optional=weight.optional,
                    dtype=weight.force_dtype or self.config.out_dtype,
                )
            )
        finalize = FinalizeModel(
            tensor_save_tasks=tuple(save_tasks), writer_task=writer_task
        )

        res = save_tasks + [finalize]
        if self._tokenizer_task:
            res.append(self._tokenizer_task)
        return res

    def plan_in_memory(self) -> List[ReturnTensor]:
        """Plan the merge to be performed in memory."""
        self._plan()
        return [
            ReturnTensor(
                weight_info=w,
                tensor_task=t,
                dtype=w.force_dtype or self.config.out_dtype,
            )
            for w, t in self._tensors
        ]

    def _plan(self):
        self.normalize_config()
        self._tensors = []

        for weight_info in self.arch_info.pre_weights(config=self.out_model_config):
            self.plan_tensor(
                weight_info,
                [weight_info] * len(self.config.slices[0].sources),
                [s.model for s in self.config.slices[0].sources],
                ConfigReader(
                    config=self.config,
                    t=0,
                    tensor_name=weight_info.name,
                ).for_out_slice(self.config.slices[0]),
            )

        for out_slice in self.config.slices:
            self.plan_slice(out_slice)

        for weight_info in self.arch_info.post_weights(config=self.out_model_config):
            self.plan_tensor(
                weight_info,
                [weight_info] * len(self.config.slices[-1].sources),
                [s.model for s in self.config.slices[-1].sources],
                ConfigReader(
                    config=self.config,
                    t=1,
                    tensor_name=weight_info.name,
                ).for_out_slice(self.config.slices[-1]),
            )
