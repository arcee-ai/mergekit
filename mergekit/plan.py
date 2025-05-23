# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from functools import lru_cache
from typing import Any, List, Optional, Tuple

from mergekit import merge_methods
from mergekit.architecture import (
    ConfiguredModuleArchitecture,
    ModelArchitecture,
    WeightInfo,
)
from mergekit.architecture.base import ConfiguredModelArchitecture
from mergekit.common import ImmutableMap, ModelReference
from mergekit.config import (
    ConfigReader,
    InputSliceDefinition,
    MergeConfiguration,
    OutputModuleDefinition,
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
    arch_info: ModelArchitecture
    options: MergeOptions
    out_model_config: Any
    _method: MergeMethod
    _tensors: List[Tuple[WeightInfo, Task]]
    _current_module_layers: int = 0
    _tokenizer_task: Optional[BuildTokenizer] = None

    def __init__(
        self,
        config: MergeConfiguration,
        arch_info: ModelArchitecture,
        options: MergeOptions,
        out_model_config: Any,
    ):
        self.config = config
        self.arch_info = arch_info
        self.options = options
        self.out_model_config = out_model_config
        self._method = merge_methods.get(config.merge_method)
        self._tensors = []

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

    def _out_module_arch(self, module: str) -> ConfiguredModuleArchitecture:
        module_def = self.arch_info.modules[module]
        return ConfiguredModuleArchitecture(
            info=module_def.architecture,
            config=self.out_model_config,
            weight_prefix=module_def.weight_prefix,
        )

    @lru_cache
    def _model_arch(self, model: ModelReference):
        return ConfiguredModelArchitecture(
            info=self.arch_info,
            config=model.config(trust_remote_code=self.options.trust_remote_code),
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
                    "Model has multiple modules, must use modules: config syntax "
                    "to work with slices"
                )
            module_name = list(self.arch_info.modules.keys())[0]
            self.config.modules = {
                module_name: OutputModuleDefinition(slices=self.config.slices)
            }
            self.config.slices = None

        # modules.models -> modules.slices
        for module_name in self.config.modules:
            module_out = self.config.modules[module_name]
            module_arch = self.arch_info.modules[module_name].architecture

            if module_out.models:
                slices_in = []
                base_included = False

                for model_in in module_out.models:
                    if base_model and model_in.model == base_model:
                        base_included = True

                    model_cfg = model_in.model.config(
                        trust_remote_code=self.options.trust_remote_code
                    )
                    num_layers = module_arch.num_layers(model_cfg)
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
                    num_layers = module_arch.num_layers(base_cfg)
                    slices_in.append(
                        InputSliceDefinition(
                            layer_range=[0, num_layers],
                            model=base_model,
                        )
                    )

                module_out.slices = [OutputSliceDefinition(sources=slices_in)]
                module_out.models = None

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
                if any(
                    name in index.tensor_paths
                    for name in [w_in.name] + (w_in.aliases or [])
                ):
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
            device=self.options.device if self.options.read_to_gpu else None,
        )

        tensor_input_task = gather_tensors
        if self._tokenizer_task and weight.is_embed:
            token_cfg = {}
            pad_to_multiple = None
            if cfg_reader.config.tokenizer:
                token_cfg = cfg_reader.config.tokenizer.tokens
                pad_to_multiple = cfg_reader.config.tokenizer.pad_to_multiple_of
            tensor_input_task = PermutedEmbeddings(
                gather_tensors=gather_tensors,
                tokenizer_task=self._tokenizer_task,
                tokens=token_cfg,
                pad_to_multiple_of=pad_to_multiple,
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
        module_name: str,
    ):
        module_arch = self._out_module_arch(module_name)
        weights_out: List[WeightInfo] = module_arch.layer_weights(
            index=self._current_module_layers,
        )
        weights_in: List[List[WeightInfo]] = [
            self._model_arch(s.model)
            .get_module(module_name)
            .layer_weights(index=s.layer_range[0] + layer_offset)
            for s in sources
        ]

        for idx, w_o in enumerate(weights_out):
            self.plan_tensor(
                weight=w_o,
                weights_in=[weights_in[j][idx] for j in range(len(weights_in))],
                models=[s.model for s in sources],
                cfg_reader=cfg_reader.with_t(t),
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
            config=self.config, slice_out=definition, t=0, module=module_def
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

        module_arch = self._out_module_arch(module_name)
        config_reader = ConfigReader(config=self.config, t=0, module=definition)

        for weight_info in module_arch.pre_weights():
            self.plan_tensor(
                weight_info,
                [weight_info] * len(definition.slices[0].sources),
                [s.model for s in definition.slices[0].sources],
                config_reader.for_tensor(tensor_name=weight_info.name).for_out_slice(
                    definition.slices[0]
                ),
            )

        for out_slice in definition.slices:
            self.plan_slice(
                out_slice,
                module_def=definition,
                module_name=module_name,
            )

        for weight_info in module_arch.post_weights():
            self.plan_tensor(
                weight_info,
                [weight_info] * len(definition.slices[0].sources),
                [s.model for s in definition.slices[-1].sources],
                config_reader.for_tensor(tensor_name=weight_info.name).for_out_slice(
                    definition.slices[-1]
                ),
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
        self._tasks = []

        for module_name in self.config.modules:
            self.plan_module(module_name, self.config.modules[module_name])
