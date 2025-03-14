# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import re
from collections import defaultdict
from typing import List, Optional

from mergekit.architecture.base import (
    ModelArchitecture,
    ModuleDefinition,
    WeightInfo,
)
from mergekit.architecture.json_definitions import (
    JsonLayerTemplates,
    JsonModuleArchDef,
    JsonModuleArchitecture,
)
from mergekit.common import ModelReference
from mergekit.options import MergeOptions

RE_LAYER_INDEX = re.compile(r"\.(\d+)\.")

logger = logging.getLogger(__name__)


def get_model_tensor_names(model: ModelReference, options: MergeOptions) -> List[str]:
    loader = model.lazy_loader(
        cache_dir=options.transformers_cache, lazy_unpickle=options.lazy_unpickle
    )
    return list(loader.index.tensor_paths.keys())


def infer_architecture_info(
    models: List[ModelReference],
    base_model: Optional[ModelReference],
    options: MergeOptions,
) -> ModelArchitecture:
    model_tensor_names = {
        model: set(get_model_tensor_names(model, options))
        for model in (set(models).union({base_model} if base_model else {}))
    }
    if base_model is None:
        base_model = models.pop(0)
    all_tensor_names = set().union(*model_tensor_names.values())
    in_all_models = all_tensor_names.intersection(*model_tensor_names.values())

    module_prefixes = set()
    module_layer_counts = defaultdict(int)
    module_templates = defaultdict(set)
    module_loose_weights = defaultdict(set)
    for tensor_name in all_tensor_names:
        if len(RE_LAYER_INDEX.findall(tensor_name)) > 1:
            raise ValueError(
                f"Tensor name {tensor_name} has more than one layer index - not supported"
            )
        elif match := RE_LAYER_INDEX.search(tensor_name):
            prefix = tensor_name[: match.start()]
            module_prefixes.add(prefix)
            layer_idx = int(match.group(1))
            module_layer_counts[prefix] = max(
                module_layer_counts[prefix], layer_idx + 1
            )
            module_templates[prefix] = module_templates[prefix].union(
                set([RE_LAYER_INDEX.sub("{layer_index}", tensor_name)])
            )

    # create a default module with no prefix
    module_prefixes.add("")

    for tensor_name in all_tensor_names:
        if RE_LAYER_INDEX.search(tensor_name):
            continue
        for prefix in module_prefixes:
            if tensor_name.startswith(prefix):
                module_loose_weights[prefix].add(tensor_name[len(prefix) :])

    if not (module_loose_weights[""] or module_templates[""]):
        module_prefixes.remove("")
    if not module_prefixes:
        raise ValueError("No modules found in models")

    logging.warning(f"Inferred {len(module_prefixes)} modules:")
    for prefix in module_prefixes:
        logging.warning(
            f"  {repr(prefix or 'default')} with {module_layer_counts[prefix]} layers, {len(module_templates[prefix])} templates, and {len(module_loose_weights[prefix])} loose weights"
        )

    def _wi(template: str) -> WeightInfo:
        optional = template.replace("{layer_index}", "0") not in in_all_models
        return WeightInfo(
            name=template,
            optional=optional,
        )

    module_archs = {}
    for prefix in module_prefixes:
        num_layers = module_layer_counts[prefix]
        module_archs[prefix or "default"] = JsonModuleArchitecture(
            definition=JsonModuleArchDef(
                model_type="",
                architectures=[],
                pre_weights=[_wi(t) for t in module_loose_weights[prefix]],
                layer_templates=JsonLayerTemplates(
                    weights=[_wi(t) for t in module_templates[prefix]]
                ),
                post_weights=[],
                num_layers_config_key=None,
                override_num_layers=num_layers,
            ),
        )

    return ModelArchitecture(
        modules={
            key: ModuleDefinition(architecture=value)
            for key, value in module_archs.items()
        },
        architectures=[],
        model_type="",
    )
