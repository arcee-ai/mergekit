# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import re
from collections import defaultdict
from functools import lru_cache
from typing import List, Optional, Tuple

import torch

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
from mergekit.common import ModelReference, get_auto_cls
from mergekit.options import MergeOptions

try:
    from transformers.modeling_utils import _get_tied_weight_keys
except ImportError:
    _get_tied_weight_keys = None

RE_LAYER_INDEX = re.compile(r"\.(\d+)\.")

LOG = logging.getLogger(__name__)


def get_model_tensor_names(model: ModelReference, options: MergeOptions) -> List[str]:
    loader = model.lazy_loader(
        cache_dir=options.transformers_cache, lazy_unpickle=options.lazy_unpickle
    )
    return list(loader.index.tensor_paths.keys())


def get_transformers_info(model: ModelReference, options: MergeOptions) -> tuple:
    try:
        cfg = model.config(
            trust_remote_code=options.trust_remote_code,
        )
        auto_cls = get_auto_cls(cfg.architectures[0])
    except Exception as e:
        LOG.warning(
            f"Unable to load config for {model.model} - tied/ignored weights will not be detected",
            exc_info=e,
        )
        return None, None, None
    try:
        with torch.device("meta"):
            model = auto_cls.from_pretrained(
                model.model.path,
                revision=model.model.revision,
                trust_remote_code=options.trust_remote_code,
                device_map="meta",
            )
    except Exception as e:
        LOG.warning(
            f"Unable to load model {model.model} with transformers - tied/ignored weights will not be detected",
            exc_info=e,
        )
        return None, None, None

    ignore_on_save = getattr(model, "_keys_to_ignore_on_save", None)
    if _get_tied_weight_keys is None:
        LOG.warning(
            "Unable to get tied weights - incompatible transformers version",
        )
        tied_keys = None
    else:
        tied_keys = _get_tied_weight_keys(model)
    if ignore_on_save is not None:
        ignore_on_save = set(ignore_on_save)

    embed_names = set()
    _embed_out = model.get_output_embeddings()
    _embed_in = model.get_input_embeddings()
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.Embedding)
            or module == _embed_out
            or module == _embed_in
        ):
            embed_names.add(name + ".weight")
    return ignore_on_save, tied_keys, embed_names


@lru_cache(maxsize=128)
def infer_architecture_info(
    models: Tuple[ModelReference, ...],
    base_model: Optional[ModelReference],
    options: MergeOptions,
) -> ModelArchitecture:
    model_tensor_names = {
        model: set(get_model_tensor_names(model, options))
        for model in (set(models).union({base_model} if base_model else {}))
    }
    models = list(models)
    if base_model is None:
        base_model = models.pop(0)
    all_tensor_names = set().union(*model_tensor_names.values())
    in_all_models = all_tensor_names.intersection(*model_tensor_names.values())

    ignore_on_save, tied_keys, embed_names = get_transformers_info(base_model, options)

    module_prefixes = set()
    module_layer_counts = defaultdict(int)
    module_templates = defaultdict(set)
    module_loose_weights = defaultdict(set)

    # capture prefixes and layer weight templates
    for tensor_name in all_tensor_names:
        if ignore_on_save and tensor_name in ignore_on_save:
            continue
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
                set([RE_LAYER_INDEX.sub(".${layer_index}.", tensor_name)])
            )

    if len(module_prefixes) == 1:
        # if there's only one prefix, put everything in the default module
        prefix = module_prefixes.pop()
        module_templates = {"": module_templates[prefix]}
        module_layer_counts = {"": module_layer_counts[prefix]}
        module_loose_weights = {"": module_loose_weights[prefix]}
        module_prefixes = {""}
    else:
        module_prefixes.add("")

    for tensor_name in all_tensor_names:
        if RE_LAYER_INDEX.search(tensor_name):
            continue
        for prefix in module_prefixes:
            if tensor_name.startswith(prefix):
                module_loose_weights[prefix].add(tensor_name[len(prefix) :])
                break

    if "" in module_prefixes and not (module_loose_weights[""] or module_templates[""]):
        module_prefixes.remove("")
    if not module_prefixes:
        raise ValueError("No modules found in models")

    logging.warning(f"Inferred {len(module_prefixes)} modules:")
    for prefix in module_prefixes:
        logging.warning(
            f"  {repr(prefix or 'default')} with {module_layer_counts[prefix]} layers, {len(module_templates[prefix])} templates, and {len(module_loose_weights[prefix])} loose weights"
        )

    def _wi(template: str, prefix: str) -> WeightInfo:
        full_name = prefix + template
        optional = (full_name.replace("${layer_index}", "0") not in in_all_models) or (
            tied_keys is not None
            and any(re.search(pat, full_name) for pat in tied_keys)
        )
        is_embed = (full_name in embed_names) or any(
            re.search(pat, full_name) for pat in tied_keys
        )  # strictly speaking you can have tied non-embedding/lm-head weights
        # but i've never seen it so let's not worry about it until this breaks something
        return WeightInfo(
            name=template,
            optional=optional,
            is_embed=is_embed,
        )

    module_archs = {}
    for prefix in module_prefixes:
        num_layers = module_layer_counts[prefix]
        module_archs[prefix or "default"] = JsonModuleArchitecture(
            definition=JsonModuleArchDef(
                model_type="",
                architectures=[],
                pre_weights=[_wi(t, "") for t in module_loose_weights[prefix]],
                layer_templates=JsonLayerTemplates(
                    weights=[_wi(t, "") for t in module_templates[prefix]]
                ),
                post_weights=[],
                num_layers_config_key=None,
                override_num_layers=num_layers,
            ),
        )

    res = ModelArchitecture(
        modules={
            key: ModuleDefinition(architecture=value)
            for key, value in module_archs.items()
        },
        architectures=[],
        model_type="",
    )
    return res
