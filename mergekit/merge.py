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

import logging
from typing import Optional

import torch
import transformers
from pydantic import BaseModel

from mergekit import merge_methods
from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference, parse_kmb
from mergekit.config import MergeConfiguration
from mergekit.graph import Executor, RuleSet
from mergekit.plan import plan
from mergekit.tokenizer import build_tokenizer


class MergeOptions(BaseModel):
    allow_crimes: bool = False
    transformers_cache: Optional[str] = None
    lora_merge_cache: Optional[str] = None
    cuda: bool = False
    low_cpu_memory: bool = False
    out_shard_size: int = parse_kmb("5B")
    copy_tokenizer: bool = True
    allow_crimes: bool = False
    clone_tensors: bool = False
    trust_remote_code: bool = False
    random_seed: Optional[int] = None
    lazy_unpickle: bool = False


def run_merge(merge_config: MergeConfiguration, out_path: str, options: MergeOptions):
    dtype: Optional[torch.dtype] = {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[merge_config.dtype]

    if options.random_seed is not None:
        transformers.trainer_utils.set_seed(options.random_seed)

    if not merge_config.models and not merge_config.slices:
        raise RuntimeError("No output requested")

    method = merge_methods.get(merge_config.merge_method)
    model_arch_info = [
        get_architecture_info(m.config()) for m in merge_config.referenced_models()
    ]
    if not options.allow_crimes:
        if not all(a == model_arch_info[0] for a in model_arch_info[1:]):
            raise RuntimeError(
                "Must specify --allow-crimes to attempt to mix different architectures"
            )
    arch_info = model_arch_info[0]

    if merge_config.tokenizer_source:
        tokenizer, embed_permutations = build_tokenizer(
            merge_config, trust_remote_code=options.trust_remote_code
        )
        tokenizer.save_pretrained(out_path, safe_serialization=True)
    else:
        tokenizer = None
        embed_permutations = None

    (targets, static_rules) = plan(
        merge_config, arch_info, embed_permutations=embed_permutations
    )

    rules = RuleSet(static_rules)
    exec = Executor(
        merge_config.referenced_models(),
        targets,
        rules,
        {"merge": method, "merge_embed": merge_methods.TokenizerPermutationMerge()},
        transformers_cache_dir=options.transformers_cache,
        lora_cache_dir=options.lora_merge_cache,
        dtype=dtype,
        cuda=options.cuda,
        low_cpu_memory=options.low_cpu_memory,
        trust_remote_code=options.trust_remote_code,
        lazy_unpickle=options.lazy_unpickle,
    )
    exec.run(
        out_path,
        max_shard_size=options.out_shard_size,
        clone_tensors=options.clone_tensors,
    )

    cfg_out = method.model_out_config(merge_config)
    if tokenizer:
        try:
            cfg_out.vocab_size = len(tokenizer.get_vocab())
        except Exception as e:
            logging.warning(
                "Unable to set vocabulary size in output config - you may need to manually correct it.",
                exc_info=e,
            )

    try:
        num_layers = sum(
            s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
            for s in merge_config.slices
        )
        setattr(cfg_out, arch_info.num_layers_config_key(), num_layers)
    except Exception as e:
        logging.warning(
            "Unable to set number of layers in output config - you may need to manually correct it.",
            exc_info=e,
        )
    cfg_out.save_pretrained(out_path)

    if options.copy_tokenizer and tokenizer is None:
        try:
            donor_model = merge_config.base_model
            if donor_model:
                donor_model = ModelReference.parse(donor_model)
            if not donor_model:
                donor_model = merge_config.referenced_models()[0]

            transformers.AutoTokenizer.from_pretrained(
                donor_model.path
            ).save_pretrained(out_path, safe_serialization=True)
        except Exception as e:
            logging.error(
                "Failed to save tokenizer. The merge was still successful, just copy it from somewhere else.",
                exc_info=e,
            )
