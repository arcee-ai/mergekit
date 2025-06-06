# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import importlib
import importlib.resources
import logging
import os
import shutil
from collections import Counter
from typing import List, Optional, Tuple

import tqdm
import transformers

from mergekit._data import chat_templates
from mergekit.architecture import ModelArchitecture, get_architecture_info
from mergekit.card import generate_card
from mergekit.common import ModelReference, set_config_value
from mergekit.config import MergeConfiguration
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache
from mergekit.multigpu_executor import MultiGPUExecutor
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner
from mergekit.tokenizer import TokenizerInfo

LOG = logging.getLogger(__name__)


def run_merge(
    merge_config: MergeConfiguration,
    out_path: str,
    options: MergeOptions,
    config_source: Optional[str] = None,
):
    if options.random_seed is not None:
        transformers.trainer_utils.set_seed(options.random_seed)

    if not merge_config.models and not merge_config.slices and not merge_config.modules:
        raise RuntimeError("No output requested")

    arch_info = get_architecture_info(merge_config, options)

    # initialize loader cache and set options
    loader_cache = LoaderCache()
    loader_cache.setup(options=options)

    # create config for output model
    cfg_out = _model_out_config(
        merge_config, arch_info, trust_remote_code=options.trust_remote_code
    )

    # warm up loader cache
    for model in (
        pbar := tqdm.tqdm(
            merge_config.referenced_models(),
            desc="Warmup loader cache",
            disable=options.quiet,
        )
    ):
        loader_cache.get(model)
    del pbar

    LOG.info("Planning operations")
    targets = MergePlanner(
        merge_config,
        arch_info,
        options=options,
        out_model_config=cfg_out,
    ).plan_to_disk(out_path=out_path)

    if options.multi_gpu:
        exec = MultiGPUExecutor(
            targets=targets,
            storage_device=None if options.low_cpu_memory else "cpu",
        )
    else:
        exec = Executor(
            targets=targets,
            math_device=options.device,
            storage_device=options.device if options.low_cpu_memory else "cpu",
        )

    tokenizer = None
    for _task, value in exec.run(quiet=options.quiet):
        if isinstance(value, TokenizerInfo):
            tokenizer = value.tokenizer

    if tokenizer:
        pad_to_multiple_of = None
        if merge_config.tokenizer and merge_config.tokenizer.pad_to_multiple_of:
            pad_to_multiple_of = merge_config.tokenizer.pad_to_multiple_of
        _update_config_vocab(
            cfg_out, arch_info, tokenizer, pad_to_multiple_of=pad_to_multiple_of
        )

    LOG.info("Saving config")
    cfg_out.save_pretrained(out_path)

    if options.write_model_card:
        if not config_source:
            config_source = merge_config.to_yaml()

        card_md = generate_card(
            config=merge_config,
            config_yaml=config_source,
            name=os.path.basename(out_path),
        )
        with open(os.path.join(out_path, "README.md"), "w", encoding="utf-8") as fp:
            fp.write(card_md)

        with open(
            os.path.join(out_path, "mergekit_config.yml"), "w", encoding="utf-8"
        ) as fp:
            fp.write(config_source)

    if tokenizer is not None:
        LOG.info("Saving tokenizer")
        _set_chat_template(tokenizer, merge_config)
        tokenizer.save_pretrained(out_path, safe_serialization=True)
    else:
        if options.copy_tokenizer:
            try:
                _copy_tokenizer(merge_config, out_path, options=options)
            except Exception as e:
                LOG.error(
                    "Failed to copy tokenizer. The merge was still successful, just copy it from somewhere else.",
                    exc_info=e,
                )
        elif merge_config.chat_template:
            LOG.warning(
                "Chat template specified but no tokenizer found. Chat template will not be saved."
            )

    _copy_tagalong_files(
        merge_config,
        out_path,
        files=arch_info.tagalong_files or [],
        options=options,
    )

    if getattr(arch_info, "post_fill_parameters", False):
        from mergekit.scripts.fill_missing_params import copy_and_fill_missing_params

        logging.info(
            f"Filling missing parameters from base model {arch_info.post_fill_parameters} into new directory"
        )
        copy_and_fill_missing_params(
            base_model_repo_id=arch_info.post_fill_parameters,
            sub_model_dir=out_path,
        )
        logging.info("Deleting initial merge directory: " + out_path)
        shutil.rmtree(out_path)


def _set_chat_template(
    tokenizer: transformers.PreTrainedTokenizerBase,
    merge_config: MergeConfiguration,
    trust_remote_code: bool = False,
):
    chat_template = merge_config.chat_template
    if not chat_template:
        return

    if chat_template == "auto":
        # see if there is a plurality chat template among the input models
        model_templates = []
        for model in merge_config.referenced_models():
            try:
                tok = transformers.AutoTokenizer.from_pretrained(
                    model.model.path,
                    revision=model.model.revision,
                    trust_remote_code=trust_remote_code,
                )
                template = tok.chat_template
                if isinstance(template, dict):
                    template = template.get("default", None)
                if template:
                    model_templates.append(template.strip())
            except Exception as e:
                LOG.warning(f"Unable to load tokenizer for {model}", exc_info=e)

        if not model_templates:
            return

        chat_template = Counter(model_templates).most_common(1)[0][0]
        LOG.info(f"Auto-selected chat template: {chat_template}")

    elif (
        t := importlib.resources.files(chat_templates).joinpath(
            chat_template + ".jinja"
        )
    ).is_file():
        chat_template = t.read_text()

    elif len(chat_template) < 20 or "{" not in chat_template:
        raise RuntimeError(f"Invalid chat template: {chat_template}")

    tokenizer.chat_template = chat_template


def _get_donor_model(
    merge_config: MergeConfiguration,
    options: MergeOptions,
) -> Tuple[ModelReference, str]:
    donor_model = merge_config.base_model or (merge_config.referenced_models()[0])
    donor_local_path = donor_model.merged(
        cache_dir=options.lora_merge_cache,
        trust_remote_code=options.trust_remote_code,
        lora_merge_dtype=options.lora_merge_dtype,
    ).local_path(cache_dir=options.transformers_cache)
    if not donor_local_path:
        raise RuntimeError(f"Unable to find local path for {donor_model}")
    return donor_model, donor_local_path


def _copy_tagalong_files(
    merge_config: MergeConfiguration,
    out_path: str,
    files: List[str],
    options: MergeOptions,
):
    donor_model, donor_local_path = _get_donor_model(merge_config, options=options)

    for file_name in files:
        fp = os.path.join(donor_local_path, file_name)
        if os.path.exists(fp):
            LOG.info(f"Copying {file_name} from {donor_model}")
            shutil.copy(
                fp,
                os.path.join(out_path, file_name),
            )

    return


def _copy_tokenizer(
    merge_config: MergeConfiguration, out_path: str, options: MergeOptions
):
    donor_model, donor_local_path = _get_donor_model(merge_config, options=options)

    if (
        (not merge_config.chat_template)
        and os.path.exists(os.path.join(donor_local_path, "tokenizer_config.json"))
        and (
            os.path.exists(os.path.join(donor_local_path, "tokenizer.json"))
            or os.path.exists(os.path.join(donor_local_path, "tokenizer.model"))
        )
    ):
        LOG.info(f"Copying tokenizer from {donor_model}")

        for file_name in [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "added_tokens.json",
            "merges.txt",
        ]:
            if os.path.exists(os.path.join(donor_local_path, file_name)):
                shutil.copy(
                    os.path.join(donor_local_path, file_name),
                    os.path.join(out_path, file_name),
                )

        return

    # fallback: try actually loading the tokenizer and saving it
    LOG.info(f"Reserializing tokenizer from {donor_model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        donor_model.model.path,
        revision=donor_model.model.revision,
        trust_remote_code=options.trust_remote_code,
    )
    _set_chat_template(tokenizer, merge_config)
    tokenizer.save_pretrained(out_path, safe_serialization=True)


def _model_out_config(
    config: MergeConfiguration,
    arch_info: ModelArchitecture,
    trust_remote_code: bool = False,
) -> transformers.PretrainedConfig:
    """Return a configuration for the resulting model."""
    if config.base_model:
        res = config.base_model.config(trust_remote_code=trust_remote_code)
    else:
        res = config.referenced_models()[0].config(trust_remote_code=trust_remote_code)
    if config.out_dtype:
        res.torch_dtype = config.out_dtype
    elif config.dtype:
        res.torch_dtype = config.dtype

    module_layers = {}
    for module_name in arch_info.modules:
        if config.modules and module_name in config.modules:
            module_def = config.modules.get(module_name)
            if module_def and module_def.slices:
                module_layers[module_name] = sum(
                    [
                        s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
                        for s in module_def.slices
                    ]
                )
        elif config.slices:
            module_layers[module_name] = sum(
                [
                    s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
                    for s in config.slices
                ]
            )

    if module_layers:
        for module_name in module_layers:
            if module_name not in arch_info.modules:
                LOG.warning(
                    f"Module {module_name} in config but not in architecture info"
                )
                continue
            module_info = arch_info.modules[module_name]
            cfg_key = module_info.architecture.num_layers_config_key()
            if not cfg_key:
                if module_layers[module_name] > 0:
                    LOG.warning(
                        f"Module {module_name} has no configuration key for number of layers, "
                        "but the number of layers is not zero."
                    )
                continue
            try:
                set_config_value(res, cfg_key, module_layers[module_name])
            except Exception as e:
                LOG.warning(
                    f"Unable to set number of layers for module {module_name} in output config "
                    "- you may need to manually correct it.",
                    exc_info=e,
                )

    return res


def _update_config_vocab(
    config: transformers.PretrainedConfig,
    arch_info: ModelArchitecture,
    tokenizer: transformers.PreTrainedTokenizerBase,
    pad_to_multiple_of: Optional[int] = None,
):
    vocab_size = len(tokenizer.get_vocab())
    if pad_to_multiple_of and vocab_size % pad_to_multiple_of:
        vocab_size = vocab_size + pad_to_multiple_of - (vocab_size % pad_to_multiple_of)
    try:
        set_config_value(
            config, arch_info.vocab_size_config_key or "vocab_size", vocab_size
        )
    except Exception as e:
        LOG.warning(
            "Unable to set vocabulary size in output config - you may need to manually correct it.",
            exc_info=e,
        )


__all__ = ["MergeOptions", "run_merge"]
