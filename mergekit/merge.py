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

import importlib
import importlib.resources
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional

import tqdm
import transformers
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from mergekit._data import chat_templates
from mergekit.architecture import (
    ArchitectureInfo,
    AutomaticArchitectureInfo,
    get_architecture_info,
)
from mergekit.card import generate_card
from mergekit.config import MergeConfiguration
from mergekit.graph import Executor
from mergekit.io.lazy_tensor_loader import ShardedTensorIndex
from mergekit.io.tasks import LoaderCache
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner
from mergekit.tokenizer import TokenizerInfo

# Overwritten by the environment variable HF_HOME if set
HF_HOME_DEFAULT = "~/.cache/huggingface"


def run_merge(
    merge_config: MergeConfiguration,
    out_path: str,
    options: MergeOptions,
    config_source: Optional[str] = None,
):
    if options.random_seed is not None:
        transformers.trainer_utils.set_seed(options.random_seed)

    if not merge_config.models and not merge_config.slices:
        raise RuntimeError("No output requested")

    arch_info = load_model_architecture(merge_config, options)

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

    logging.info("Planning operations")
    targets = MergePlanner(
        merge_config,
        arch_info,
        options=options,
        out_model_config=cfg_out,
    ).plan_to_disk(out_path=out_path)

    exec = Executor(
        tasks=targets,
        math_device="cuda" if options.cuda else "cpu",
        storage_device="cuda" if options.low_cpu_memory else "cpu",
    )

    tokenizer = None
    for _task, value in exec.run(quiet=options.quiet):
        if isinstance(value, TokenizerInfo):
            tokenizer = value.tokenizer

    if tokenizer:
        _update_config_vocab(cfg_out, tokenizer)

    logging.info("Saving config")
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

    if tokenizer is None:
        if options.copy_tokenizer:
            try:
                _copy_tokenizer(
                    merge_config, out_path, trust_remote_code=options.trust_remote_code
                )
            except Exception as e:
                logging.error(
                    "Failed to copy tokenizer. The merge was still successful, just copy it from somewhere else.",
                    exc_info=e,
                )
        elif merge_config.chat_template:
            logging.warning(
                "Chat template specified but no tokenizer found. Chat template will not be saved."
            )

    if tokenizer:
        logging.info("Saving tokenizer")
        _set_chat_template(tokenizer, merge_config)
        tokenizer.save_pretrained(out_path, safe_serialization=True)


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
                logging.warning(f"Unable to load tokenizer for {model}", exc_info=e)

        if not model_templates:
            return

        chat_template = Counter(model_templates).most_common(1)[0][0]
        logging.info(f"Auto-selected chat template: {chat_template}")

    elif importlib.resources.is_resource(chat_templates, chat_template + ".jinja"):
        with importlib.resources.open_text(
            chat_templates, chat_template + ".jinja"
        ) as fp:
            chat_template = fp.read()

    elif len(chat_template) < 20 or "{" not in chat_template:
        raise RuntimeError(f"Invalid chat template: {chat_template}")

    tokenizer.chat_template = chat_template


def _copy_tokenizer(
    merge_config: MergeConfiguration, out_path: str, trust_remote_code: bool = False
):
    donor_model = merge_config.base_model or (merge_config.referenced_models()[0])

    if (
        (not merge_config.chat_template)
        and os.path.exists(
            os.path.join(donor_model.model.path, "tokenizer_config.json")
        )
        and (
            os.path.exists(os.path.join(donor_model.model.path, "tokenizer.json"))
            or os.path.exists(os.path.join(donor_model.model.path, "tokenizer.model"))
        )
    ):
        logging.info(f"Copying tokenizer from {donor_model}")

        for file_name in [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
        ]:
            if os.path.exists(os.path.join(donor_model.model.path, file_name)):
                shutil.copy(
                    os.path.join(donor_model.model.path, file_name),
                    os.path.join(out_path, file_name),
                )

        return

    # fallback: try actually loading the tokenizer and saving it
    logging.info(f"Reserializing tokenizer from {donor_model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        donor_model.model.path,
        revision=donor_model.model.revision,
        trust_remote_code=trust_remote_code,
    )
    _set_chat_template(tokenizer, merge_config)
    tokenizer.save_pretrained(out_path, safe_serialization=True)


def _model_out_config(
    config: MergeConfiguration,
    arch_info: ArchitectureInfo,
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

    if config.slices:
        try:
            num_layers = sum(
                s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
                for s in config.slices
            )
            setattr(res, arch_info.num_layers_config_key(), num_layers)
        except Exception as e:
            logging.warning(
                "Unable to set number of layers in output config - you may need to manually correct it.",
                exc_info=e,
            )

    return res


def _update_config_vocab(
    config: transformers.PretrainedConfig,
    tokenizer: transformers.PreTrainedTokenizerBase,
):
    try:
        config.vocab_size = len(tokenizer.get_vocab())
    except Exception as e:
        logging.warning(
            "Unable to set vocabulary size in output config - you may need to manually correct it.",
            exc_info=e,
        )


def load_model_architecture(merge_config, options):
    model_arch_info = [
        get_architecture_info(m.config(trust_remote_code=options.trust_remote_code))
        for m in merge_config.referenced_models()
    ]
    if any(a is False for a in model_arch_info):
        # Attempt to load the architecture automatically if it's not specified
        model_arch_info = [
            AutomaticArchitectureInfo(
                arch_name=source_model.model.path,
                parameter_names=_get_model_parameter_names(source_model.model.path),
            )
            for source_model in merge_config.referenced_models()
        ]
        if not all(
            a.all_weights(None) == model_arch_info[0].all_weights(None)
            for a in model_arch_info[1:]
        ):
            raise RuntimeError(
                "AutomaticArchitectureInfo only supports models with the same architecture"
            )
    else:
        if not options.allow_crimes and not all(
            a == model_arch_info[0] for a in model_arch_info[1:]
        ):
            raise RuntimeError(
                "Must specify --allow-crimes to attempt to mix different architectures"
            )

    return model_arch_info[0]


def _get_model_parameter_names(repo_id: str) -> list:
    """
    Get the parameter names of a model from a Hugging Face repo or local directory.

    This function checks if the model is available locally or in the Hugging Face cache.
    If the model is not available, it attempts to download it. If the download fails,
    it raises an error. Once the model is resolved, it returns the list of tensor paths.

    :param repo_id: The model's repo ID, URL, or local directory path.
    :return: A list of parameter names.
    """
    # Try to resolve the model directory, either locally or by downloading
    model_dir = _resolve_model_directory(repo_id)

    # Attempt to get the tensor paths from the resolved directory
    return list(ShardedTensorIndex.from_disk(str(model_dir)).tensor_paths.keys())


def _resolve_model_directory(repo_id: str) -> Path:
    """
    Resolve the model directory either from a local path, URL, or by downloading from Hugging Face.

    :param repo_id: The model's repo ID, URL, or local directory path.
    :return: The path to the resolved model directory.
    """
    if Path(repo_id).is_dir():
        # If it's a local directory, return the path
        return Path(repo_id)

    try:
        # Use Hugging Face snapshot_download to check cache or download the model
        return Path(snapshot_download(repo_id))
    except HfHubHTTPError:
        raise ValueError(f"Model {repo_id} not found on Hugging Face Hub.")
    except Exception as e:
        raise ValueError(f"Error locating model {repo_id}: {e}")


__all__ = ["MergeOptions", "run_merge"]
