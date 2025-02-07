# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import os
from typing import Generator, List, Optional

import huggingface_hub
import yaml
from huggingface_hub.utils import HFValidationError
from yaml.nodes import SequenceNode as SequenceNode

from mergekit import merge_methods
from mergekit.config import MergeConfiguration, ModelReference

CARD_TEMPLATE = """---
{metadata}
---
# {name}

This is a merge of pre-trained language models created using [mergekit](https://github.com/cg123/mergekit).

## Merge Details
### Merge Method

This model was merged using the {merge_method} merge method{base_text}.

### Models Merged

The following models were included in the merge:
{model_list}

### Configuration

The following YAML configuration was used to produce this model:

```yaml
{config_yaml}
```
"""

CARD_TEMPLATE_LORA = """---
{metadata}
---
# {name}

This is a LoRA extracted from a language model. It was extracted using [mergekit](https://github.com/arcee-ai/mergekit).

## LoRA Details

{details}

### Parameters

The following command was used to extract this LoRA adapter:

```sh
{invocation}
```
"""


def is_hf(path: str) -> bool:
    """
    Determines if the given path is a Hugging Face model repository.

    Args:
        path: A string path to check.
    """
    if path[0] in "/~" or path.count("/") > 1:
        return False  # definitely a local path
    if not os.path.exists(path):
        return True  # If path doesn't exist locally, it must be a HF repo
    try:
        return huggingface_hub.repo_exists(path, repo_type="model", token=False)
    except HFValidationError:
        return False


def extract_hf_paths(models: List[ModelReference]) -> Generator[str, None, None]:
    """
    Yields all valid Hugging Face paths from a list of ModelReference objects.

    Args:
        models: A list of ModelReference objects.
    """
    for model in models:
        if is_hf(model.model.path):
            yield model.model.path

        if model.lora and is_hf(model.lora.path):
            yield model.lora.path


def method_md(merge_method: str) -> str:
    """
    Returns a markdown string for the given merge method.

    Args:
        merge_method: A string indicating the merge method used.
    """
    try:
        method = merge_methods.get(merge_method)
    except RuntimeError:
        return merge_method
    ref_url = method.reference_url()
    name = method.pretty_name() or method.name()
    if ref_url and ref_url.strip():
        return f"[{name}]({ref_url})"
    return name


def maybe_link_hf(path: str) -> str:
    """
    Convert a path to a clickable link if it's a Hugging Face model path.

    Args:
        path: A string path to possibly convert to a link.
    """
    if is_hf(path):
        return f"[{path}](https://huggingface.co/{path})"
    return path


def modelref_md(model: ModelReference) -> str:
    """
    Generates markdown description for a ModelReference object.

    Args:
        model: A ModelReference object.

    Returns:
        A markdown formatted string describing the model reference.
    """
    text = maybe_link_hf(model.model.path)
    if model.lora:
        text += " + " + maybe_link_hf(model.lora.path)
    return text


def generate_card(
    config: MergeConfiguration,
    config_yaml: str,
    name: Optional[str] = None,
) -> str:
    """
    Generates a markdown card for a merged model configuration.

    Args:
        config: A MergeConfiguration object.
        config_yaml: YAML source text of the config.
        name: An optional name for the model.
    """
    if not name:
        name = "Untitled Model (1)"

    hf_bases = list(extract_hf_paths(config.referenced_models()))
    tags = ["mergekit", "merge"]

    actual_base = config.base_model
    if config.merge_method == "slerp":
        # curse my past self
        actual_base = None

    base_text = ""
    if actual_base:
        base_text = f" using {modelref_md(actual_base)} as a base"

    model_bullets = []
    for model in config.referenced_models():
        if model == actual_base:
            # actual_base is mentioned in base_text - don't include in list
            continue

        model_bullets.append("* " + modelref_md(model))

    return CARD_TEMPLATE.format(
        metadata=yaml.dump(
            {"base_model": hf_bases, "tags": tags, "library_name": "transformers"}
        ),
        model_list="\n".join(model_bullets),
        base_text=base_text,
        merge_method=method_md(config.merge_method),
        name=name,
        config_yaml=config_yaml,
    )


def generate_card_lora(
    base_ref: ModelReference,
    finetuned_ref: ModelReference,
    invocation: str,
    name: str,
    base_vocab_size: Optional[int] = None,
    final_vocab_size: Optional[int] = None,
) -> str:
    if not name:
        name = "Untitled LoRA Model (1)"

    hf_bases = list(extract_hf_paths([base_ref, finetuned_ref]))
    tags = ["mergekit", "peft"]

    details = (
        f"This LoRA adapter was extracted from {modelref_md(finetuned_ref)} "
        f"and uses {modelref_md(base_ref)} as a base."
    )

    if base_vocab_size and final_vocab_size and base_vocab_size != final_vocab_size:
        verb = "extended" if final_vocab_size > base_vocab_size else "reduced"
        details += (
            f"\n\n [!WARNING]\n> The vocabulary size has been {verb} from the base "
            f"model's {base_vocab_size} to {final_vocab_size}. To load this adapter, "
            f"you must first call `model.resize_token_embeddings({final_vocab_size})`."
        )

    return CARD_TEMPLATE_LORA.format(
        metadata=yaml.dump(
            {"base_model": hf_bases, "tags": tags, "library_name": "peft"}
        ),
        name=name,
        details=details,
        invocation=invocation,
    )
