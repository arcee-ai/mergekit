import copy
import json
import os.path
from typing import Dict, List, Optional
from typing_extensions import Annotated

import safetensors.torch
import torch
import transformers
import typer
import yaml
from pydantic import BaseModel
from tqdm import tqdm

from common import LLAMA_LAYER_MEMBERS, ModelReference
from lazy_tensors import LazyTensorLoader, TensorWriter


class LayerSlice(BaseModel):
    model: str
    start: int
    end: int
    scale: Optional[float] = None


class BakllamaConfig(BaseModel):
    layer_slices: List[LayerSlice]
    embedding_source: Optional[str] = None
    lm_head_source: Optional[str] = None


def process(
    config: BakllamaConfig,
    out_path: str,
    clone_tensors: bool = False,
    copy_tokenizer: bool = True,
):
    if config.embedding_source is None:
        config.embedding_source = config.layer_slices[0].model

    if config.lm_head_source is None:
        config.lm_head_source = config.layer_slices[-1].model

    referenced_models = list(
        set(
            [s.model for s in config.layer_slices]
            + [config.embedding_source, config.lm_head_source]
        )
    )

    configs: List[transformers.LlamaConfig] = [
        transformers.LlamaConfig.from_pretrained(m) for m in referenced_models
    ]

    (hidden_size, imm_size) = configs[0].hidden_size, configs[0].intermediate_size
    assert all(
        (cfg.hidden_size, cfg.intermediate_size) == (hidden_size, imm_size)
        for cfg in configs
    ), "Source models must all have same feature dimensions"

    loaders: Dict[str, LazyTensorLoader] = {}
    for model in referenced_models:
        ref = ModelReference(path=model)
        loaders[model] = LazyTensorLoader(ref.tensor_index())

    new_config = copy.deepcopy(configs[0])
    new_config.num_hidden_layers = sum([s.end - s.start for s in config.layer_slices])

    os.makedirs(out_path, exist_ok=True)
    new_config.save_pretrained(out_path)

    layer_sources = []
    for s in config.layer_slices:
        for source_idx in range(s.start, s.end):
            layer_sources.append((s.model, source_idx, s.scale))

    writer = TensorWriter(out_path)

    writer.save_tensor(
        "model.embed_tokens.weight",
        loaders[config.embedding_source].get_tensor("model.embed_tokens.weight"),
        clone=clone_tensors,
    )
    for layer_idx, (model_name, source_layer_idx, scale) in enumerate(
        tqdm(layer_sources)
    ):
        for tensor_name in LLAMA_LAYER_MEMBERS:
            source_key = f"model.layers.{source_layer_idx}.{tensor_name}.weight"

            weight = loaders[model_name].get_tensor(source_key)
            if "down_proj" in tensor_name and scale is not None:
                weight *= scale

            dst_key = f"model.layers.{layer_idx}.{tensor_name}.weight"
            writer.save_tensor(dst_key, weight, clone=clone_tensors)

    writer.save_tensor(
        "model.norm.weight",
        loaders[config.lm_head_source].get_tensor("model.norm.weight"),
        clone=clone_tensors,
    )
    writer.save_tensor(
        "lm_head.weight",
        loaders[config.lm_head_source].get_tensor("lm_head.weight"),
        clone=clone_tensors,
    )
    writer.finalize()

    if copy_tokenizer:
        transformers.AutoTokenizer.from_pretrained(layer_sources[0][0]).save_pretrained(
            out_path, safe_serialization=True
        )


def main(
    config_path: str,
    out_path: str,
    clone_tensors: Annotated[
        bool,
        typer.Option(
            help="Clone tensors before saving, to allow multiple occurrences of the same layer"
        ),
    ] = False,
    copy_tokenizer: bool = True,
):
    with open(config_path, "r", encoding="utf-8") as file:
        config = BakllamaConfig(**yaml.safe_load(file))

    process(
        config, out_path, clone_tensors=clone_tensors, copy_tokenizer=copy_tokenizer
    )


if __name__ == "__main__":
    typer.run(main)
