# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import Dict, Optional, Tuple

import torch
import tqdm
import transformers

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io import LazyTensorLoader, TensorWriter
from mergekit.merge import MergeOptions
from mergekit.moe.config import Expert, MoEMergeConfig


def initialize_io(
    config: MoEMergeConfig,
    out_path: str,
    merge_options: MergeOptions,
) -> Tuple[Dict[ModelReference, LazyTensorLoader], LazyTensorLoader, TensorWriter]:
    base_model = config.base_model
    loaders: Dict[ModelReference, LazyTensorLoader] = {}
    for model in tqdm.tqdm(
        [base_model] + [e.source_model for e in config.experts], desc="Warm up loaders"
    ):
        loaders[model] = model.lazy_loader(
            cache_dir=merge_options.transformers_cache,
            lazy_unpickle=merge_options.lazy_unpickle,
        )

    base_loader = loaders.get(base_model)
    writer = TensorWriter(
        out_path=out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )

    return loaders, base_loader, writer


def select_dtype(
    config: MoEMergeConfig, base_cfg: transformers.PretrainedConfig
) -> Optional[torch.dtype]:
    out_dtype = None
    if config.dtype:
        out_dtype = dtype_from_name(config.dtype)

    if out_dtype is None and base_cfg.torch_dtype:
        out_dtype = base_cfg.torch_dtype
        if isinstance(out_dtype, str):
            out_dtype = dtype_from_name(out_dtype)
    return out_dtype


def noise_and_scale(
    tensor: torch.Tensor, expert: Expert, is_residual: bool = False
) -> torch.Tensor:
    if expert.noise_scale is not None:
        noise = torch.randn_like(tensor) * expert.noise_scale
        tensor = tensor + noise
    if is_residual and expert.residual_scale is not None:
        tensor = tensor * expert.residual_scale
    return tensor


def copy_tensor_out(
    weight_info: WeightInfo,
    loader: LazyTensorLoader,
    writer: TensorWriter,
    expert: Optional[Expert] = None,
    is_residual: bool = False,
    output_name: Optional[str] = None,
    out_dtype: Optional[torch.dtype] = None,
    clone: bool = False,
):
    out_tensor_name = output_name or weight_info.name
    aliases = weight_info.aliases or []
    if not weight_info.optional:
        aliases += weight_info.tied_names or []
    try:
        tensor = loader.get_tensor(
            weight_info.name,
            aliases=aliases,
        )
    except KeyError:
        tensor = None
    if tensor is None:
        if weight_info.optional:
            return
        logging.error(f"Missing weight: {weight_info.name} / {out_tensor_name}")
        raise KeyError(out_tensor_name)

    if expert:
        tensor = noise_and_scale(tensor, expert, is_residual=is_residual)
    writer.save_tensor(
        out_tensor_name,
        tensor.to(dtype=out_dtype),
        clone=clone,
    )
