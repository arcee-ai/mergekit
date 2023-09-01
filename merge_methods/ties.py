import logging
from typing import Dict, Optional, Union

import torch
from pydantic import BaseModel
from typing_extensions import Literal

from common import ModelReference, dtype_from_name


class TiesMergeOptions(BaseModel):
    base_model: ModelReference
    density: Union[float, Dict[ModelReference, float]] = 0.33
    weight: Optional[Dict[ModelReference, float]] = None
    int8_mask: bool = False
    normalize: bool = True
    dtype: Literal[None, "bfloat16", "float16", "float32"] = None
    consensus_method: Literal["sum", "count"] = "sum"


def ties_merge_tensors(
    options: Union[TiesMergeOptions, Dict],
    param_name: str,
    tensors: Dict[ModelReference, torch.Tensor],
) -> torch.Tensor:
    if isinstance(options, Dict):
        options = TiesMergeOptions(**options)

    # expand density and weight parameters to a dict
    if isinstance(options.density, float):
        density = {model: options.density for model in tensors}
    else:
        density = options.density

    if options.weight is None:
        weight = {model: 1 for model in tensors}
    else:
        weight = options.weight

    base = tensors[options.base_model]
    # resolve dtype for mask and result
    mask_dtype = torch.int8 if options.int8_mask else base.dtype
    if options.dtype is None:
        ty = base.dtype
    else:
        ty = dtype_from_name(options.dtype)
        base = base.to(ty)

    deltas = []
    weights = []
    model_names = list(tensors.keys())
    for model_name in model_names:
        if model_name == options.base_model:
            continue

        x = tensors[model_name].to(ty)
        if x.shape != base.shape:
            if "lm_head" in param_name or "embed_tokens" in param_name:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model_name}:{param_name}")
            else:
                logging.warning(
                    f"skipping {model_name}:{param_name} due to size mismatch"
                )
                continue

        if (x == base).view(-1).all():
            continue

        deltas.append(sparsify(x - base, density[model_name]))
        weights.append(weight[model_name])

        del tensors[model_name]
        del x

    if deltas:
        deltas = torch.stack(deltas, dim=0)
        weights = torch.tensor(weights, dtype=ty, device=deltas.device)
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = weights * deltas

        mask = get_mask(
            weighted_deltas,
            method=options.consensus_method,
            mask_dtype=mask_dtype,
        )

        mixed_delta = (weighted_deltas * mask).sum(dim=0)

        if options.normalize:
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
            mixed_delta /= divisor

        res = base + mixed_delta
    else:
        res = base

    return res.to(ty)


def sparsify(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.view(-1).shape[0])

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.topk(w, k=k, largest=True)
    mask.view(-1)[topk.indices] = 1

    return tensor * mask


def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = (sign * delta.abs()).sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign
