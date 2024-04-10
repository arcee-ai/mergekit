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

from enum import Enum

import torch


class SparsificationMethod(str, Enum):
    magnitude = "magnitude"
    random = "random"
    sample = "sample"
    ranked = "ranked"


def rescale_sum(tensor: torch.Tensor, mask: torch.Tensor):
    """Rescales the values to match the original tensor sum."""
    org_sum = tensor.abs().sum()
    new_sum = (tensor * mask).abs().sum()

    if org_sum >= 1e-8:
        tensor *= org_sum / new_sum
    else:
        pass

    return tensor * mask


def magnitude(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.view(-1).shape[0])

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.argsort(w, descending=True)[:k]
    mask.view(-1)[topk] = 1

    if rescale:
        res = rescale_sum(tensor, mask)
    else:
        res = tensor * mask

    return res


def bernoulli(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    if density >= 1:
        return tensor

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32

    mask = torch.bernoulli(
        torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
    )
    res = tensor.to(work_dtype) * mask
    if rescale:
        res /= density
    return res.to(tensor.dtype)


def ranked(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    if density >= 1:
        return tensor
    
    # Handle if the tensor is already sparser than the density (In line with trimming).
    if ((tensor.abs() ** 0.0).mean() / (tensor.abs() ** 0.0).max()) <= density:
        return tensor

    work_dtype = tensor.dtype
    size = int(tensor.view(-1).shape[0])

    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    sort = torch.argsort(w, descending=True)
    
    mask.view(-1)[sort] = torch.linspace(1, 0, steps=size, device=w.device.type, dtype=work_dtype).pow((1 / density) - 1)
    mask = torch.bernoulli(mask)
    
    if rescale:
        res = rescale_sum(tensor, mask)
    else:
        res = tensor * mask

    return res


def sample(tensor: torch.Tensor, density: float, rescale: bool) -> torch.Tensor:
    """Samples the tensor as it's own mask, then shifts mean to fit density."""
    if density >= 1 or tensor.abs().max() == 0.0 or tensor.abs().max() == float("inf"):
        return tensor

    # Handle if the tensor is already sparser than the density (In line with trimming).
    if ((tensor.abs() ** 0.0).mean() / (tensor.abs() ** 0.0).max()) <= density:
        return tensor

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32

    # Find the power that makes the distribution fit the density
    i = 0
    power = 1.0
    avg = tensor.abs().mean() / tensor.abs().max()
    while (avg - density) <= 1e-5 and i < 15:
        intermediate = tensor.abs() ** power
        avg = intermediate.mean() / intermediate.max()
        power += avg - density
        if power < 0:
            power = 0
        i += 1

    intermediate = tensor.abs() ** power
    mask = torch.bernoulli((intermediate / intermediate.max()).to(work_dtype))

    if rescale:
        res = rescale_sum(tensor, mask)
    else:
        res = tensor * mask
    return res.to(tensor.dtype)


def sparsify(
    tensor: torch.Tensor,
    density: float,
    method: SparsificationMethod,
    rescale: bool = False,
) -> torch.Tensor:
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density, rescale=rescale)
    elif method == SparsificationMethod.random:
        return bernoulli(tensor, density=density, rescale=rescale)
    elif method == SparsificationMethod.sample:
        return sample(tensor, density=density, rescale=rescale)
    elif method == SparsificationMethod.ranked:
        return ranked(tensor, density=density, rescale=rescale)
    else:
        raise NotImplementedError(method)
