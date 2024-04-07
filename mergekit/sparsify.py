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
    rescaled_random = "rescaled_random"
    sample = "sample"

def magnitude(tensor: torch.Tensor, density: float) -> torch.Tensor:
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


def bernoulli(
    tensor: torch.Tensor, density: float, rescale: bool = True
) -> torch.Tensor:
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

def sample(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Samples the tensor as it's own mask, then shifts mean to fit density."""
    if density >= 1 or tensor.abs().max() == 0.0:
        return tensor

    if (tensor.device.type == "cpu") or tensor.dtype != torch.bfloat16:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        origin_type = tensor.dtype
        tensor = tensor.to(torch.float32)

    intermediate = tensor.abs()
    avg = intermediate.mean() / intermediate.max()

    # Handle if the tensor is already sparser than the density (In line with trimming).
    if ((intermediate**0.0).mean() / (intermediate**0.0).max()) <= density:
        if (tensor.device.type == "cpu") or tensor.dtype != torch.bfloat16:
            return tensor.to(origin_type)
        return tensor

    # Find the power that makes the distribution fit the density
    i = 0; power = 1.0
    while i < 15:
        # print("Average: ", avg)
        # print("Density: ", density)
        # print("Diff: ", avg - density)
        power += avg - density
        # print("Power: ", power)
        if power < 0:
            power = 0
        intermediate = tensor.abs()**power
        # print("Intermediate: ", intermediat)
        avg = intermediate.mean() / intermediate.max()
        i += 1

    mask = torch.bernoulli(intermediate / intermediate.max())
    
    tensor *= mask
    if (tensor.device.type == "cpu") or tensor.dtype != torch.bfloat16:
        return tensor.to(origin_type)
    return tensor

def sparsify(
    tensor: torch.Tensor, density: float, method: SparsificationMethod
) -> torch.Tensor:
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density)
    elif method == SparsificationMethod.random:
        return bernoulli(tensor, density=density, rescale=False)
    elif method == SparsificationMethod.rescaled_random:
        return bernoulli(tensor, density=density, rescale=True)
    elif method == SparsificationMethod.sample:
        return sample(tensor, density=density)
    else:
        raise NotImplementedError(method)
