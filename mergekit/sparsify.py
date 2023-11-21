from enum import Enum

import torch


class SparsificationMethod(str, Enum):
    magnitude = "magnitude"
    random = "random"
    rescaled_random = "rescaled_random"


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

    mask = torch.bernoulli(torch.full_like(input=tensor, fill_value=density))
    res = tensor * mask
    if rescale:
        res /= density
    return res


def sparsify(
    tensor: torch.Tensor, density: float, method: SparsificationMethod
) -> torch.Tensor:
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density)
    elif method == SparsificationMethod.random:
        return bernoulli(tensor, density=density, rescale=False)
    elif method == SparsificationMethod.rescaled_random:
        return bernoulli(tensor, density=density, rescale=True)
    else:
        raise NotImplementedError(method)
