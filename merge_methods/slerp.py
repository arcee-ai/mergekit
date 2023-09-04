from typing import Dict, Union

import numpy as np
import torch
from pydantic import BaseModel
from torch import lerp

from common import ModelReference, rectify_embed_sizes


class SlerpMergeOptions(BaseModel):
    base_model: ModelReference
    t: float


def slerp_merge_tensors(
    options: Union[SlerpMergeOptions, Dict],
    param_name: str,
    tensors: Dict[ModelReference, torch.Tensor],
) -> torch.Tensor:
    if isinstance(options, Dict):
        options = SlerpMergeOptions(**options)

    if len(tensors) == 1:
        return list(tensors.values())[0]
    elif len(tensors) != 2:
        raise RuntimeError("Slerp merge expects exactly two models")

    [a, b] = list(tensors.items())
    if a[0] != options.base_model:
        [a, b] = [b, a]

    rectify_embed_sizes(param_name, [a, b])

    return slerp(options.t, a[1], b[1])


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """
    Spherical linear interpolation

    From: https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    """
    c = False
    if not isinstance(v0, np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1, np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    if c:
        res = torch.from_numpy(v2).to("cuda")
    else:
        res = v2
    return res
