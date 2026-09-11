# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Dict, Tuple

from mergekit.merge_methods.arcee_fusion import arcee_fusion_merge_method
from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.generalized_task_arithmetic import (
    ConsensusMethod,
    GeneralizedTaskArithmeticMerge,
)
from mergekit.merge_methods.karcher import karcher_merge_method
from mergekit.merge_methods.linear import linear_merge
from mergekit.merge_methods.model_stock import model_stock_merge_method
from mergekit.merge_methods.multislerp import multislerp
from mergekit.merge_methods.nearswap import nearswap_merge
from mergekit.merge_methods.nuslerp import nuslerp_merge_method
from mergekit.merge_methods.passthrough import passthrough_merge_method
from mergekit.merge_methods.ram import ram_merge, ramplus_tl_merge
from mergekit.merge_methods.sce import sce_merge
from mergekit.merge_methods.slerp import slerp_merge_method
from mergekit.sparsify import SparsificationMethod

_METHODS: Dict[str, MergeMethod] = {}


def register(method: MergeMethod) -> None:
    """Make a constructed method available by name; duplicate names are errors."""
    if method.spec.name in _METHODS:
        raise ValueError(f"Merge method {method.spec.name!r} is already registered")
    _METHODS[method.spec.name] = method


def get(name: str) -> MergeMethod:
    try:
        return _METHODS[name]
    except KeyError:
        raise RuntimeError(f"Unimplemented merge method {name}") from None


def registered_methods() -> Tuple[MergeMethod, ...]:
    """Return the methods currently available by name."""
    return tuple(_METHODS.values())


for method in (
    multislerp,
    nearswap_merge,
    ram_merge,
    ramplus_tl_merge,
    sce_merge,
    linear_merge,
    slerp_merge_method,
    nuslerp_merge_method,
    passthrough_merge_method,
    model_stock_merge_method,
    arcee_fusion_merge_method,
    karcher_merge_method,
    # generalized task arithmetic methods
    GeneralizedTaskArithmeticMerge(
        consensus_method=None,
        sparsification_method=None,
        default_normalize=False,
        default_rescale=False,
        method_name="task_arithmetic",
        method_pretty_name="Task Arithmetic",
        method_reference_url="https://arxiv.org/abs/2212.04089",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=ConsensusMethod.sum,
        sparsification_method=SparsificationMethod.magnitude,
        default_normalize=True,
        default_rescale=False,
        method_name="ties",
        method_pretty_name="TIES",
        method_reference_url="https://arxiv.org/abs/2306.01708",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=ConsensusMethod.sum,
        sparsification_method=SparsificationMethod.random,
        default_normalize=False,
        default_rescale=True,
        method_name="dare_ties",
        method_pretty_name="DARE TIES",
        method_reference_url="https://arxiv.org/abs/2311.03099",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=None,
        sparsification_method=SparsificationMethod.random,
        default_normalize=False,
        default_rescale=True,
        method_name="dare_linear",
        method_pretty_name="Linear DARE",
        method_reference_url="https://arxiv.org/abs/2311.03099",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=None,
        sparsification_method=SparsificationMethod.magnitude_outliers,
        default_normalize=False,
        default_rescale=False,
        method_name="breadcrumbs",
        method_pretty_name="Model Breadcrumbs",
        method_reference_url="https://arxiv.org/abs/2312.06795",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=ConsensusMethod.sum,
        sparsification_method=SparsificationMethod.magnitude_outliers,
        default_normalize=False,
        default_rescale=False,
        method_name="breadcrumbs_ties",
        method_pretty_name="Model Breadcrumbs with TIES",
        method_reference_url="https://arxiv.org/abs/2312.06795",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=ConsensusMethod.sum,
        sparsification_method=SparsificationMethod.della_magprune,
        default_normalize=True,
        default_rescale=True,
        method_name="della",
        method_pretty_name="DELLA",
        method_reference_url="https://arxiv.org/abs/2406.11617",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=None,
        sparsification_method=SparsificationMethod.della_magprune,
        default_normalize=False,
        default_rescale=True,
        method_name="della_linear",
        method_pretty_name="Linear DELLA",
        method_reference_url="https://arxiv.org/abs/2406.11617",
    ),
):
    register(method)
del method
