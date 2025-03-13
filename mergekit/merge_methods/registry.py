# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Dict, List

from mergekit.merge_methods.arcee_fusion import ArceeFusionMerge
from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.generalized_task_arithmetic import (
    ConsensusMethod,
    GeneralizedTaskArithmeticMerge,
)
from mergekit.merge_methods.linear import LinearMerge
from mergekit.merge_methods.model_stock import ModelStockMerge
from mergekit.merge_methods.nuslerp import NuSlerpMerge
from mergekit.merge_methods.passthrough import PassthroughMerge
from mergekit.merge_methods.slerp import SlerpMerge
from mergekit.sparsify import SparsificationMethod

STATIC_MERGE_METHODS: List[MergeMethod] = [
    LinearMerge(),
    SlerpMerge(),
    NuSlerpMerge(),
    PassthroughMerge(),
    ModelStockMerge(),
    ArceeFusionMerge(),
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
]

REGISTERED_MERGE_METHODS: Dict[str, MergeMethod] = {
    method.name(): method for method in STATIC_MERGE_METHODS
}
