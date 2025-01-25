# Copyright (C) 2025 Arcee AI
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

from typing import Dict, List

from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.generalized_task_arithmetic import (
    ConsensusMethod,
    GeneralizedTaskArithmeticMerge,
)
from mergekit.merge_methods.linear import LinearMerge
from mergekit.merge_methods.model_stock import ModelStockMerge
from mergekit.merge_methods.nearswap import NearSwapMerge
from mergekit.merge_methods.nuslerp import NuSlerpMerge
from mergekit.merge_methods.passthrough import PassthroughMerge
from mergekit.merge_methods.sce import SCEMerge
from mergekit.merge_methods.slerp import SlerpMerge
from mergekit.sparsify import SparsificationMethod

STATIC_MERGE_METHODS: List[MergeMethod] = [
    LinearMerge(),
    SlerpMerge(),
    NuSlerpMerge(),
    PassthroughMerge(),
    ModelStockMerge(),
    SCEMerge(),
    NearSwapMerge(),
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
        sparsification_method=SparsificationMethod.rank_magnitude_sampling,
        default_normalize=True,
        default_rescale=True,
        method_name="della",
        method_pretty_name="DELLA",
        method_reference_url="https://arxiv.org/abs/2406.11617",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=None,
        sparsification_method=SparsificationMethod.rank_magnitude_sampling,
        default_normalize=False,
        default_rescale=True,
        method_name="della_linear",
        method_pretty_name="Linear DELLA",
        method_reference_url="https://arxiv.org/abs/2406.11617",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=None,
        sparsification_method=SparsificationMethod.consensus_ta,
        default_normalize=False,
        default_rescale=False,
        method_name="consensus_ta",
        method_pretty_name="Consensus Task Arithmetic",
        method_reference_url="https://arxiv.org/abs/2405.07813",
    ),
    GeneralizedTaskArithmeticMerge(
        consensus_method=ConsensusMethod.sum,
        sparsification_method=SparsificationMethod.consensus_ties,
        default_normalize=True,
        default_rescale=False,
        method_name="consensus_ties",
        method_pretty_name="Consensus TIES",
        method_reference_url="https://arxiv.org/abs/2405.07813",
    ),
]

REGISTERED_MERGE_METHODS: Dict[str, MergeMethod] = {
    method.name(): method for method in STATIC_MERGE_METHODS
}
