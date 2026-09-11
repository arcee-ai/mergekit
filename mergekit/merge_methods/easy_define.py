# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Decorator-based merge method definitions."""

from typing import Callable, Optional

import torch

from mergekit.merge_methods.base import (
    FunctionalMergeMethod,
    InputContract,
    method_from_function,
)


def merge_method(
    name: str,
    reference_url: Optional[str] = None,
    pretty_name: Optional[str] = None,
    contract: Optional[InputContract] = None,
) -> Callable[[Callable[..., torch.Tensor]], FunctionalMergeMethod]:
    """Define and register a consumer-neutral merge method.

    The decorated function receives one ``TensorGroup`` and parameters annotated
    with ``Shared[T]`` or ``PerInput[T]``. The returned callable method accepts a
    ``MergeBatch`` and returns a ``MergedBatch``.
    """

    def _wrap(func: Callable[..., torch.Tensor]) -> FunctionalMergeMethod:
        # Import lazily so registry construction can import method modules.
        from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS

        method = method_from_function(
            func,
            name=name,
            pretty_name=pretty_name,
            reference_url=reference_url,
            contract=contract,
        )
        if name in REGISTERED_MERGE_METHODS:
            raise ValueError(f"Merge method {name!r} is already registered")
        REGISTERED_MERGE_METHODS[name] = method
        return method

    return _wrap


__all__ = ["merge_method"]
