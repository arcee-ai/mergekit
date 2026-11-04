# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
"""Time magnitude sparsification on CPU against the previous sort-based selection.

Usage: python benchmarks/bench_sparsify.py [--runs 5] [--density 0.5]

Shapes are the per-layer weight matrices of a Llama-3-8B-sized model. The
"sort" column re-implements the selection that sparsify.magnitude used before
switching to kthvalue, so both columns run on the same torch build.
"""

import argparse
import statistics
import time

import torch

from mergekit.sparsify import SparsificationMethod, sparsify

SHAPES = {
    "q_proj 4096x4096": (4096, 4096),
    "k_proj 1024x4096": (1024, 4096),
    "mlp 14336x4096": (14336, 4096),
}


def sort_magnitude(tensor: torch.Tensor, density: float) -> torch.Tensor:
    k = max(1, int(density * tensor.numel()))
    mask = torch.zeros(tensor.numel(), dtype=tensor.dtype)
    w = tensor.abs().reshape(-1).float()
    mask[torch.argsort(w, descending=True)[:k]] = 1
    return tensor * mask.reshape_as(tensor)


def median_seconds(fn, runs: int) -> float:
    fn()  # warm up
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--density", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    print(f"{'tensor':<20}{'dtype':<10}{'sort (s)':>10}{'new (s)':>10}{'speedup':>9}")
    for name, shape in SHAPES.items():
        for dtype in (torch.bfloat16, torch.float32):
            # A fine-tuning delta: difference of two checkpoints stored in dtype.
            base = (torch.randn(shape) * 0.02).to(dtype)
            tuned = (base.float() + torch.randn(shape) * 1e-3).to(dtype)
            delta = tuned - base
            old = median_seconds(lambda: sort_magnitude(delta, args.density), args.runs)
            new = median_seconds(
                lambda: sparsify(delta, args.density, SparsificationMethod.magnitude),
                args.runs,
            )
            kept = sparsify(delta, args.density, SparsificationMethod.magnitude)
            assert int((kept != 0).sum()) <= max(1, int(args.density * delta.numel()))
            dt = str(dtype).replace("torch.", "")
            print(f"{name:<20}{dt:<10}{old:>10.3f}{new:>10.3f}{old / new:>8.1f}x")


if __name__ == "__main__":
    main()
