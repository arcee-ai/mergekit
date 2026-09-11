"""Measure merge compute with already-loaded inputs, excluding loading and saving.

Run with: python benchmarks/merge_methods.py [--device cuda] [--threads 1]
The graph adapter still merges one output at a time. Compare its complete execute()
with a direct kernel call to expose adapter costs. The legacy linear reference
reproduces main's low-precision numerical path, not its graph overhead.
"""

import argparse

import torch
from torch.utils.benchmark import Timer

from mergekit import merge_methods
from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods import TensorBatch
from mergekit.merge_methods.task_adapter import ExecuteMergeMethodTask


def legacy_linear(tensors):
    stacked = torch.stack(list(tensors.values()))
    weight = torch.tensor([0.25, 0.75], dtype=stacked.dtype, device=stacked.device)
    weight = weight.reshape(2, *((1,) * (stacked.ndim - 1)))
    return (stacked * weight).sum(0) / weight.sum(0)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    refs = tuple(ModelReference.model_validate(name) for name in ("base", "other"))
    info = WeightInfo(name="benchmark.weight")
    print(f"Torch {torch.__version__}; device={device}; threads={args.threads}")
    print("method,dtype,elements,path,median_ms,peak_extra_cuda_MiB")
    for dtype in (torch.bfloat16, torch.float32):
        for shape in ((32,), (2048, 2048)):
            tensors = {
                ref: torch.randn(shape, device=device, dtype=dtype) for ref in refs
            }
            for name in ("linear", "slerp"):
                method = merge_methods.get(name)
                task = ExecuteMergeMethodTask.from_parameters(
                    method_name=name,
                    gather_tensors=GatherTensors(
                        weight_info=ImmutableMap({r: info for r in refs})
                    ),
                    model_order=refs,
                    base_model=refs[0],
                    output_weight=info,
                    parameters=ImmutableMap(
                        {"normalize": True} if name == "linear" else {"t": 0.75}
                    ),
                    input_parameters=ImmutableMap(
                        {
                            r: ImmutableMap({"weight": w})
                            for r, w in zip(refs, (0.25, 0.75))
                        }
                    ),
                )
                batch = TensorBatch(
                    tuple(t.unsqueeze(0) for t in tensors.values()), base_index=0
                )
                coefficients = (
                    {
                        "weight": torch.tensor(
                            [[0.25, 0.75]], device=device, dtype=torch.float64
                        )
                    }
                    if name == "linear"
                    else {"t": torch.tensor([0.75], device=device, dtype=torch.float64)}
                )
                paths = {
                    "graph_adapter": lambda: task.execute(tensors),
                    "direct_kernel": lambda: method.merge_batch(batch, **coefficients),
                }
                if name == "linear":
                    paths["legacy_numerics"] = lambda: legacy_linear(tensors)
                for label, fn in paths.items():
                    fn()
                    measurement = Timer(
                        stmt="fn()", globals={"fn": fn}, num_threads=args.threads
                    ).blocked_autorange(min_run_time=0.2)
                    peak = "n/a"
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        before = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        result = fn()
                        torch.cuda.synchronize(device)
                        peak = f"{(torch.cuda.max_memory_allocated(device) - before) / 2**20:.2f}"
                        del result
                    print(
                        f"{name},{dtype},{next(iter(tensors.values())).numel()},{label},{measurement.median * 1000:.4f},{peak}"
                    )


if __name__ == "__main__":
    main()
