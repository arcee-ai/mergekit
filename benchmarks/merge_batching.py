"""Compare singleton graph execution with multi-output state-dict merging.

Inputs and graph tasks are prepared before measurement. Both paths retain all
outputs and include tensor validation and any packing, but exclude loading,
scheduling, tokenizer alignment, and saving. Graph parameters are already bound
during task construction; the state-dict path binds them on each call. CUDA peak allocation includes
outputs and scratch, not already-loaded inputs or allocator-reserved memory.

Examples (run from the repository root in the project environment):
    python benchmarks/merge_batching.py --shape 32 --groups 1 256
    python benchmarks/merge_batching.py --shape 2048 2048 --groups 1 4
    python benchmarks/merge_batching.py --device cuda --shape 2048 2048 --groups 1 4
"""

import argparse
import math

import torch
from torch.utils.benchmark import Timer

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods import BatchOptions, merge_state_dicts
from mergekit.merge_methods.task_adapter import ExecuteMergeMethodTask


def make_paths(models, method_name, batch_options):
    refs = tuple(models)
    parameters = {"weight": [0.25, 0.75]} if method_name == "linear" else {"t": 0.75}
    tasks = []
    for name in models[refs[0]]:
        info = WeightInfo(name=name)
        task = ExecuteMergeMethodTask.from_parameters(
            method_name=method_name,
            gather_tensors=GatherTensors(
                weight_info=ImmutableMap({ref: info for ref in refs})
            ),
            model_order=refs,
            base_model=refs[0],
            output_weight=info,
            parameters=ImmutableMap(
                {"normalize": True} if method_name == "linear" else parameters
            ),
            input_parameters=ImmutableMap(
                {
                    ref: ImmutableMap(
                        {"weight": weight} if method_name == "linear" else {}
                    )
                    for ref, weight in zip(refs, (0.25, 0.75))
                }
            ),
        )
        tasks.append((name, task, {ref: state[name] for ref, state in models.items()}))

    def graph_singletons():
        return {name: task.execute(tensors) for name, task, tensors in tasks}

    def state_dict_batch():
        return merge_state_dicts(
            models,
            method_name,
            base=refs[0],
            parameters=parameters,
            batch_options=batch_options,
        )

    return {"graph_singletons": graph_singletons, "state_dict_batch": state_dict_batch}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--shape", type=int, nargs="+", default=[32])
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 256])
    parser.add_argument("--max-packed-mib", type=int, default=64)
    args = parser.parse_args()
    if min([args.threads, args.max_packed_mib, *args.shape, *args.groups]) <= 0:
        parser.error(
            "Threads, dimensions, group counts, and packing limit must be positive"
        )
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    refs = tuple(ModelReference.model_validate(name) for name in ("base", "other"))
    options = BatchOptions(max_bytes=args.max_packed_mib * 1024**2)
    print(f"Torch {torch.__version__}; device={device}; threads={args.threads}")
    print("method,dtype,elements,groups,path,median_ms,peak_extra_cuda_MiB")
    for dtype in (torch.bfloat16, torch.float32):
        for count in args.groups:
            models = {
                ref: {
                    f"weight_{i}": torch.randn(args.shape, device=device, dtype=dtype)
                    for i in range(count)
                }
                for ref in refs
            }
            for name in ("linear", "slerp"):
                paths = make_paths(models, name, options)
                # Validate equivalent workloads before comparing their performance.
                expected = paths["graph_singletons"]()
                actual = paths["state_dict_batch"]()
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key])
                del expected, actual
                for label, fn in paths.items():
                    measurement = Timer(
                        stmt="fn()", globals={"fn": fn}, num_threads=args.threads
                    ).blocked_autorange(min_run_time=0.2)
                    peak = "n/a"
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        initial = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        outputs = fn()
                        torch.cuda.synchronize(device)
                        peak = f"{(torch.cuda.max_memory_allocated(device) - initial) / 1024**2:.2f}"
                        del outputs
                    print(
                        f"{name},{dtype},{math.prod(args.shape)},{count},{label},"
                        f"{measurement.median * 1000:.4f},{peak}",
                        flush=True,
                    )
                # Do not retain a previous workload's models through closures.
                del paths, fn
            del models


if __name__ == "__main__":
    main()
