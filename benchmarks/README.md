# Merge compute benchmarks

Run from the repository root in the project environment:

```sh
python benchmarks/merge_methods.py --device cpu --threads 1
python benchmarks/merge_methods.py --device cuda
python benchmarks/merge_batching.py --device cuda --shape 32 --groups 1 256
python benchmarks/merge_batching.py --device cuda --shape 2048 2048 --groups 1 4
```

`merge_methods.py` measures `ExecuteMergeMethodTask.execute()` with already-loaded
inputs. Planning (including parameter resolution and binding), graph scheduling,
loading, tokenizer alignment, and writing are excluded. The direct-kernel comparison
uses borrowed singleton views and excludes adapter overhead. The legacy linear
comparison reproduces main's numerical implementation, including packing, but
excludes its graph adapter.

`merge_batching.py` compares repeated singleton graph calls with one
`merge_state_dicts` call. Both paths retain all outputs and include tensor validation
and any packing. Singleton inputs are borrowed; compatible outputs in larger
batches are stacked separately for each input. Graph parameters are bound before
timing; the state-dict API binds parameters on each invocation. The script checks that their outputs agree before
measuring. Use `--max-packed-mib` to vary the state-dict packing budget (64 MiB by
default).

CUDA peak allocation includes returned outputs and scratch, but excludes source
tensors. It does not measure allocator-reserved memory or process-wide GPU memory. Run benchmarks without concurrent tests or GPU workloads.

## Precision and memory

Linear uses float64 coefficients and one float64 accumulator, normalizes using
the float64 coefficient sum, then casts to the aligned input dtype. Normalization
rejects a zero coefficient sum. GPU kernels consume the inputs directly; CPU
execution may additionally cast the current input to float64. No full-precision
copy of every input is retained, so accumulator scratch is independent of input
count. Float64 arithmetic trades throughput for a simple, consistent precision
policy; CPU BF16 execution can be substantially slower than the legacy path.

For a singleton output from two 2048×2048 BF16 inputs, the accumulator needs 32 MiB
and the returned output needs 8 MiB: 40 MiB extra at peak on the GPU. FP32 output
needs 16 MiB, giving a 48 MiB peak. Both calls borrow their inputs instead of packing.
CPU TensorIterator may also allocate a 32 MiB cast of the current input, giving
64 MiB at peak for either dtype in this example.

SLERP retains its chunked, device-local implementation and borrows singleton inputs.
Kernels accept arbitrary strides and explicitly call `contiguous()` or `stack()`
when they need contiguous storage; all inputs must be treated as read-only.
The graph still executes one output at a time; cross-output batching benefits the
in-memory API. Packing limits exclude kernel scratch and returned outputs, so
batching multiple outputs can increase peak memory.

## H100 observations

Measured on an NVIDIA H100 PCIe with Torch 2.14.0+cu126 and one CPU thread, using
weights 0.25/0.75. These are illustrative medians, not performance guarantees.
For one 2048×2048 output:

| Method | Dtype | Graph execute (ms) | Direct kernel (ms) | Graph peak extra (MiB) |
| --- | --- | ---: | ---: | ---: |
| Linear | BF16 | 0.293 | 0.200 | 40.00 |
| Linear | FP32 | 0.272 | 0.197 | 48.00 |
| SLERP | BF16 | 1.619 | 1.065 | 28.01 |
| SLERP | FP32 | 0.921 | 0.907 | 28.01 |

The legacy linear numerical path measured 0.108 ms / 40 MiB for BF16 and
0.155 ms / 80 MiB for FP32. Resolving parameters during planning removes repeated
binding from graph execution, but tensor checks, coefficient preparation, and
dispatch still cost time. Normalized linear kernels also synchronize to reject
a zero coefficient sum. These measurements do not establish an end-to-end speedup over main.

For four BF16 2048×2048 outputs, linear took 1.001 ms / 64 MiB as graph singletons
and 1.024 ms / 168 MiB through the batched state-dict API. SLERP took
4.516 ms / 52 MiB and 4.164 ms / 92 MiB, respectively. Packing several large
outputs increased memory and did not improve linear throughput in this workload.
Use `BatchOptions(max_groups=1)` to execute state-dict weights as singletons when
minimizing scratch is more useful than batching outputs.
