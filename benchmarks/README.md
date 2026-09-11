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
loading, tokenizer alignment, and writing are excluded. The prepacked comparison
also excludes packing and adapter overhead. The legacy linear comparison reproduces
main's numerical implementation, including packing, but excludes its graph adapter.

`merge_batching.py` compares repeated singleton graph calls with one
`merge_state_dicts` call. Both paths retain all outputs and include tensor validation
and packing. Graph parameters are bound before timing; the state-dict API binds
parameters on each invocation. The script checks that their outputs agree before
measuring. Use `--max-packed-mib` to vary the state-dict packing budget (64 MiB by
default).

CUDA peak allocation includes returned outputs and scratch, but excludes source
tensors and prepacked buffers. It does not measure allocator-reserved memory or
process-wide GPU memory. Run benchmarks without concurrent tests or GPU workloads.

## Precision and memory

Linear accumulates in one float32 output buffer (float64 for float64 inputs),
normalizes using a small coefficient sum reduced in float64, then casts the result
to the input dtype. It does not cast coefficients to float16/bfloat16. GPU kernels
consume the low-precision inputs directly; CPU execution may additionally cast the
current input. Scratch stays bounded independently of the input count.

For two 2048×2048 BF16 inputs, GPU packing needs 16 MiB, the accumulator needs 16 MiB,
and the returned output needs 8 MiB: 40 MiB extra at peak. For FP32 inputs, the output
is already the accumulator, giving a 48 MiB peak. Packing limits exclude kernel
scratch and returned outputs, so batching multiple outputs can increase peak memory.

SLERP retains its chunked, device-local implementation. The graph still executes
one output at a time; cross-output batching benefits the in-memory API.

## H100 observations

Measured on an NVIDIA H100 PCIe with Torch 2.14.0+cu126 and one CPU thread, using
weights 0.25/0.75. These are illustrative medians, not performance guarantees.
For one 2048×2048 output:

| Method | Dtype | Graph execute (ms) | Prepacked kernel (ms) | Graph peak extra (MiB) |
| --- | --- | ---: | ---: | ---: |
| Linear | BF16 | 0.241 | 0.121 | 40.00 |
| Linear | FP32 | 0.240 | 0.105 | 48.00 |
| SLERP | BF16 | 1.373 | 1.215 | 44.01 |
| SLERP | FP32 | 1.126 | 0.980 | 60.01 |

The legacy linear numerical path measured 0.107 ms / 40 MiB for BF16 and
0.156 ms / 80 MiB for FP32. Resolving parameters during planning removes repeated
binding from graph execution, but tensor checks, packing, and dispatch still cost
time. These measurements do not establish an end-to-end speedup over main.

For four BF16 2048×2048 outputs, linear took 0.980 ms / 64 MiB as graph singletons
and 0.823 ms / 120 MiB through the batched state-dict API. SLERP took
5.398 ms / 68 MiB and 4.713 ms / 92 MiB, respectively. Packing several outputs
increased memory while reducing execution time for these workloads.
