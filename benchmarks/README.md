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

Linear uses float32 coefficients and one float32 accumulator for float16, BF16,
and FP32 inputs; float64 inputs use float64. It normalizes before casting back to
the aligned input dtype and rejects a zero coefficient sum in the working precision.
Nearly cancelling coefficients may lose accuracy. GPU kernels consume the inputs
directly; CPU execution may additionally cast the current input. No full-precision
copy of every input is retained, so accumulator scratch is independent of input count.

For a singleton output from two 2048×2048 BF16 inputs, the accumulator needs 16 MiB
and the returned output needs 8 MiB: 24 MiB extra at peak on the GPU. FP32 inputs
use the accumulator itself as the output, giving a 16 MiB peak. Both calls borrow
their inputs instead of packing. CPU TensorIterator may also allocate a 16 MiB
cast of the current BF16 input, giving a 32 MiB peak.

SLERP retains its chunked, device-local implementation and borrows singleton inputs.
Kernels accept arbitrary strides and explicitly call `contiguous()` or `stack()`
when they need contiguous storage; all inputs must be treated as read-only.
The graph still executes one output at a time; cross-output batching benefits the
in-memory API. Packing limits exclude kernel scratch and returned outputs, so
batching multiple outputs can increase peak memory.

## H100 observations

Measured on an NVIDIA H100 PCIe with Torch 2.14.0+cu126 and one CPU thread, using
weights 0.25/0.75 and the float32 working-precision policy. For one 2048×2048 output:

| Method | Dtype | Graph execute (ms) | Direct kernel (ms) | Graph peak extra (MiB) |
| --- | --- | ---: | ---: | ---: |
| Linear | BF16 | 0.217 | 0.125 | 24.00 |
| Linear | FP32 | 0.205 | 0.102 | 16.00 |
| SLERP | BF16 | 1.081 | 1.008 | 28.01 |
| SLERP | FP32 | 0.894 | 0.769 | 28.01 |

The legacy linear calculation measured 0.103 ms / 40 MiB for BF16 and
0.155 ms / 80 MiB for FP32. Normalized linear kernels still synchronize to reject
a zero coefficient sum. These illustrative medians exclude loading and saving;
they do not establish an end-to-end speedup over main.

On CPU with one thread, the same linear direct-kernel benchmark measured 7.66 ms
versus 15.23 ms for the legacy BF16 calculation, and 4.43 ms versus 45.00 ms for
FP32. These measurements also exclude loading, saving, and graph overhead.
