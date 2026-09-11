# Merge compute benchmark

Run from the repository root in the project environment:

```sh
python benchmarks/merge_methods.py --device cpu --threads 1
python benchmarks/merge_methods.py --device cuda
```

This measures the actual `ExecuteMergeMethodTask.execute()` path with already-loaded
inputs. It excludes graph scheduling, loading, tokenizer alignment, and writing.
The prepacked comparison excludes input packing and parameter binding; it is not
an end-to-end alternative. The legacy linear comparison reproduces main's numerical
implementation, including packing, but excludes its graph-adapter overhead.

## Local CPU observations

Measured with Torch 2.5.1+cu124, one CPU thread, two 2048×2048 input tensors, and
weights 0.25/0.75. Times are illustrative medians, not performance guarantees.

| Linear path | bfloat16 (ms) | float32 (ms) |
| --- | ---: | ---: |
| Graph adapter, multiply-then-reduce kernel | 107.4 | 78.9 |
| Graph adapter, matrix-product kernel | 68.3 | 55.5 |
| Prepacked matrix-product kernel | 52.6 | 27.5 |
| Legacy low-precision numerics | 51.5 | 82.1 |

The matrix product removes the full weighted-input intermediate while retaining
float32 accumulation for low-precision inputs and float64 for float64 inputs.
Low-precision inputs still require a float32 conversion buffer, released before
normalization. With these shapes, packed inputs occupy 16 MiB in bfloat16 or 32 MiB
in float32; the bfloat16-to-float32 conversion adds another 32 MiB. Packing limits
do not include that conversion, outputs, or other kernel scratch space.

The graph adapter still submits **one output at a time**. Small tensors are dominated
by binding and packing overhead (roughly 0.17 ms for a 32-element linear merge here).
Cross-output batching benefits in-memory callers, not the current YAML scheduler.
Further scheduling changes should be measured separately from kernel changes.

CUDA was unavailable for these measurements. On CUDA the script also reports peak
additional allocated tensor memory, including the returned output but excluding
already-loaded inputs and any prepacked buffer. It does not report allocator-reserved
memory or process-wide GPU memory.

## Cross-output batching

`merge_batching.py` compares repeated singleton graph-adapter calls against one
`merge_state_dicts` call for the same inputs and coefficients:

```sh
python benchmarks/merge_batching.py --shape 32 --groups 1 256 --threads 1
python benchmarks/merge_batching.py --shape 2048 2048 --groups 1 4 --threads 1
python benchmarks/merge_batching.py --device cuda --shape 2048 2048 --groups 1 4
```

Both paths include validation, parameter binding, and packing, and retain all
outputs. Graph construction, scheduling, loading, tokenizer alignment, and writing
are excluded. The script checks that outputs agree before measuring. On CUDA it
reports peak additional allocated memory, including outputs and scratch but
excluding already-loaded inputs. Use `--max-packed-mib` to vary the state-dict
packing budget (default 64 MiB).

Local CPU measurements with Torch 2.5.1+cu124 and one thread:

| Method | Dtype | Outputs × shape | Singleton graph (ms) | State-dict batch (ms) |
| --- | --- | --- | ---: | ---: |
| Linear | bfloat16 | 256 × 32 | 50.14 | 14.86 |
| SLERP | bfloat16 | 256 × 32 | 113.26 | 9.27 |
| Linear | float32 | 256 × 32 | 46.52 | 13.74 |
| SLERP | float32 | 256 × 32 | 98.36 | 8.31 |
| Linear | bfloat16 | 4 × 2048×2048 | 294.15 | 324.98 |
| SLERP | bfloat16 | 4 × 2048×2048 | 224.23 | 287.34 |
| Linear | float32 | 4 × 2048×2048 | 243.31 | 250.30 |
| SLERP | float32 | 4 × 2048×2048 | 301.68 | 277.52 |

These are illustrative measurements, not performance guarantees. Batching helped
many small outputs substantially, but was not uniformly faster for large weights.
Singleton calls through the two APIs were of similar cost. CUDA was unavailable;
GPU throughput and peak-memory results must be measured on the target hardware.
