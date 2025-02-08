# mergekit-multi: Multi-Stage Model Merging

## What is mergekit-multi?

`mergekit-multi` is a command-line tool for executing complex model merging workflows with multiple interdependent stages. It allows you to:

1. Chain multiple merge operations together
2. Use outputs from previous merges as inputs to subsequent ones
3. Automatically handle dependencies between merge steps
4. Cache intermediate results for faster re-runs

## Usage

Basic command structure:
```bash
mergekit-multi <config.yaml> \
  --intermediate-dir ./intermediates \
  ([--out-path ./final-merge] | if config has unnamed merge) \
  [options]
```

## Configuration File Format

Create a YAML file with multiple merge configurations separated by `---`. Each should contain:

- `name`: Unique identifier for intermediate merges (except final merge)
- Standard mergekit configuration parameters

Example with Final Merge (`multimerge.yaml`):
```yaml
name: first-merge
merge_method: linear
models:
  - model: mistralai/Mistral-7B-v0.1
  - model: BioMistral/BioMistral-7B
parameters:
  weight: 0.5
---
name: second-merge
merge_method: slerp
base_model: first-merge  # Reference previous merge
models:
  - model: NousResearch/Hermes-2-Pro-Mistral-7B
parameters:
  t: 0.5
---
# Final merge (no name)
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
models:
  - model: second-merge
    parameters:
      density: 0.6
      weight: 0.5
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      density: 0.8
      weight: 0.5
```

### Example with All Named Merges:
```yaml
name: first-merge
merge_method: task_arithmetic
...
---
name: second-merge
merge_method: slerp
...
---
name: third-merge
merge_method: linear
...
```

## Key Options

- `--intermediate-dir`: Directory to store partial merge results
- `--out-path`: Output path for final merge (only applies when one merge has no `name`)
- `--lazy/--no-lazy`: Don't rerun existing intermediate merges (default: true)
- Standard mergekit options apply (e.g., `--cuda`, `--out-shard-size`, `--multi-gpu`)

## How It Works

When you run `mergekit-multi`, it topologically sorts your merge configurations to determine the correct order of execution. The merges are then processed sequentially, using outputs from previous steps as inputs for subsequent ones as needed.

All intermediate merges are saved in your specified `--intermediate-dir` using their configured names. By default, the tool will skip any merge operations that already have existing output files. To force re-execution of all merges, use the `--no-lazy` flag.
