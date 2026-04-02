# LRP Merge

**Layer-wise Relevance Propagation (LRP) Merge** is a model merging method that uses relevance scores to determine which weights are most important during the merge process.

## Overview

LRP was originally developed for interpreting neural network predictions by propagating relevance scores backward through the network layers. LRP Merge adapts this technique for model merging by using LRP-computed importance scores to guide which weights should be preserved when combining fine-tuned models.

## Algorithm

LRP Merge follows these steps:

1. **Compute Task Vectors**: For each fine-tuned model, compute the delta from the base model: `delta = fine_tuned - base`

2. **Get Importance Scores**: Retrieve pre-computed LRP importance scores for each model. These scores indicate which weights contributed most to the model's predictions during LRP analysis.

3. **Sparsify Based on Importance**: Using the `density` parameter, select the top-k most important weights:
   - `k = density * numel(weights)`
   - Create a binary mask: `mask = topk(importance, k)`
   - Apply mask: `sparse_delta = delta * mask`

4. **Weighted Averaging**: Combine sparse deltas from all models using weights:
   - `merged_deltas = sum_i(weight_i * sparse_delta_i) / sum(weights)`

5. **Add Back to Base**: `result = base + merged_deltas`

## Merge Configuration

```yaml
merge_method: lrp
base_model: path/to/base_model

models:
  - model: path/to/model_a
    parameters:
      weight: 0.5
  - model: path/to/model_b
    parameters:
      weight: 0.5

parameters:
  density: 0.7  # Fraction of weights to retain (0.0 to 1.0)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `density` | float | 0.7 | Fraction of weights to retain based on importance scores |
| `weight` | float | 1.0 | Per-model weight for weighted averaging |

## Computing LRP Scores

Before using LRP Merge with importance scores, you need to compute LRP scores for each fine-tuned model:

```bash
python -m mergekit.lrp_computer \
    --model path/to/model \
    --output path/to/lrp_scores \
    --rule epsilon
```

### LRP Rules

| Rule | Description |
|------|-------------|
| `epsilon` | Epsilon rule: `R_j = sum_k (z_jk / (sum_j z_jk + epsilon)) * R_k` |
| `gamma` | Gamma rule: Enhances positive contributions |
| `alpha_beta` | Alpha-beta rule: Separates positive and negative contributions |

### Configuration Options

```bash
--rule epsilon|gamma|alpha_beta  # LRP rule to use (default: epsilon)
--device cuda|cpu               # Device for computation (default: cuda if available)
--prompts "prompt1" "prompt2"  # Sample prompts for LRP computation
```

The LRP computer will save:
- `lrp_scores.pt` - Tensor with relevance scores for each parameter
- `lrp_metadata.json` - Metadata about the computation

## How LRP Scores Work

LRP works by propagating relevance scores backward through the neural network. At each layer, the relevance is distributed to neurons based on their contribution to the layer's output:

1. **Forward Pass**: Input is propagated through the network to get predictions
2. **Relevance Propagation**: Starting from the output, relevance is distributed backward using the LRP rule
3. **Parameter Importance**: Each parameter's importance is the sum of relevance scores flowing through it

Parameters with higher LRP scores are considered more important for the model's behavior and should be preserved during merging.

## When to Use LRP Merge

LRP Merge is particularly useful when:

- You have fine-tuned models and want to understand which parts of each model are most important
- You want a principled, gradient-based approach to determining importance rather than pure magnitude
- Merging models trained on different tasks where task-specific weights can be identified via LRP

## Example Use Case

```yaml
# Merge two models fine-tuned on different aspects of fake news detection
merge_method: lrp
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

models:
  - model: models/global-fake-news-detector
    parameters:
      weight: 0.6
  - model: models/local-fake-news-detector
    parameters:
      weight: 0.4

parameters:
  density: 0.7
```

## Comparison with Other Methods

| Method | Importance Metric | Sparsity |
|--------|------------------|----------|
| LRP Merge | LRP relevance scores | Top-k by relevance |
| Task Arithmetic | Direct delta | None |
| TIES | Magnitude | Top-k by magnitude |
| DELLA | Magnitude (row-adaptive) | Bernoulli sampling |
| Arcee Fusion | KL divergence + delta | Dynamic threshold |

## Reference

[MergeKit](https://github.com/arcee-ai/mergekit)
