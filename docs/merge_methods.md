# Merge Method Guide

## Table of Contents

- [Overview](#overview)
- [Basic Merging Methods](#basic-merging-methods)
  - [Linear (`linear`)](#linear-linear)
- [Spherical Interpolation Methods](#spherical-interpolation-methods)
  - [SLERP (`slerp`)](#slerp-slerp)
  - [NuSLERP (`nuslerp`)](#nuslerp-nuslerp)
  - [Multi-SLERP (`multislerp`)](#multi-slerp-multislerp)
  - [Karcher Mean (`karcher`)](#karcher-mean-karcher)
- [Task Vector Methods](#task-vector-methods)
  - [Task Arithmetic (`task_arithmetic`)](#task-arithmetic-task_arithmetic)
  - [TIES-Merging (`ties`)](#ties-merging-ties)
  - [DARE (`dare_linear`, `dare_ties`)](#dare-dare_linear-dare_ties)
  - [DELLA (`della`, `della_linear`)](#della-della-della_linear)
  - [Model Breadcrumbs (`breadcrumbs`, `breadcrumbs_ties`)](#model-breadcrumbs-breadcrumbs-breadcrumbs_ties)
  - [SCE (`sce`)](#sce-sce)
- [Specialized Methods](#specialized-methods)
  - [Model Stock (`model_stock`)](#model-stock-model_stock)
  - [Nearswap (`nearswap`)](#nearswap-nearswap)
  - [Arcee Fusion (`arcee_fusion`)](#arcee-fusion-arcee_fusion)
  - [Passthrough (`passthrough`)](#passthrough-passthrough)
- [Summary](#summary)
- [Contributing](#contributing)

## Overview

This guide provides detailed information about the various model merging algorithms available in `mergekit`. Each method has specific use cases, parameters, and applications for combining machine learning models.

## Basic Merging Methods

### Linear (`linear`)

**Concept:** Computes a simple weighted average of the parameters from the input models. This is one of the most basic and widely used merging techniques.

**Use Cases:**

- Averaging multiple checkpoints of the same fine-tuning run ("model soups")
- Combining models with very similar architectures and training data
- Simple ensemble-like behavior in a single model

**Inputs:** Takes 2 or more models. No `base_model` is typically used.

**Key Parameters:**

- `weight` (per-model): The contribution of each model to the average
- `normalize` (global): If `true` (default), weights are normalized to sum to 1

**Reference:** [Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time](https://arxiv.org/abs/2203.05482)

---

## Spherical Interpolation Methods

### SLERP (`slerp`)

**Concept:** Performs Spherical Linear Interpolation in the weight space between two models. This creates a path along a hypersphere, ensuring the interpolated model maintains a similar "norm" or "magnitude" to the original models.

**Use Cases:**

- Creating smooth transitions or intermediate points between two distinct models
- Exploring the space between two models with potentially different capabilities

**Inputs:** Requires exactly 2 models. One model must be specified as `base_model`.

**Key Parameters:**

- `t` (global): Interpolation factor. `t=0` yields the `base_model`, `t=1` yields the other model

**Reference:** [Wikipedia: Slerp](https://en.wikipedia.org/wiki/Slerp)

### NuSLERP (`nuslerp`)

**Concept:** An enhanced version of SLERP offering more flexible configuration and faster execution. It allows SLERP between two models directly. If a `base_model` is provided, NuSLERP calculates task vectors (the difference between each of the two main models and the base model) and then performs SLERP on these task vectors before adding the result back to the `base_model`.

**Use Cases:**

- Similar to SLERP, but with more control over weighting for the two primary models.
- To replicate the behavior of the original slerp method (if `base_model` is *not* used, and weights are set to `1-t` for the first model and `t` for the second).
- To perform SLERP on task vectors when a `base_model` is provided, allowing for interpolation of model changes *relative* to a common ancestor.

**Inputs:** Requires exactly 2 models. A `base_model` can optionally be provided (it must be distinct from the two main models).

**Key Parameters:**

- `weight` (per-model): Relative weighting for each of the two main models. These are used to calculate the interpolation factor `t` (where `t = model2_weight / (model1_weight + model2_weight)`).
- `nuslerp_flatten` (global): If `false`, performs row/column-wise interpolation. Default `true`
- `nuslerp_row_wise` (global): If `true` (and `nuslerp_flatten` is `false`), SLERPs row vectors instead of column vectors. Default `false`

### Multi-SLERP (`multislerp`)

**Concept:** Implements barycentric interpolation on a hypersphere for more than two models. It projects points onto a tangent space at their weighted Euclidean mean, performs interpolation, and projects back.

**Use Cases:**

- Creating a spherical average of multiple models
- Finding a central point in the weight space of several related models

**Inputs:** Takes 2 or more models. A `base_model` can optionally be provided to operate in task vector space.

**Key Parameters:**

- `weight` (per-model): Relative weighting for each model
- `normalize_weights` (global): If `true` (default), weights are normalized
- `eps` (global): Small constant for numerical stability. Default `1e-8`

### Karcher Mean (`karcher`)

**Concept:** Computes the Karcher mean (also known as the Riemannian barycenter or Fréchet mean) of the input model parameters. This provides a geometrically sound way to average points on a manifold, which is suitable for model weights.

**Use Cases:**

- Finding a "central" or "average" model among a set of diverse models in a way that respects the geometry of the weight space
- More robust averaging than simple linear averaging, especially for models far apart in weight space

**Inputs:** Takes 2 or more models. No `base_model` is used.

**Key Parameters:**

- `max_iter` (global): Maximum iterations for the Karcher mean algorithm. Default `10`
- `tol` (global): Convergence tolerance. Default `1e-5`

**Reference:** [Wikipedia: Karcher mean](https://en.wikipedia.org/wiki/Karcher_mean)

---

## Task Vector Methods

*The following methods build upon the concept of "task vectors," which represent the difference between a fine-tuned model and a base model.*

### Task Arithmetic (`task_arithmetic`)

**Concept:** Computes "task vectors" for each model by subtracting a `base_model`. These task vectors are then combined as a weighted average and added back to the `base_model`.

**Use Cases:**

- Combining skills from multiple models fine-tuned from a common ancestor
- Transferring specific capabilities (e.g., coding ability, instruction following) from one model to another
- Steering style or behavior of a model by adding small task vectors from other models

**Inputs:** Requires a `base_model` and one or more other models.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector in the merge
- `lambda` (global): Scaling factor applied to the summed task vectors before adding back to the base. Default `1.0`

**Reference:** [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089)

### TIES-Merging (`ties`)

**Concept:** Builds on Task Arithmetic by sparsifying task vectors and applying a sign consensus algorithm. This helps to resolve interference when merging multiple models and retain more of their individual strengths.

**Use Cases:**

- Merging a larger number of models effectively
- Reducing parameter interference and negative synergy between merged models

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector
- `density` (per-model): Fraction of weights to retain in each sparsified task vector
- `lambda` (global): As in Task Arithmetic

**Reference:** [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)

### DARE (`dare_linear`, `dare_ties`)

**Concept:** Similar to TIES, DARE sparsifies task vectors to reduce interference. However, DARE uses random pruning with a novel rescaling technique to better match the performance of the original models.

**Variants:**

- `dare_linear`: DARE pruning without the TIES sign consensus
- `dare_ties`: DARE pruning *with* the TIES sign consensus

**Use Cases:**

- Robustly combining multiple fine-tuned models, often yielding better performance than TIES in some scenarios

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector
- `density` (per-model): Fraction of weights to retain after random pruning
- `lambda` (global): As in Task Arithmetic
- `rescale` (global, for `dare_linear`): If `true` (default), applies DARE's rescaling

**Reference:** [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)

### DELLA (`della`, `della_linear`)

**Concept:** Extends DARE by using adaptive pruning based on parameter magnitudes within each row of the delta parameters (task vectors). It calculates keep probabilities for each parameter: parameters with larger magnitudes within a row are assigned higher probabilities of being kept, while parameters with smaller magnitudes are assigned lower probabilities. These keep probabilities are scaled to range from `density - epsilon` (for the smallest magnitude element in a row) to `density + epsilon` (for the largest magnitude element in a row). This method aims to retain important changes while reducing interference, followed by DARE-like rescaling.

**Variants:**

- `della`: DELLA pruning with TIES sign consensus
- `della_linear`: DELLA pruning without TIES sign consensus

**Use Cases:**

- Fine-grained control over pruning by prioritizing parameters with larger magnitude changes
- Combining models where preserving the most significant changes is crucial

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector
- `density` (per-model): Target fraction of weights to retain in differences from the base model
- `epsilon` (per-model): Defines the half-width of the range for keep probabilities. Keep probabilities for parameters in a row will range from `density - epsilon` to `density + epsilon`, mapped from the smallest to largest magnitude parameters in that row, respectively. `epsilon` must be chosen such that `density - epsilon > 0` and `density + epsilon < 1`.
- `lambda` (global): As in Task Arithmetic

**Reference:** [DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling](https://arxiv.org/abs/2406.11617)

### Model Breadcrumbs (`breadcrumbs`, `breadcrumbs_ties`)

**Concept:** An extension of task arithmetic designed to sparsify task vectors by pruning parameters with both the smallest and the largest absolute magnitudes (often considered outliers). This method operates in two main steps on the task vector (the difference between a fine-tuned model and the `base_model`):

1. First, a `gamma` fraction of the parameters with the *largest* absolute magnitudes are identified for removal.
2. Then, parameters with the *smallest* absolute magnitudes are identified for removal. The quantity of these smallest parameters to remove is determined such that the final `density` of parameters *retained* in the task vector is achieved, after accounting for the largest ones removed.

The intention is to isolate and merge the "meaty," mid-range magnitude changes from the task vector, potentially filtering out noise (smallest changes) and overly dominant or conflicting large changes (largest changes).

**Variants:**

- `breadcrumbs`: Model Breadcrumbs pruning without TIES sign consensus
- `breadcrumbs_ties`: Model Breadcrumbs pruning *with* TIES sign consensus

**Use Cases:**

- Merging models where extreme parameter changes might be detrimental or noisy
- Refining task vectors by focusing on mid-range modifications, removing both the least significant and most extreme changes

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `weight` (per-model): Weight for each model's task vector.
- `gamma` (per-model): The fraction of parameters with the *largest* absolute magnitudes in the task vector to be pruned (removed). For example, a `gamma` of `0.01` targets the removal of the top 1% of parameters with the highest absolute values. This parameter corresponds to `β` (beta) as described in the reference paper.
- `density` (per-model): The final target fraction of parameters to *retain* in the task vector after both pruning steps (removal of largest `gamma` fraction and a corresponding fraction of smallest magnitude parameters).
  - The fraction of parameters with the *smallest* absolute magnitudes that will be pruned is calculated based on `density` and `gamma`. Specifically, it is `max(0, 1.0 - density - gamma)`.
  - **Example:** If `density: 0.9` and `gamma: 0.01`:
    - The top `0.01` (1%) largest magnitude parameters are removed.
    - The bottom `1.0 - 0.9 - 0.01 = 0.09` (9%) smallest magnitude parameters are also removed.
    - This results in `0.9` (90%) of the parameters being retained.
  - **Edge Case:** If `gamma` is set high enough such that `gamma >= 1.0 - density` (meaning `1.0 - density - gamma <= 0`), then the number of largest magnitude parameters actually pruned will be adjusted to `1.0 - density`, and no smallest magnitude parameters will be pruned (i.e., the fraction of smallest parameters pruned becomes 0). This ensures the `density` target is always respected and represents the fraction of parameters kept.
- `lambda` (global): As in Task Arithmetic.

**Reference:** [Model Breadcrumbs: Scaling Multi-Task Model Merging with Sparse Masks](https://arxiv.org/abs/2312.06795)

### SCE (`sce`)

**Concept:** The SCE (Select, Calculate, Erase) method performs adaptive matrix-level merging. It first computes task vectors (differences from the `base_model`). Then, it follows a three-step process for each parameter matrix (tensor):

1. **Select (Variance-Based Masking):** Optionally, parameter *positions* that show low variance across the different models' task vectors are identified and zeroed out. This is controlled by the `select_topk` parameter. If `select_topk < 1.0`, only the top `select_topk` fraction of parameter positions with the highest variance are kept active in the task vectors for subsequent steps.
2. **Calculate (Weighting):** Matrix-level merging coefficients (weights) are calculated for each model's task vector. These weights are derived from the mean of the squares of the elements within each task vector and are normalized across the models.
3. **Erase (Sign Consensus):** The sign-consensus algorithm from TIES is applied to the task vectors.

Finally, the (variance-selected, calculated-weighted, and sign-agreed) task vectors are summed together, normalized by the sum of the effective applied weights at each position, and then added back to the `base_model`.

**Use Cases:**

- Dynamically weighting the contribution of different models at the matrix level based on parameter variance and calculated importance
- Useful when some models contribute more significantly or consistently to certain parameter matrices than others
- Merging models by focusing on high-variance, consistently signed changes

**Inputs:** Requires 2 or more models, plus one `base_model`.

**Key Parameters:**

- `select_topk` (global): The fraction of parameter positions to retain based on their variance values across the different input models' task vectors. For each parameter position, variance is calculated across all task vectors. Only positions corresponding to the `select_topk` fraction with the highest variances are kept (i.e., their values in all task vectors are preserved for the next steps). Positions with lower variance are zeroed out in all task vectors. Set to 1.0 (default) to disable this variance-based selection step. This corresponds to `τ` (tau) in the reference paper.

**Reference:** [FuseChat: Knowledge Fusion of Chat Models](https://arxiv.org/abs/2408.07990)

---

## Specialized Methods

### Model Stock (`model_stock`)

**Concept:** Uses geometric properties of fine-tuned models relative to a base_model to compute an optimized interpolation weight. It then performs a linear interpolation between the `base_model` and the average of the other input models using this computed weight. Specifically, task vectors (differences between other models and the `base_model`) are used to calculate pairwise cosine similarities. The average of these similarities informs the interpolation factor `t`. The final merged tensor is `t * average_of_other_models + (1 - t) * base_model`.

**Use Cases:**

- Finding effective weights for linearly combining multiple models where one model serves as a clear reference (`base_model`).
- When a more principled, data-driven approach to linear interpolation between a base and a group of variants is desired over manual weight tuning.
- Particularly strong for combining different training runs that were fine-tuned from the same `base_model` over the same or similar datasets.

**Inputs:** Requires at least 3 models: one `base_model` and at least two other models.

**Key Parameters:**

- `filter_wise` (global): If `true`, weight calculation is per-row rather than per-tensor (not generally recommended). Default `false`

**Reference:** [Model Stock: All we need is just a few fine-tuned models](https://arxiv.org/abs/2403.19522)

### Nearswap (`nearswap`)

**Concept:** Interpolates the base model with parameters from a secondary model primarily where they are already similar. The interpolation strength towards the secondary model is inversely proportional to the absolute difference of their parameters, modulated by the `t` parameter. When the parameters are similar, the interpolation is stronger, and when they are different, it is weaker.

**Use Cases:**

- Selectively pulling in similar parameters from a secondary model while preserving different parameters from the base model
- Fine-grained parameter-wise merging that respects the existing structure of the base model

**Inputs:** Requires exactly 2 models. One model must be specified as `base_model`.

**Key Parameters:**

- `t` (global): Controls the interpolation strength. Higher values increase the influence of the secondary model for similar parameters

**Algorithm:** For each parameter, computes `weight = (t / |base - secondary|).clamp(0, 1)`, then returns `weight * secondary + (1 - weight) * base`

**Reference:** [QuartetAnemoi-70B-t0.0001 on Hugging Face](https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001)

### Arcee Fusion (`arcee_fusion`)

**Concept:** Merges two models by dynamically identifying and fusing important parameter changes. It calculates importance scores based on parameter differences and KL divergence, then uses a dynamic threshold to create a fusion mask.

**Use Cases:**

- Intelligently combining two models by prioritizing the most salient differences

**Inputs:** Requires exactly 2 models. One model must be specified as `base_model`.

**Key Parameters:** None beyond standard model selection

**Reference:** [MergeKit v0.1 Release Blog](https://www.arcee.ai/blog/meet-mergekit-v0-1-arcee-fusion-expanded-model-support-multi-gpu-acceleration)

### Passthrough (`passthrough`)

**Concept:** A no-op merge method that simply passes input tensors through unmodified from a single input model.

**Use Cases:**

- Layer-stacking or "Frankenmerging" where you assemble a model from specific unmodified layer ranges or individual tensors from one or more "donor" models
- Useful as a building block in more complex `slices` configurations

**Inputs:** Takes exactly 1 model.

**Key Parameters:**

- `scale` (per-model, optional): A scalar to multiply the tensor by. Useful for scaling specific layers, e.g., `{"filter": "down_proj", "value": 0.5}`

---

## Summary

Merge methods serve different purposes and have different design goals. The choice of method depends on your specific use case, including the number of models, their relationships, and the desired characteristics of the final merged model.

For beginners, starting with `linear`, `nuslerp`, or `task_arithmetic` can provide good results. For more advanced use cases, methods like `ties`, `dare_ties`, or `della` offer sophisticated ways to handle interference between multiple models while preserving their individual strengths. There is no "best" merge method; the right choice depends on your specific needs and the models you are working with, and selection is often more art than science. I encourage you to experiment with different methods and parameters vigorously to both find the best results for your use case and to learn more about how these methods work. Happy merging!

## Contributing

If you have ideas for new merge methods or improvements to existing ones, we'd be glad to have you involved! Check out the [Contributing Guide](../CONTRIBUTING.md) and [Creating a Merge Method](create_a_merge_method.md) for more information on how to get started.
