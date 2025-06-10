# mergekit-tokensurgeon

`mergekit-tokensurgeon` is a command line utility for "transplanting" tokenizers between models. It reconstructs embeddings for a donor tokenizer inside the base model's embedding space so that the resulting model can operate with the donor vocabulary and ID mapping.

The default approach uses **Orthogonal Matching Pursuit (OMP)** to approximate unseen token embeddings as sparse combinations of tokens shared between the two vocabularies. This provides a training-free way to align tokenizers with minimal loss in downstream performance. The method is described in detail in the paper [*Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit*](https://arxiv.org/abs/2506.06607).

Other approximation strategies are also implemented (e.g. common-vocabulary interpolation, subword based methods, PCA and more). You can control which technique is used via the `--approximation-method` option described below.

## Usage

```bash
mergekit-tokensurgeon \
  path/to/base_model \
  path/to/donor_model \
  ./output_model \
  [options]
```

This command creates a new model at `./output_model` whose tokenizer matches the donor model. The main embeddings and language modeling head are updated so existing weights remain aligned with the new vocabulary.

### Key Options

- `--k`: Sparsity level (e.g., for `omp`, `stb`, `mp_rope`) or number of neighbors (e.g., for `common_interpolation`). (default: 64).
- `--approximation-method`: One of `omp` (default), `common_interpolation`, `subword`, `mean`, `zero`, `randn`, `john_hewitt`, `landmark_pca`, `stb`, or `mp_rope`.
- `--weight-scheme`: Weighting scheme for common interpolation (`distance_proportional`, `barycentric`, `least_squares`).
- `--subword-method`: How to combine subword pieces when using the `subword` method (`mean`, `sum`, `weighted_mean`, `first_last`).
- `--prefix-match` / `--byte-match`: Reuse existing embeddings that share a prefix or byte representation with donor tokens.
- `--magikarp`: Filter out poorly trained tokens using the Magikarp heuristic before approximation.
- Standard mergekit options such as `--device` and `--trust-remote-code` are also accepted.

Run `mergekit-tokensurgeon --help` for the full list of arguments.

## Approximation Methods

`mergekit-tokensurgeon` implements a number of strategies for generating embeddings for tokens that do not exist in the base model. The method is selected with `--approximation-method`:

- **omp** – Orthogonal Matching Pursuit (default). Approximates each missing token as a sparse linear combination of up to `--k` shared tokens.
- **common_interpolation** – Finds the nearest overlapping tokens and interpolates between them using one of several weighting schemes (controlled with `--weight-scheme`).
- **subword** – Breaks the token into pieces using the base tokenizer and combines their embeddings according to `--subword-method`.
- **landmark_pca** – Builds a linear map between the donor and base embedding spaces from the shared tokens using PCA and applies it to the new tokens.
- **stb** – Forms sparse (approximately) orthogonal token bases for both models and transfers coefficients between them.
- **mp_rope** – Matching pursuit that accounts for rotary position embeddings to better align positional structure.
- **john_hewitt** – Samples from the distribution of the base embeddings to generate new vectors. See [John Hewitt's page](https://www.cs.columbia.edu/~johnhew/vocab-expansion.html) for details.
- **mean**, **zero**, **randn** – Simpler heuristics that fill new tokens with the base embedding mean, all zeros, or Gaussian noise respectively.

## Practical Tips

- For most models we recommend `--approximation-method omp --k 64`, which balances quality and compute cost.
- The script operates entirely offline; no additional training is required.
- Large discrepancies in numerical tokenization schemes can degrade math-heavy tasks. If both tokenizers split numbers in a similar way the transplanted model usually retains its arithmetic ability.

## Further Reading

The theoretical motivation, implementation details and empirical evaluation of the OMP approach are presented in the accompanying paper. The `mergekit-tokensurgeon` tool exposes these techniques for practical use in merging workflows.
