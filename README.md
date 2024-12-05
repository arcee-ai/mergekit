# LRP-Merge: Layer-wise Relevance Propagation for LLM Merging

> [!NOTE]
> [Charles Goddard](https://github.com/cg123), author of `mergekit`, has joined [arcee.ai](https://www.arcee.ai/). Development of `mergekit` continues with their backing and it remains open source. For more details, see the blog post [here](https://blog.arcee.ai/arcee-and-mergekit-unite/).

LRP-Merge is a custom model merging method built on top of [Mergekit](https://github.com/arcee-ai/mergekit). It uses **Layer-wise Relevance Propagation (LRP)** scores to identify and preserve functionally critical weights during model merging.

---

##  The Core Concept

### The Problem with Standard Merging

Traditional merging (linear averaging) dilutes fine-tuned knowledge because it treats all weights equally. In reality, only a small fraction of weights drive new capabilities.

### The LRP Solution

LRP-Merge applies an XAI technique to score each weight's contribution to correct predictions:

1. **Compute Task Vector:** `δ = θ_fine_tuned - θ_base`
2. **LRP-Based Trimming:** Keep weights with highest LRP relevance scores
3. **Weighted Averaging:** Merge sparse task vectors, add back to base

---

# Instructions for Running LRP-Merge on Google Colab

This guide explains how to train your models using Google Colab's free GPU, which is much faster than training on a CPU-only laptop.

## Why Use Colab?

| Feature | Your Laptop (CPU) | Google Colab (Free GPU) |
|---------|-------------------|-------------------------|
| Training Time (1 epoch) | 2-4 hours | 5-15 minutes |
| Model Size Supported | GPT-2 (124M) | TinyLlama (1.1B) |
| Batch Size | 1 | 4-8 |
| Max Sequence Length | 64-128 | 256-512 |

## Prerequisites

1. A Google account (free)
2. Your LRP merge method files uploaded to Google Drive
3. This notebook: `LRP_Merge_Colab_Training.ipynb`

## Required Files to Upload

You need to upload these files from your `D:\LRP merge method` folder to Google Drive:

### Core Python Files (Required)
```
LRP merge method/
├── finetune_fakenews.py       # Training script (FIXED for Colab)
├── evaluate_models.py         # Evaluation script (FIXED for Colab)
├── lrp_merge_pipeline.py      # LRP merge pipeline (FIXED for Colab)
├── lrp_computer.py           # LRP score computation (FIXED for Colab)
```

### Supporting Files
```
├── base.py                   # Base classes
├── common.py                 # Common utilities
├── graph.py                  # Graph operations
├── tasks.py                  # Task definitions
├── sparsify.py              # Sparsification
├── embed.py                 # Embedding utilities
├── registry.py              # Registry
└── __init__.py              # Package init
```

### Configuration Files
```
├── lrp_config.yaml          # LRP merge configuration template
└── pyproject.toml          # Python dependencies (optional)
```

### Your Datasets (Important!)
```
datasets/
└── synthetic/               # Your downloaded datasets
    ├── train.csv           # Training data (~800 samples)
    └── test.csv            # Test data (~200 samples)
```

**Note:** The notebook is configured to use `datasets/synthetic/train.csv` by default.

### Optional: Mergekit Repository
```
mergekit_repo/              # Only if you're using custom merge methods
└── mergekit/
    └── ...
```

### Complete File List (Copy-Paste Ready)

Create a ZIP file with these files:
```
base.py
common.py
embed.py
finetune_fakenews.py
evaluate_models.py
graph.py
lrp.py
lrp_computer.py
lrp_config.yaml
lrp_merge_pipeline.py
pyproject.toml
registry.py
sparsify.py
tasks.py
__init__.py
download_fakenews_datasets.py
datasets/synthetic/train.csv
datasets/synthetic/test.csv
```

## Step-by-Step Instructions

### Step 1: Prepare Your Files

#### Option A: Upload to Google Drive (Recommended)

1. Open [Google Drive](https://drive.google.com)
2. Create a folder called `LRP merge method`
3. Upload all files from your `D:\LRP merge method` folder to this Drive folder
4. Wait for upload to complete

#### Option B: Upload Directly to Colab

1. Create a ZIP file of your `D:\LRP merge method` folder
2. Upload this notebook to Colab
3. Skip to "Upload Files Manually" section in the notebook

### Step 2: Open the Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **File** → **Upload notebook**
3. Select `LRP_Merge_Colab_Training.ipynb`
4. Wait for the notebook to load

### Step 3: Enable GPU Runtime

**IMPORTANT: This step is crucial!**

1. Click **Runtime** in the top menu
2. Select **Change runtime type**
3. Under "Hardware accelerator", select **GPU**
4. Click **Save**
5. The runtime will restart

**Verify GPU is enabled:**
- Look at the top right corner
- It should say "RAM" and "Disk" with a checkmark
- If you see "Connecting" or "Busy", wait for it to connect

### Step 4: Run the Notebook

1. Click **Runtime** → **Run all** (or run cells one by one)
2. When prompted, authenticate Google Drive:
   - A link will appear
   - Click the link and sign in to your Google account
   - Copy the authorization code
   - Paste it back in Colab
   - Press Enter

### Step 5: Wait for Training

The notebook will automatically:
1. Mount your Google Drive
2. Install all dependencies
3. Create sample datasets
4. Train the GLOBAL model (~10-15 minutes)
5. Train the LOCAL model (~10-15 minutes)
6. Compute LRP scores (~5 minutes)
7. Merge the models (~5 minutes)
8. Test and evaluate (~5 minutes)

**Total time: ~40-60 minutes**

### Step 6: Save Results

The trained models are automatically saved to your Google Drive at:
```
My Drive/LRP merge method/models/ (varies based on user's folder's structure)
├── tinyllama-global/  (varies from model to model)
├── tinyllama-local/
└── merged-model/
```

You can also download them as a ZIP file (see the last cell).

## Configuration Options

### Change the Model

In the notebook, find this cell:
```python
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (varies from model to model)
```

Options:
- `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` - Good quality, requires ~8GB GPU
- `"gpt2"` - Fastest, works on any GPU
- `"gpt2-medium"` - Balance of speed and quality

### Change Training Parameters

```python
EPOCHS = 3          # Increase for better results (try 5-10)
BATCH_SIZE = 4      # Can increase if you have more GPU memory
MAX_SAMPLES = 1000  # Increase for better training (try 5000-10000)
```

### Use Your Own Dataset

1. Upload your dataset CSV to Google Drive in the `datasets` folder
2. Update the path in the notebook:
```python
DATASET = "datasets/your_dataset.csv"
```

## Troubleshooting

### "No GPU detected" Error

**Solution:**
1. Go to Runtime → Change runtime type
2. Select GPU
3. Click Save
4. Runtime will restart - run cells again

### "Out of Memory" Error

**Solution:** Reduce memory usage:
```python
BATCH_SIZE = 2      # Reduce from 4
MAX_SAMPLES = 500   # Reduce from 1000
MODEL_NAME = "gpt2" # Use smaller model
```

### "File not found" Error

**Solution:** Check your Google Drive path:
```python
# Update this to match your Drive structure
LRP_PATH = "/content/drive/MyDrive/LRP merge method"
```

### Training Takes Too Long

**Solutions:**
1. Use smaller model: `MODEL_NAME = "gpt2"`
2. Reduce samples: `MAX_SAMPLES = 500`
3. Reduce epochs: `EPOCHS = 1`

### Runtime Disconnected

Colab may disconnect after ~90 minutes of inactivity or 12 hours max.

**Solutions:**
1. Click around periodically to keep it active
2. Save results to Drive frequently (the notebook does this automatically)
3. Use Colab Pro ($10/month) for longer runtimes

## Colab Pro vs Free

| Feature | Free | Pro ($10/month) |
|---------|------|-----------------|
| GPU | K80/T4 | T4/P100 |
| Runtime limit | 12 hours | 24 hours |
| Idle timeout | 90 min | None |
| Background execution | No | Yes |

For LRP-Merge training, the **free tier is usually sufficient**.

## Downloading Results

### Method 1: From Google Drive
1. Open [Google Drive](https://drive.google.com)
2. Navigate to `My Drive/LRP merge method/models/`
3. Right-click the model folder → Download

### Method 2: Direct Download from Colab
Run the last cell in the notebook which creates a ZIP file and downloads it.

### Method 3: Use `gdown` (Command Line)
```bash
pip install gdown
gdown <file_id_from_drive>
```

## Tips for Success

1. **Start small:** Run with `MAX_SAMPLES=500` first to test
2. **Monitor GPU:** Click the RAM/Disk indicator to see GPU usage
3. **Save frequently:** The notebook saves to Drive automatically
4. **Don't close browser:** Keep the Colab tab open during training
5. **Use Chrome:** Works best with Google Chrome browser

## Advanced: Training Multiple Models

To train multiple variations, duplicate the training cells:

```python
# Model A with different learning rate
!python finetune_fakenews.py \
    --dataset datasets/global_train.csv \
    --output models/global-v2 \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 5 \
    --lr 5e-5  # Different learning rate

# Model B with more data
!python finetune_fakenews.py \
    --dataset datasets/global_train_large.csv \
    --output models/global-v3 \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --epochs 5 \
    --max-samples 5000
```

## Next Steps After Training

1. Download the merged model
2. Use it locally with:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/merged-model")
tokenizer = AutoTokenizer.from_pretrained("path/to/merged-model")
```
3. Or deploy it to Hugging Face Hub

## Questions?

- Check the notebook's error messages
- Look at the output logs in each cell
- Make sure your Drive is properly mounted
- Verify GPU is enabled in Runtime settings

---

**Remember:** Colab sessions are temporary. Always save your results to Google Drive before closing!


## mergekit

> [!NOTE]  
> [Charles Goddard](https://github.com/cg123), author of `mergekit`, has joined [arcee.ai](https://www.arcee.ai/). Development of `mergekit` continues with their backing and it remains open source. For more details, see the blog post [here](https://blog.arcee.ai/arcee-and-mergekit-unite/).

# mergekit

`mergekit` is a toolkit for merging pre-trained language models. `mergekit` uses an out-of-core approach to perform unreasonably elaborate merges in resource-constrained situations. Merges can be run entirely on CPU or accelerated with as little as 8 GB of VRAM. Many merging algorithms are supported, with more coming as they catch my attention.

Features:

- Supports Llama, Mistral, GPT-NeoX, StableLM, and more
- Many [merge methods](#merge-methods)
- GPU or CPU execution
- Lazy loading of tensors for low memory use
- Interpolated gradients for parameter values (inspired by Gryphe's [BlockMerge_Gradient](https://github.com/Gryphe/BlockMerge_Gradient) script)
- Piecewise assembly of language models from layers ("Frankenmerging")
- [Mixture of Experts merging](#mixture-of-experts-merging)
- [LORA extraction](#lora-extraction)
- [Evolutionary merge methods](#evolutionary-merge-methods)

🌐 GUI Launch Alert 🤗 - We are excited to announce the launch of a mega-GPU backed graphical user interface for mergekit in Arcee! This GUI simplifies the merging process, making it more accessible to a broader audience. Check it out and contribute at the [Arcee App](https://app.arcee.ai). There is also a [Hugging Face Space](https://huggingface.co/mergekit-community) with limited amounts of GPUs.

## Installation

```sh
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit

pip install -e .  # install the package and make scripts available
```

If the above fails with the error of:

```
ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode:
(A "pyproject.toml" file was found, but editable mode currently requires a setuptools-based build.)
```

You may need to upgrade pip to > 21.3 with the command `python3 -m pip install --upgrade pip`

## Usage

The script `mergekit-yaml` is the main entry point for `mergekit`. It takes a YAML configuration file and an output path, like so:

```sh
mergekit-yaml path/to/your/config.yml ./output-model-directory [--cuda] [--lazy-unpickle] [--allow-crimes] [... other options]
```

This will run the merge and write your merged model to `./output-model-directory`.

For more information on the arguments accepted by `mergekit-yaml` run the command `mergekit-yaml --help`.

### Uploading to Huggingface

When you have a merged model you're happy with, you may want to share it on the Hugging Face Hub. `mergekit` generates a `README.md` for your merge with some basic information for a model card. You can edit it to include more details about your merge, like giving it a good name or explaining what it's good at; rewrite it entirely; or use the generated `README.md` as-is. It is also possible to edit your `README.md` online once it has been uploaded to the Hub.

Once you're happy with your model card and merged model, you can upload it to the Hugging Face Hub using the [huggingface_hub](https://huggingface.co/docs/huggingface_hub/index) Python library.

```
# log in to huggingface with an access token (must have write permission)
huggingface-cli login
# upload your model
huggingface-cli upload your_hf_username/my-cool-model ./output-model-directory .
```

The [documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-upload) for `huggingface_hub` goes into more detail about other options for uploading.

## Merge Configuration

Merge configurations are YAML documents specifying the operations to perform in order to produce your merged model.
Below are the primary elements of a configuration file:

- `merge_method`: Specifies the method to use for merging models. See [Merge Methods](#merge-methods) for a list.
- `slices`: Defines slices of layers from different models to be used. This field is mutually exclusive with `models`.
- `models`: Defines entire models to be used for merging. This field is mutually exclusive with `slices`.
- `base_model`: Specifies the base model used in some merging methods.
- `parameters`: Holds various parameters such as weights and densities, which can also be specified at different levels of the configuration.
- `dtype`: Specifies the data type used for the merging operation.
- `tokenizer_source`: Determines how to construct a tokenizer for the merged model.

### Parameter Specification

Parameters are flexible and can be set with varying precedence. They can be specified conditionally using tensor name filters, which allows finer control such as differentiating between attention heads and fully connected layers.

Parameters can be specified as:

- **Scalars**: Single floating-point values.
- **Gradients**: List of floating-point values, specifying an interpolated gradient.

The parameters can be set at different levels, with decreasing precedence as follows:

1. `slices.*.sources.parameters` - applying to a specific input slice
2. `slices.*.parameters` - applying to a specific output slice
3. `models.*.parameters` or `input_model_parameters` - applying to any tensors coming from specific input models
4. `parameters` - catchall

### Tokenizer Source

The `tokenizer_source` field of a configuration file determines what tokenizer is used by the merged model. This also effects how embeddings and language model heads are merged.

This functionality is still experimental and may break. Please file an issue if you encounter any issues with it.

Valid values:

- `base`: use the tokenizer from the base model
- `union`: construct a tokenizer with all tokens from all models
- `model:<model_path>`: use the tokenizer from a specific model

If set, mergekit will find a mapping between each model's vocabulary and the output tokenizer. This allows models with different vocabularies or added tokens to be meaningfully merged.

`tokenizer_source` is compatible with all merge methods, but when used `lm_head`/`embed_tokens` will be merged linearly. For two-model merges, the `embed_slerp` parameter can be set to `true` to use SLERP instead.

If the `tokenizer_source` field is not set, mergekit will fall back to its legacy default behavior. The tokenizer for the base model (or first model in the merge, if no base model is specified) will be copied to the output directory. The parameter matrices for `lm_head`/`embed_tokens` will be truncated to the smallest size present in the merge. In _most_ cases this corresponds to using the tokenizer for the base model.

### Examples

Several examples of merge configurations are available in [`examples/`](examples/).

## Merge Methods

A quick overview of the currently supported merge methods:

| Method                                                                                           | `merge_method` value | Multi-Model | Uses base model |
| ------------------------------------------------------------------------------------------------ | -------------------- | ----------- | --------------- |
| Linear ([Model Soups](https://arxiv.org/abs/2203.05482))                                         | `linear`             | ✅          | ❌              |
| SLERP                                                                                            | `slerp`              | ❌          | ✅              |
| [Task Arithmetic](https://arxiv.org/abs/2212.04089)                                              | `task_arithmetic`    | ✅          | ✅              |
| [TIES](https://arxiv.org/abs/2306.01708)                                                         | `ties`               | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [TIES](https://arxiv.org/abs/2306.01708)                | `dare_ties`          | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [Task Arithmetic](https://arxiv.org/abs/2212.04089)     | `dare_linear`        | ✅          | ✅              |
| Passthrough                                                                                      | `passthrough`        | ❌          | ❌              |
| [Model Breadcrumbs](https://arxiv.org/abs/2312.06795)                                            | `breadcrumbs`        | ✅          | ✅              |
| [Model Breadcrumbs](https://arxiv.org/abs/2312.06795) + [TIES](https://arxiv.org/abs/2306.01708) | `breadcrumbs_ties`   | ✅          | ✅              |
| [Model Stock](https://arxiv.org/abs/2403.19522)                                                  | `model_stock`        | ✅          | ✅              |
| [DELLA](https://arxiv.org/abs/2406.11617)                                                  | `della`        | ✅          | ✅              |
| [DELLA](https://arxiv.org/abs/2406.11617) [Task Arithmetic](https://arxiv.org/abs/2212.04089)                                                  | `della_linear`        | ✅          | ✅              |
### Linear

The classic merge method - a simple weighted average.

Parameters:

- `weight` - relative (or absolute if `normalize=False`) weighting of a given tensor
- `normalize` - if true, the weights of all models contributing to a tensor will be normalized. Default behavior.

### SLERP

Spherically interpolate the parameters of two models. One must be set as `base_model`.

Parameters:

- `t` - interpolation factor. At `t=0` will return `base_model`, at `t=1` will return the other one.

### [Task Arithmetic](https://arxiv.org/abs/2212.04089)

Computes "task vectors" for each model by subtracting a base model. Merges the task vectors linearly and adds back the base. Works great for models that were fine tuned from a common ancestor. Also a super useful mental framework for several of the more involved merge methods.

Parameters: same as [Linear](#linear)

### [TIES](https://arxiv.org/abs/2306.01708)

Builds on the task arithmetic framework. Resolves interference between models by sparsifying the task vectors and applying a sign consensus algorithm. Allows you to merge a larger number of models and retain more of their strengths.

Parameters: same as [Linear](#linear), plus:

- `density` - fraction of weights in differences from the base model to retain

### [DARE](https://arxiv.org/abs/2311.03099)

In the same vein as TIES, sparsifies task vectors to reduce interference. Differs in that DARE uses random pruning with a novel rescaling to better match performance of the original models. DARE can be used either with the sign consensus algorithm of TIES (`dare_ties`) or without (`dare_linear`).

Parameters: same as [TIES](#ties) for `dare_ties`, or [Linear](#linear) for `dare_linear`

### Passthrough

`passthrough` is a no-op that simply passes input tensors through unmodified. It is meant to be used for layer-stacking type merges where you have only one input model. Useful for frankenmerging.

### [Model Breadcrumbs](https://arxiv.org/abs/2312.06795)

An extension of task arithmetic that discards both small and extremely large differences from the base model. As with DARE, the Model Breadcrumbs algorithm can be used with (`breadcrumbs_ties`) or without (`breadcrumbs`) the sign consensus algorithm of TIES.

Parameters: same as [Linear](#linear), plus:

- `density` - fraction of weights in differences from the base model to retain
- `gamma` - fraction of largest magnitude differences to remove

Note that `gamma` corresponds with the parameter `β` described in the paper, while `density` is the final density of the sparsified tensors (related to `γ` and `β` by `density = 1 - γ - β`). For good default values, try `density: 0.9` and `gamma: 0.01`.

### [Model Stock](https://arxiv.org/abs/2403.19522)

Uses some neat geometric properties of fine tuned models to compute good weights for linear interpolation. Requires at least three models, including a base model.

Parameters:

- `filter_wise`: if true, weight calculation will be per-row rather than per-tensor. Not recommended.

### [DELLA](https://arxiv.org/abs/2406.11617)

Building upon DARE, DELLA uses adaptive pruning based on parameter magnitudes. DELLA first ranks parameters in each row of delta parameters and assigns drop probabilities inversely proportional to their magnitudes. This allows it to retain more important changes while reducing interference. After pruning, it rescales the remaining parameters similar to [DARE](#dare). DELLA can be used with (`della`) or without (`della_linear`) the sign elect step of TIES

Parameters: same as [Linear](#linear), plus:
- `density` - fraction of weights in differences from the base model to retain
- `epsilon` - maximum change in drop probability based on magnitude. Drop probabilities assigned will range from `density - epsilon` to `density + epsilon`. (When selecting values for `density` and `epsilon`, ensure that the range of probabilities falls within 0 to 1)
- `lambda` - scaling factor for the final merged delta parameters before merging with the base parameters.

## LoRA extraction

Mergekit allows extracting PEFT-compatible low-rank approximations of finetuned models.

### Usage

```sh
mergekit-extract-lora finetuned_model_id_or_path base_model_id_or_path output_path [--no-lazy-unpickle] --rank=desired_rank
```

## Mixture of Experts merging

The `mergekit-moe` script supports merging multiple dense models into a mixture of experts, either for direct use or for further training. For more details see the [`mergekit-moe` documentation](docs/moe.md).

## Evolutionary merge methods

See `docs/evolve.md` for details.

## ✨ Merge in the Cloud ✨

We host merging on Arcee's cloud GPUs - you can launch a cloud merge in the [Arcee App](https://app.arcee.ai). Or through python - grab an ARCEE_API_KEY:

`export ARCEE_API_KEY=<your-api-key>`
`pip install -q arcee-py`

```
import arcee
arcee.merge_yaml("bio-merge","./examples/bio-merge.yml")
```

Check your merge status at the [Arcee App](https://app.arcee.ai)

When complete, either deploy your merge:

```
arcee.start_deployment("bio-merge", merging="bio-merge")
```

Or download your merge:

`!arcee merging download bio-merge`


## Citation

We now have a [paper](https://arxiv.org/abs/2403.13257) you can cite for the MergeKit library:

```bibtex
@article{goddard2024arcee,
  title={Arcee's MergeKit: A Toolkit for Merging Large Language Models},
  author={Goddard, Charles and Siriwardhana, Shamane and Ehghaghi, Malikeh and Meyers, Luke and Karpukhin, Vlad and Benedict, Brian and McQuade, Mark and Solawetz, Jacob},
  journal={arXiv preprint arXiv:2403.13257},
  year={2024}
}
```
