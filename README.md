# LRP-Merge: Layer-wise Relevance Propagation for LLM Merging

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

`mergekit` is a toolkit for merging pre-trained language models, using a variety of merge methods including TIES, linear, and slerp merging. The toolkit also enables piecewise assembly of a language model from layers.

This branch features a new unified merge script that takes a YAML configuration file defining the operations to perform.

## Configuration

Below are the primary elements of a configuration file:

- `merge_method`: Specifies the method to use for merging models. Can be one of 'ties', 'linear', 'slerp', or 'passthrough'.
- `slices`: Defines slices of layers from different models to be used. This field is mutually exclusive with `models`.
- `models`: Defines entire models to be used for merging. This field is mutually exclusive with `slices`.
- `base_model`: Specifies the base model used in some merging methods.
- `parameters`: Holds various parameters such as weights and densities, which can also be specified at different levels of the configuration.
- `dtype`: Specifies the data type for the merging operation.

### Parameter Specification

Parameters are flexible and can be set with varying precedence. They can be specified conditionally using tensor name filters, which allows finer control such as differentiating between attention heads and fully connected layers.

Parameters can be specified as:

- **Scalars**: Single floating-point values.
- **Gradients**: List of floating-point values, specifying an interpolated gradient.

The parameters can be set at different levels, with decreasing precedence as follows:

1. `slices.*.sources.parameters` - applying to a specific input slice
2. `slices.*.parameters` - applying to a specific output slice
3. `input_model_parameters` - applying to any tensors coming from specific input models
4. `parameters` - catchall


### Merge Methods

#### **[Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)** (`"ties"`)
Requires a base model.
Parameters:
- `density` - fraction of weights in differences from the base model to retain
- `weight` - relative (or absolute if `normalize=False`) weighting of a given tensor
- `normalize` - if true, the weights of all models contributing to a tensor will be normalized. Default behavior.


#### Linear
Does not require a base model. Takes parameters `weight` and `normalize`, with same definition as above.


#### SLERP
Requires exactly two models, one of which must be the base model. Takes one parameter - `t` - the interpolation factor from the base model to the secondary model.

### Examples

- Simple linear merge of multiple models:

  ```yml
  models:
    - model: psmathur/orca_mini_v3_13b
      parameters:
        weight: 1.0
    - model: WizardLM/WizardLM-13B-V1.2
      parameters:
        weight: 0.3
    - model: garage-bAInd/Platypus2-13B
      parameters:
        weight: 0.5
  merge_method: linear
  dtype: float16
  ```

- `bakllama.py` style layer recombination:

  ```yml
  slices:
    - sources:
      - model: psmathur/orca_mini_v3_13b
        layer_range: [0, 24]
    - sources:
      - model: garage-bAInd/Platypus2-13B
        layer_range: [20, 40]
  merge_method: passthrough
  dtype: float16
  ```

- Gradient SLERP with different weights for mlp/self attention:

  ```yml
  slices:
    - sources:
        - model: psmathur/orca_mini_v3_13b
          layer_range: [0, 40]
        - model: garage-bAInd/Platypus2-13B
          layer_range: [0, 40]
  merge_method: slerp
  base_model: psmathur/orca_mini_v3_13b
  parameters:
    t:
      - filter: self_attn
        value: [0, 0.5, 0.3, 0.7, 1]
      - filter: mlp
        value: [1, 0.5, 0.7, 0.3, 0]
      - value: 0.5 # fallback for rest of tensors
  dtype: float16
  ```

#### Usage

Once you have created the YAML configuration file, run `main.py` with the config file and output path as arguments:

```sh
python main.py path/to/your/config.yml ./output-model-directory [--cuda]
```
