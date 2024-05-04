# mergekit-moe

`mergekit-moe` is a script for combining Mistral or Llama models of the same size into Mixtral Mixture of Experts models. The script will combine the self-attention and layer normalization parameters from a "base" model with the MLP parameters from a set of "expert" models.

If using the `hidden` or `cheap_embed` gate mode, the output model will be usable without any further training. If you are initializing a model to do further training on, such as for sparse upcycling, then use the `random` gate mode to get a model ready for training.

## Configuration

`mergekit-moe` uses its own YML configuration syntax, which looks like so:

```yml
base_model: path/to/self_attn_donor
gate_mode: hidden # one of "hidden", "cheap_embed", or "random"
dtype: bfloat16 # output dtype (float32, float16, or bfloat16)
## (optional)
# experts_per_token: 2
experts:
  - source_model: expert_model_1
    positive_prompts:
      - "This is a prompt that is demonstrative of what expert_model_1 excels at"
    ## (optional)
    # negative_prompts:
    #   - "This is a prompt expert_model_1 should not be used for"
  - source_model: expert_model_2
  # ... and so on
```

The script takes two arguments, an input config and an output path: `mergekit-moe ./config.yml ./my-clowncar-moe-12x180B`

Currently the script can output models that use the Mixtral, Deepseek MoE, or Qwen MoE architectures. Some output architectures support a shared expert which will be activated for all tokens, which can be configured like this:

```yml
base_model: path/to/self_attn_donor
gate_mode: hidden # one of "hidden", "cheap_embed", or "random"
dtype: bfloat16 # output dtype (float32, float16, or bfloat16)
experts:
  ...
shared_experts:
  - source_model: model_name
    positive_prompts: # required by Qwen MoE for "hidden" gate mode, otherwise not allowed
      - "blah blah"
    # (optional, but recommended:)
    residual_scale: 0.1 # downweight output from shared expert to prevent overcooking the model
```

Currently only up to one shared expert is supported.

An appropriate architecture will be inferred based on the input models and presence or absence of shared experts in your configuration. Alternatively, you can explicitly specify an output architecture by setting the `architecture:` field in your config. For example:

```yml
base_model: path/to/self_attn_donor
architecture: qwen
# ... and so on
```

### Gate Modes

There are three methods for populating the MoE gates implemented.

#### "hidden"

Uses the hidden state representations of the positive/negative prompts for MoE gate parameters. Best quality and most effective option; the default. Requires evaluating each prompt using the base model so you might not be able to use this on constrained hardware (depending on the model). You can use `--load-in-8bit` or `--load-in-4bit` to reduce VRAM usage.

#### "cheap_embed"

Uses only the raw token embedding of the prompts, using the same gate parameters for every layer. Distinctly less effective than "hidden". Can be run on much, much lower end hardware.

#### "random"

Randomly initializes the MoE gates. Good for if you are going to fine tune the model afterwards, or maybe if you want something a little unhinged? I won't judge.

## Example Configurations

Sparse upcycling of smol_llama into a 8x220M MoE:

```yml
base_model: BEE-spoke-data/smol_llama-220M-GQA
gate_mode: random
dtype: bfloat16
experts:
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
  - source_model: BEE-spoke-data/smol_llama-220M-GQA
# and then train the sucker!
```

Shove some Mistral models in a clown car:

```yml
base_model: NousResearch/Hermes-2-Pro-Mistral-7B
gate_mode: hidden
dtype: bfloat16
experts:
  - source_model: NousResearch/Hermes-2-Pro-Mistral-7B
    positive_prompts:
      - "<|im_start|>user\nHello, who are you?<|im_end|>"
      - "<|im_start|>user\nI need help with"
  - source_model: BioMistral/BioMistral-7B-DARE
    positive_prompts:
      - "As a doctor of medicine,"
  - source_model: PocketDoc/Dans-AdventurousWinds-7b
    positive_prompts:
      - "[Genres: Science Fiction]\n[Tags: humor, old school, sci fi]"
      - "> get ye flask"
      - "[Mode: Interactive Storyteller]"
  - source_model: VAGOsolutions/SauerkrautLM-7b-HerO
    positive_prompts:
      - "<|im_start|>user\nWie geht es dir?<|im_end|>"
      - "Das ist ein Satz auf Deutsch."
```

## FAQ

### What does the "Your model has duplicated tensors but the --clone-tensors flag is not set" warning mean?

Answer from [Charles O. Goddard (cg123)](https://github.com/cg123)
(also see [this GitHub issue](https://github.com/arcee-ai/mergekit/issues/279#issuecomment-2081818104)):

> This is completely benign. This happens when a single tensor from a model is used in multiple places, like when doing sparse upcycling with the moe script or doing passthrough merges that repeat layers. Having `--clone-tensors` set can use slightly more memory, but having it unset will slow down saving and introduce small memory usage spikes in cases where this warning occurs. It's honestly a small enough difference that the warning could be removed entirely.
