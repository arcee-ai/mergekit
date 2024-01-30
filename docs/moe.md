# mergekit-moe

`mergekit-moe` is a script for combining Mistral or Llama models of the same size into Mixtral Mixture of Experts models. The script will combine the self-attention and layer normalization parameters from a "base" model with the MLP parameters from a set of "expert" models. `mergekit-moe` uses its own YML configuration syntax, which looks like so:

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

## Gate Modes

There are three methods for populating the MoE gates implemented.

### "hidden"

Uses the hidden state representations of the positive/negative prompts for MoE gate parameters. Best quality and most effective option; the default. Requires evaluating each prompt using the base model so you might not be able to use this on constrained hardware (depending on the model). You can use `--load-in-8bit` or `--load-in-4bit` to reduce VRAM usage.

### "cheap_embed"

Uses only the raw token embedding of the prompts, using the same gate parameters for every layer. Distinctly less effective than "hidden". Can be run on much, much lower end hardware.

### "random"

Randomly initializes the MoE gates. Good for if you are going to fine tune the model afterwards, or maybe if you want something a little unhinged? I won't judge.
