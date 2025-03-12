# Whisper Model Merging Guide

This guide explains how to effectively merge Whisper speech recognition models using MergeKit, with a focus on preserving multilingual capabilities while adding language specialization.

## Introduction to Whisper Models

Whisper is an encoder-decoder speech recognition model developed by OpenAI. It has strong multilingual capabilities and can be fine-tuned for specific languages or domains. When merging Whisper models, we need to consider:

1. The encoder-decoder architecture
2. Preserving multilingual capabilities
3. Handling LoRA fine-tuned models

## Basic Whisper Model Merging

The simplest way to merge Whisper models is using a weighted average approach:

```yaml
# examples/whisper-merge-multilingual.yml
models:
  - model: openai/whisper-large-v2  # Base multilingual model
    parameters:
      weight: 0.7
  - model: user/whisper-specialized-french  # French-specialized model
    parameters:
      weight: 0.3
merge_method: slerp
```

Run the merge with:

```bash
mergekit-yaml examples/whisper-merge-multilingual.yml --out_dir ./whisper-merged
```

## Merging LoRA-Fine-Tuned Whisper Models

If you have Whisper models fine-tuned with LoRA, you can merge them directly:

```yaml
# examples/whisper-lora-multilingual.yml
models:
  - model: openai/whisper-large-v2  # Base model
    lora: user/whisper-french-lora  # French LoRA adaptation
    parameters:
      weight: 0.6
  - model: openai/whisper-large-v2  # Same base model
    lora: user/whisper-german-lora  # German LoRA adaptation  
    parameters:
      weight: 0.4
merge_method: slerp
```

### Extracting LoRA from Fine-Tuned Whisper Models

If you have a fully fine-tuned Whisper model (not a LoRA adapter), you can extract a LoRA adapter from it:

```bash
mergekit-extract-whisper-lora \
  --model user/whisper-french-finetuned \
  --base-model openai/whisper-large-v2 \
  --out-path ./whisper-french-lora \
  --max-rank 32
```

You can also extract LoRA for only the encoder or decoder:

```bash
# Extract LoRA for encoder only (audio processing)
mergekit-extract-whisper-lora \
  --model user/whisper-french-finetuned \
  --base-model openai/whisper-large-v2 \
  --out-path ./whisper-french-encoder-lora \
  --encoder-only \
  --max-rank 32

# Extract LoRA for decoder only (text generation)
mergekit-extract-whisper-lora \
  --model user/whisper-french-finetuned \
  --base-model openai/whisper-large-v2 \
  --out-path ./whisper-french-decoder-lora \
  --decoder-only \
  --max-rank 32
```

## Advanced: Encoder-Decoder Weighted Merging

For more control over the merging process, you can use the encoder-decoder weighted merge method, which allows you to apply different weights to the encoder and decoder components:

```yaml
# examples/whisper-encoder-decoder-weighted.yml
models:
  - model: openai/whisper-large-v2  # Base model with good audio processing
    parameters:
      encoder_weight: 0.8  # Prioritize encoder (audio processing)
      decoder_weight: 0.3
  - model: user/whisper-specialized  # Specialized model with better text generation
    parameters:
      encoder_weight: 0.2
      decoder_weight: 0.7  # Prioritize decoder (text generation)
merge_method: encoder_decoder_weighted
```

This approach is particularly useful when:
- You want to preserve the audio processing capabilities of one model
- You want to use the text generation capabilities of another model
- You want to balance multilingual capabilities with language specialization

## Preserving Multilingual Capabilities

To preserve the multilingual capabilities of the base Whisper model while adding specialization for a specific language:

1. Use higher weights for the base model in the encoder (audio processing)
2. Use higher weights for the specialized model in the decoder (text generation)
3. Consider using task arithmetic to selectively apply specializations

## Testing Your Merged Model

After merging, you should evaluate your model on:

1. The target language(s) to ensure specialization is effective
2. Other languages to ensure multilingual capabilities are preserved
3. Various audio conditions to ensure robustness

## Conclusion

Merging Whisper models effectively requires understanding the encoder-decoder architecture and carefully balancing multilingual capabilities with language specialization. The tools and methods provided by MergeKit make it possible to create custom Whisper models that combine the strengths of multiple models. 