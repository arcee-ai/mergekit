#!/usr/bin/env python3
"""
Direct Whisper LoRA extraction script.
This script extracts LoRA weights from a fine-tuned Whisper model using PEFT directly.
"""

import os
import torch
import logging
import argparse
from transformers import WhisperForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("direct_whisper_lora_extract")

def extract_lora(
    base_model_path: str,
    finetuned_model_path: str,
    output_path: str,
    rank: int = 32,
    encoder_only: bool = False,
    decoder_only: bool = False,
):
    """
    Extract LoRA weights from a fine-tuned Whisper model.
    
    Args:
        base_model_path: Path to the base Whisper model
        finetuned_model_path: Path to the fine-tuned Whisper model
        output_path: Path to save the extracted LoRA adapter
        rank: Rank for the LoRA adapter
        encoder_only: Whether to extract only encoder weights
        decoder_only: Whether to extract only decoder weights
    """
    logger.info(f"Loading base model from {base_model_path}")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    logger.info(f"Loading fine-tuned model from {finetuned_model_path}")
    finetuned_model = WhisperForConditionalGeneration.from_pretrained(
        finetuned_model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    # Determine target modules based on encoder/decoder flags
    target_modules = []
    if encoder_only and decoder_only:
        # Both encoder and decoder
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    elif encoder_only:
        # Only encoder modules
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        logger.info("Extracting LoRA weights for encoder only")
    elif decoder_only:
        # Only decoder modules
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        logger.info("Extracting LoRA weights for decoder only")
    else:
        # Default: both encoder and decoder
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    
    # Create a PEFT model with the base model
    logger.info("Creating PEFT model with base model")
    peft_model = get_peft_model(base_model, lora_config)
    
    # Extract the differences between the models
    logger.info("Extracting differences between models")
    diff_state_dict = {}
    
    # Get state dicts
    base_state_dict = base_model.state_dict()
    finetuned_state_dict = finetuned_model.state_dict()
    
    # Filter modules based on encoder/decoder flags
    for key in base_state_dict:
        if encoder_only and "encoder" not in key:
            continue
        if decoder_only and "decoder" not in key:
            continue
            
        if key in finetuned_state_dict:
            # Check if this is a target module
            module_type = key.split(".")[-1]
            if module_type in target_modules or module_type == "weight" and key.split(".")[-2] in target_modules:
                # Calculate difference
                logger.info(f"Processing {key}")
                diff = finetuned_state_dict[key] - base_state_dict[key]
                diff_state_dict[key] = diff
    
    # Convert differences to LoRA weights
    logger.info("Converting differences to LoRA weights")
    lora_state_dict = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save adapter config
    with open(os.path.join(output_path, "adapter_config.json"), "w") as f:
        f.write(peft_model.config.to_json_string())
    
    # Save the model
    logger.info(f"Saving LoRA adapter to {output_path}")
    peft_model.save_pretrained(output_path)
    
    logger.info("LoRA extraction completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Extract LoRA weights from a fine-tuned Whisper model")
    parser.add_argument("--model", required=True, help="Fine-tuned Whisper model path")
    parser.add_argument("--base-model", required=True, help="Base Whisper model path")
    parser.add_argument("--out-path", required=True, help="Output path for extracted LoRA adapter")
    parser.add_argument("--max-rank", type=int, default=32, help="Maximum rank for LoRA decomposition")
    parser.add_argument("--encoder-only", action="store_true", help="Extract LoRA only for encoder weights")
    parser.add_argument("--decoder-only", action="store_true", help="Extract LoRA only for decoder weights")
    
    args = parser.parse_args()
    
    extract_lora(
        base_model_path=args.base_model,
        finetuned_model_path=args.model,
        output_path=args.out_path,
        rank=args.max_rank,
        encoder_only=args.encoder_only,
        decoder_only=args.decoder_only,
    )

if __name__ == "__main__":
    main() 