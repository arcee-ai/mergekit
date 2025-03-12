#!/usr/bin/env python3
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import click
import logging
from typing import List, Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration

from mergekit.common import ModelReference
from mergekit.options import MergeOptions, add_merge_options
from mergekit.scripts.extract_lora import plan_extraction, make_config_dict, TaskVectorDecompositionTask
from mergekit.multigpu_executor import MultiGPUExecutor
from mergekit.graph import Executor


@click.command("mergekit-extract-whisper-lora", cls=click.core.Command)
@click.option(
    "--model",
    required=True,
    help="Fine-tuned Whisper model path",
)
@click.option(
    "--base-model",
    required=True,
    help="Base Whisper model path",
)
@click.option(
    "--out-path",
    required=True,
    help="Output path for extracted LoRA adapter",
)
@click.option(
    "--max-rank",
    type=int,
    default=128,
    help="Maximum rank for LoRA decomposition",
)
@click.option(
    "--encoder-only/--no-encoder-only",
    is_flag=True,
    default=False,
    help="Extract LoRA only for encoder weights",
)
@click.option(
    "--decoder-only/--no-decoder-only",
    is_flag=True,
    default=False,
    help="Extract LoRA only for decoder weights",
)
@click.option(
    "--distribute-scale/--no-distribute-scale",
    is_flag=True,
    default=True,
    help="Distribute scale between A and B matrices",
)
@click.option(
    "--sv-epsilon",
    type=float,
    default=0,
    help="Threshold for singular values to discard",
    show_default=True,
)
@add_merge_options
def main(
    base_model: str,
    model: str,
    out_path: str,
    max_rank: int,
    encoder_only: bool,
    decoder_only: bool,
    distribute_scale: bool,
    sv_epsilon: float,
    merge_options: MergeOptions,
):
    """Extract a LoRA adapter from a fine-tuned Whisper model.
    
    This tool allows extracting LoRA weights from a fine-tuned Whisper model,
    with options to focus on either the encoder (audio processing) or decoder
    (text generation) components.
    """
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("extract_whisper_lora")
    
    # Apply global options
    merge_options.apply_global_options()
    
    # Validate models can be loaded before proceeding
    logger.info(f"Validating models: {model} and {base_model}")
    try:
        # Try to load models to verify they're accessible
        logger.info(f"Loading fine-tuned model: {model}")
        test_model = WhisperForConditionalGeneration.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        logger.info(f"Successfully loaded fine-tuned model with shape: {test_model.get_input_embeddings().weight.shape}")
        
        logger.info(f"Loading base model: {base_model}")
        test_base = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        logger.info(f"Successfully loaded base model with shape: {test_base.get_input_embeddings().weight.shape}")
        
        # Free memory
        del test_model
        del test_base
        torch.cuda.empty_cache()
        
        logger.info("Model validation successful")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise ValueError(f"Could not load one or both models: {e}")
    
    # Determine include/exclude patterns based on flags
    include_regexes = []
    exclude_regexes = []
    
    if encoder_only and decoder_only:
        # If both are set, we don't need to filter
        logging.info("Both encoder-only and decoder-only flags set, extracting all weights")
    elif encoder_only:
        logging.info("Extracting LoRA weights for encoder only")
        include_regexes.append(".*encoder.*")
        exclude_regexes.append(".*decoder.*")
    elif decoder_only:
        logging.info("Extracting LoRA weights for decoder only")
        include_regexes.append(".*decoder.*")
        exclude_regexes.append(".*encoder.*")
    
    # Use the standard extraction logic with Whisper-specific filtering
    base_model_ref = ModelReference.model_validate(base_model)
    model_ref = ModelReference.model_validate(model)
    
    plan_result = plan_extraction(
        base_model_ref=base_model_ref.merged(
            cache_dir=merge_options.lora_merge_cache,
            trust_remote_code=merge_options.trust_remote_code,
            lora_merge_dtype=merge_options.lora_merge_dtype,
        ),
        model_ref=model_ref.merged(
            cache_dir=merge_options.lora_merge_cache,
            trust_remote_code=merge_options.trust_remote_code,
            lora_merge_dtype=merge_options.lora_merge_dtype,
        ),
        modules_to_save=[],  # No modules to save at full rank by default
        out_path=out_path,
        options=merge_options,
        max_rank=max_rank,
        distribute_scale=distribute_scale,
        embed_lora=False,  # Don't extract LoRA weights for embeddings by default
        include_regexes=include_regexes,
        exclude_regexes=exclude_regexes,
        sv_epsilon=sv_epsilon,
        skip_undecomposable=False,
    )

    tasks = plan_result.tasks
    if merge_options.multi_gpu:
        executor = MultiGPUExecutor(
            tasks, storage_device="cpu" if not merge_options.low_cpu_memory else None
        )
    else:
        executor = Executor(
            tasks,
            math_device="cuda" if merge_options.cuda else "cpu",
            storage_device="cuda" if merge_options.low_cpu_memory else "cpu",
        )

    module_real_ranks = {}
    for task, result in executor.run():
        if isinstance(task, TaskVectorDecompositionTask):
            module_real_ranks[task.weight_info.name.removesuffix(".weight")] = result[
                0
            ].shape[0]

    real_max_rank = max(module_real_ranks.values()) if module_real_ranks else max_rank
    config_dict = make_config_dict(
        base_ref=base_model_ref,
        max_rank=real_max_rank,
        modules_to_save=[],
        target_modules=list(
            set(key.split(".")[-1] for key in module_real_ranks.keys())
        ),
        module_ranks=module_real_ranks,
    )
    
    logging.info(f"Extracted LoRA adapter saved to {out_path}")
    logging.info(f"Maximum rank used: {real_max_rank}")


if __name__ == "__main__":
    main() 