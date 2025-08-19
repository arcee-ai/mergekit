# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import importlib
import importlib.resources
import logging
import os
import shutil
from collections import Counter
from typing import List, Optional, Tuple

import tqdm
import transformers

from collections import Counter, defaultdict
from typing import Optional, Dict, Any, DefaultDict, List
from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import re
import gc
import os
import shutil
import torch
import json
import torch.nn as nn
from torch.nn import Parameter
from transformers import AutoModelForCausalLM
from torch.optim import AdamW
import deepspeed
from deepspeed.utils import groups
import torch.distributed as dist
from deepspeed import comm as ds_comm
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from deepspeed.runtime.zero.partition_parameters import PartitionedParamStatus

from llmtailor._data import chat_templates
from llmtailor.architecture import ModelArchitecture, get_architecture_info
from llmtailor.card import generate_card
from llmtailor.common import ModelReference, set_config_value
from llmtailor.config import MergeConfiguration
from llmtailor.graph import Executor
from llmtailor.io.tasks import LoaderCache
from llmtailor.multigpu_executor import MultiGPUExecutor
from llmtailor.options import MergeOptions
from llmtailor.plan import MergePlanner
from llmtailor.tokenizer import TokenizerInfo

LOG = logging.getLogger(__name__)

#########################################
# Optimizer checkpoint keys
#########################################
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
DS_CONFIG = "ds_config"
DS_VERSION = "ds_version"
FP32_FLAT_GROUPS = 'fp32_flat_groups'
ZERO_STAGE = 'zero_stage'
LOSS_SCALER = 'loss_scaler'
DYNAMIC_LOSS_SCALE = 'dynamic_loss_scale'
PARTITION_COUNT = 'partition_count'
OVERFLOW = 'overflow'

BASE_OPTIMIZER_STATE = 'base_optimizer_state'
BASE_OPTIMIZER_STATE_STEP = 'base_optimizer_state_step'
SINGLE_PARTITION_OF_FP32_GROUPS = "single_partition_of_fp32_groups"
GROUP_PADDINGS = 'group_paddings'
CLIP_GRAD = 'clip_grad'
FP32_WEIGHT_KEY = "fp32"

#########################################
# Module checkpoint keys
#########################################
PARAM = 'param'
PARAM_SHAPES = 'param_shapes'
BUFFER_NAMES = 'buffer_names'
FROZEN_PARAM_SHAPES = 'frozen_param_shapes'
FROZEN_PARAM_FRAGMENTS = 'frozen_param_fragments'

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
PARAM_GROUPS = 'param_groups'
tqdm.tqdm = tqdm 

_SHARD_CACHE: dict[str, dict] = {} 

def _load_one_optim_shard(path: str):
    """
    Loaders with memory caching: If the same path has already been read, the cache is reused to avoid duplicate I/O.
    Process-local caching—also available in ProcessPoolExecutor workers.
    """
    if path not in _SHARD_CACHE:
        _SHARD_CACHE[path] = TorchCheckpointEngine().load(path, map_location='cpu')
    return _SHARD_CACHE[path][OPTIMIZER_STATE_DICT]

def run_merge(
    merge_config: MergeConfiguration,
    out_path: str,
    options: MergeOptions,
    config_source: Optional[str] = None,
):
    if options.random_seed is not None:
        transformers.trainer_utils.set_seed(options.random_seed)

    if not merge_config.models and not merge_config.slices and not merge_config.modules:
        raise RuntimeError("No output requested")

    arch_info = get_architecture_info(merge_config, options)

    # initialize loader cache and set options
    loader_cache = LoaderCache()
    loader_cache.setup(options=options)

    # create config for output model
    cfg_out = _model_out_config(
        merge_config, arch_info, trust_remote_code=options.trust_remote_code
    )

    # warm up loader cache
    for model in (
        pbar := tqdm.tqdm(
            merge_config.referenced_models(),
            desc="Warmup loader cache",
            disable=options.quiet,
        )
    ):
        loader_cache.get(model)
    del pbar

    LOG.info("Planning operations")


    #########################################    #########################################
    # Merge optimizer states
    #########################################   

    num_gpus = options.num_gpus

    # Load optimizer states for each model
    optimizer_states = {}
    accumulated_range_end = 0
    new_ranges = {}

    def _get_ds_config(load_dir, tag):
        ds_config = {}
        ds_version = {}

        zero_ckpt_names = _get_all_zero_checkpoint_names(load_dir, tag)
        zero_sd_list = []
        for i, ckpt_name in enumerate(zero_ckpt_names):
            _state = TorchCheckpointEngine().load(
                ckpt_name,
                map_location='cpu',
            )

            zero_sd_list.append(_state)
        ds_config = zero_sd_list[0][DS_CONFIG]
        ds_version = zero_sd_list[0][DS_VERSION]
        return ds_config, ds_version

    def _get_all_zero_checkpoint_state_dicts(zero_ckpt_names):
        """Load multiple ZeRO-3 optimizer shards in parallel using multiple processes."""
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            zero_optimizer_sd = list(pool.map(_load_one_optim_shard, zero_ckpt_names))

        return zero_optimizer_sd

    def _get_all_zero_checkpoints(load_dir, tag):
        zero_ckpt_names = _get_all_zero_checkpoint_names(load_dir, tag)
        if zero_ckpt_names is not None:
            return _get_all_zero_checkpoint_state_dicts(zero_ckpt_names)

        return None
    
    def _get_all_zero_checkpoint_names(load_dir, tag):
        mp_rank = 0
        zero_ckpt_names = _get_mp_rank_zero_checkpoint_names(load_dir=load_dir,
                                                                  tag=tag,
                                                                  mp_rank=mp_rank,
                                                                  dp_world_size=num_gpus)
        for i, ckpt_name in enumerate(zero_ckpt_names):
            if not os.path.exists(ckpt_name):
                # transparently handle the old file pattern for optim_states
                if "optim_states.pt" in ckpt_name:
                    ckpt_name_try = ckpt_name.replace("_optim_states.pt", "optim_states.pt")
                    if os.path.exists(ckpt_name_try):
                        zero_ckpt_names[i] = ckpt_name_try
                        continue
                # Try previous step (off-by-one)
                try:
                    step_num = int(tag.replace("global_step", ""))
                    prev_tag = f"global_step{step_num-1}"
                    prev_ckpt_name = ckpt_name.replace(tag, prev_tag)
                    if os.path.exists(prev_ckpt_name):
                        zero_ckpt_names[i] = prev_ckpt_name
                        continue
                except Exception:
                    pass

        return zero_ckpt_names
    
    def _get_mp_rank_zero_checkpoint_names(load_dir, tag, mp_rank, dp_world_size=num_gpus):
        zero_ckpt_names = []
        for dp_rank in range(dp_world_size):
            ckpt_name = _get_rank_zero_ckpt_name(checkpoints_path=load_dir,
                                                      tag=tag,
                                                      mp_rank=mp_rank,
                                                      dp_rank=dp_rank)
            zero_ckpt_names.append(ckpt_name)

        return zero_ckpt_names
    
    def _get_rank_zero_ckpt_name(checkpoints_path, tag, mp_rank, dp_rank):
        file_prefix = _get_zero_ckpt_prefix(dp_rank)
        zero_ckpt_name = os.path.join(
            checkpoints_path,
            str(tag),
            f"{file_prefix}_mp_rank_{mp_rank:02d}_optim_states.pt",
        )
        return zero_ckpt_name
    
    def _get_zero_ckpt_prefix(dp_rank):
        return f'{"bf16_"}zero_pp_rank_{dp_rank}'
    
    def get_tag_from_path(path, flag = True):
        checkpoint_name = os.path.basename(path)
        if checkpoint_name.startswith("checkpoint-") and flag:
            step_num = checkpoint_name.split("-")[1]
            return f"global_step{step_num}"
        elif flag == False:
            step_num = int(checkpoint_name.split("-")[1])-1
            return f"global_step{str(step_num)}"
        else:
            return None
    
    def get_parameter_names(model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result
    
    # --- Parallelized copy_files function using ThreadPoolExecutor ---
    from concurrent.futures import ThreadPoolExecutor
    def _copy_one(src: Path, dst: Path, *, use_link: bool):
        try:
            if use_link:
                os.link(src, dst)  # O(1) reflink/hardlink
            else:
                shutil.copy2(src, dst, follow_symlinks=False)
        except FileExistsError:
            pass
        except FileNotFoundError:
            print(f"{src} not found")

    def copy_files(src_folder, dest_folder, file_names, max_workers=8):
        """
        Parallelized version of copy_files. Each file is copied in a separate thread.
        Uses hard links if possible (same filesystem), otherwise falls back to copy2.
        """
        os.makedirs(dest_folder, exist_ok=True)
        src_folder = Path(src_folder)
        dest_folder = Path(dest_folder)
        same_fs = src_folder.stat().st_dev == dest_folder.stat().st_dev

        def copy_task(fname):
            _copy_one(src_folder / fname, dest_folder / fname, use_link=same_fs)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(copy_task, file_names))

    def create_scheduler(old_path, layer_ranges, new_path, num_hidden_layers):
        name = 'scheduler.pt'
        base_scheduler = torch.load(os.path.join(old_path, name))
        for model_path in layer_ranges.keys():
            src_file = os.path.join(model_path, name)
            scheduler = torch.load(src_file)
            for layer_range in layer_ranges[model_path]:
                new_init_layer = layer_range[0][0]
                new_end_layer = layer_range[0][1]
                old_init_layer = layer_range[1][0]
                old_end_layer = layer_range[1][1]
                base_scheduler['_last_lr'][new_init_layer+1:new_end_layer+1] = scheduler['_last_lr'][old_init_layer+1:old_end_layer+1]
                base_scheduler['_last_lr'][new_init_layer+num_hidden_layers+2:new_end_layer+num_hidden_layers+2] = scheduler['_last_lr'][old_init_layer+num_hidden_layers+2:old_end_layer+num_hidden_layers+2]

        torch.save(base_scheduler, os.path.join(new_path, name))
                    

    # copy the files from the old path to the new path
    rng_file_names = [f"rng_state_{i}.pth" for i in range(num_gpus)]
    model_state_file_names = [f"zero_pp_rank_{i}_mp_rank_00_model_states.pt" for i in range(num_gpus)]
    
    # file_names_input is the file names that the user want to copy from the old path to the new path
    file_names_input = ['generation_config.json', 'latest', 'trainer_state.json', 'training_args.bin', 'zero_to_fp32.py'] + rng_file_names
    # split and remove extra spaces
    file_names = [name.strip() for name in file_names_input if name.strip()]

    # Create a dictionary mapping load directories to layer ranges
    model_layer_map = []
    for slice_def in merge_config.slices:
        for source in slice_def.sources:
            model_path = source.model.model.path
            if model_path not in model_layer_map:
                model_layer_map.append({model_path: source.layer_range})
    # Create a dictionary to store layer ranges for each model path
    old_ranges = {}
    for map_item in model_layer_map:
        model_path = list(map_item.keys())[0]
        layer_range = list(map_item.values())[0]
        if model_path not in old_ranges:
            old_ranges[model_path] = [layer_range]
        else:
            old_ranges[model_path].append(layer_range)

    model_list = [list(map.keys())[0] for map in model_layer_map]
    # Get unique model paths
    model_list = list(dict.fromkeys([list(map.keys())[0] for map in model_layer_map]))
    model_list = sorted(model_list, key=lambda x: int(x.split('-')[-1]))
    
    tag_list = [get_tag_from_path(model_path) for model_path in model_list]
    tag_list = sorted(tag_list, key=lambda tag: int(tag.replace("global_step", "")))
    tag_last = tag_list[-1]

    new_optimizer_path = os.path.join(out_path, str(tag_last))

    old_path = model_list[-1]
    old_optimizer_path = os.path.join(old_path, str(tag_last))

    with open(os.path.join(old_path, 'config.json'), 'r', encoding='utf-8') as file:
        data = json.load(file)

    # num_hidden_layers is the total layers of the pre-trained base model
    num_hidden_layers = data.get("num_hidden_layers")
    model_type = data.get("model_type")
    tie_word_embeddings = data.get("tie_word_embeddings")

    # call the copy function
    copy_files(old_path, out_path, file_names_input)
    if os.path.exists(old_optimizer_path):
        copy_files(old_optimizer_path, new_optimizer_path, model_state_file_names)
    else:
        tag_list = [get_tag_from_path(model_path, flag = False) for model_path in model_list]
        tag_list = sorted(tag_list, key=lambda tag: int(tag.replace("global_step", "")))
        tag_last = tag_list[-1]
        old_optimizer_path = os.path.join(old_path, str(tag_last))
        new_optimizer_path = os.path.join(out_path, str(tag_last))
        copy_files(old_optimizer_path, new_optimizer_path, model_state_file_names)

    # Calculate new ranges based on accumulated values
    for idx, map in enumerate(model_layer_map):
        model_path = list(map.keys())[0]
        original_ranges = list(map.values())[0]
        tag = get_tag_from_path(model_path)
        # Calculate new ranges based on accumulated values
        if new_ranges != []:
            start_range = accumulated_range_end
            end_range = accumulated_range_end + (original_ranges[1] - original_ranges[0])
            if model_path not in new_ranges:
                new_ranges[model_path] = [[(start_range, end_range), original_ranges]]
            else:
                new_ranges[model_path].append([(start_range, end_range), original_ranges])
            accumulated_range_end = end_range
        else:
            new_ranges[model_path] = [[(0, original_ranges[1] - original_ranges[0]), original_ranges]]
            accumulated_range_end = original_ranges[1] - original_ranges[0]
        try:    
            zero_sd_list = _get_all_zero_checkpoints(load_dir=model_path, tag=tag)
            optimizer_states[model_path] = {
                'states': zero_sd_list,
                'ranges': new_ranges[model_path]
            }
        except Exception as e:
            tag = get_tag_from_path(model_path, flag = False)
            zero_sd_list = _get_all_zero_checkpoints(load_dir=model_path, tag=tag)
            optimizer_states[model_path] = {
                'states': zero_sd_list,
                'ranges': new_ranges[model_path]
            }
    print(new_ranges)

    # create the scheduler
    create_scheduler(old_path, new_ranges, out_path, num_hidden_layers)

    # Find the model path with the highest checkpoint number
    newest_model_path = None
    highest_checkpoint = -1
    
    for model_path in new_ranges.keys():
        # Extract the checkpoint number from the path
        checkpoint_part = model_path.split('/')[-1]
        if checkpoint_part.startswith('checkpoint-'):
            try:
                checkpoint_num = int(checkpoint_part.split('-')[-1])
                if checkpoint_num > highest_checkpoint:
                    highest_checkpoint = checkpoint_num
                    newest_model_path = model_path
            except ValueError:
                # If conversion fails, try other formats
                pass
    
    # If no valid checkpoint found, fallback to the last model in the list
    if newest_model_path is None:
        newest_model_path = list(new_ranges.keys())[-1]
    newest_model_data = optimizer_states[newest_model_path]

    # Process optimizer states for each GPU
    if model_type != "mistral" and tie_word_embeddings == True:
        print("model_type is not mistral and tie_word_embeddings is True")
        for i in range(num_gpus):
            num_groups = 2 * num_hidden_layers + 2

            new_state_dict = {}
            new_fp32_flat_groups = [None] * num_groups
            new_param_groups_value = [None] * num_groups
            
            zero_stage = newest_model_data['states'][i][ZERO_STAGE]
            loss_scaler = newest_model_data['states'][i][LOSS_SCALER]
            dynamic_loss_scale = newest_model_data['states'][i][DYNAMIC_LOSS_SCALE]
            overflow = newest_model_data['states'][i][OVERFLOW]
            partition_count = newest_model_data['states'][i][PARTITION_COUNT]
            
            # Process optimizer states for each GPU
            for model_path, model_data in optimizer_states.items():
                tag = get_tag_from_path(model_path)
                loaded_state_dict = model_data['states'][i][OPTIMIZER_STATE_DICT]
                loaded_fp32_flat_groups = model_data['states'][i][FP32_FLAT_GROUPS]
                loaded_param_groups = model_data['states'][i][OPTIMIZER_STATE_DICT][PARAM_GROUPS]

                # Process each layer range for this model
                for layer_range in model_data['ranges']:
                    new_init_layer = layer_range[0][0]
                    new_end_layer = layer_range[0][1]
                    old_init_layer = layer_range[1][0]
                    old_end_layer = layer_range[1][1]
                    # print(new_init_layer, new_end_layer, old_init_layer, old_end_layer)

                    # add the token_embedded and norm_layer
                    if old_init_layer == old_end_layer and old_init_layer == 0:
                        new_state_dict[0] = loaded_state_dict['state'][0]
                        new_param_groups_value[0] = loaded_param_groups[0]
                        new_fp32_flat_groups[0] = loaded_fp32_flat_groups[0]

                    if old_init_layer == old_end_layer and old_init_layer == num_hidden_layers+1:
                        new_state_dict[num_hidden_layers+1] = loaded_state_dict['state'][num_hidden_layers+1]
                        new_param_groups_value[num_hidden_layers+1] = [g for g in loaded_param_groups if g["params"][0] == num_hidden_layers+1][0]
                        new_fp32_flat_groups[num_hidden_layers+1] = loaded_fp32_flat_groups[loaded_param_groups.index([g for g in loaded_param_groups if g["params"][0] == num_hidden_layers+1][0])]
                    
                    if old_init_layer != old_end_layer:
                        for k in range(new_init_layer, new_end_layer):
                            # get the source index by the relative position
                            src_idx1 = k - new_init_layer + old_init_layer + 1
                            src_idx2 = k - new_init_layer + old_init_layer + num_hidden_layers + 2
                            # print(loaded_state_dict['state'])

                            # Add state for the current layer and its corresponding k+18 layer
                            new_state_dict[k+1] = loaded_state_dict['state'][src_idx1]
                            new_state_dict[k+num_hidden_layers+2] = loaded_state_dict['state'][src_idx2]

                            new_param_groups_value[k+1] = [g for g in loaded_param_groups if g["params"][0] == src_idx1][0]
                            new_param_groups_value[k+num_hidden_layers+2] = [g for g in loaded_param_groups if g["params"][0] == src_idx2][0]

                            index1 = loaded_param_groups.index([g for g in loaded_param_groups if g["params"][0] == src_idx1][0])
                            index2 = loaded_param_groups.index([g for g in loaded_param_groups if g["params"][0] == src_idx2][0])

                            new_fp32_flat_groups[k+1] = loaded_fp32_flat_groups[index1]
                            new_fp32_flat_groups[k+num_hidden_layers+2] = loaded_fp32_flat_groups[index2]

            # sort the new_state_dict by the key
            new_state_dict = dict(sorted(new_state_dict.items()))

            # Combine state dict with param groups
            new_optimizer = {
                'state': new_state_dict,
                'param_groups': new_param_groups_value
            }
            
            inside_state = {
                OPTIMIZER_STATE_DICT: new_optimizer,
                FP32_FLAT_GROUPS: new_fp32_flat_groups,
                ZERO_STAGE: zero_stage,
                LOSS_SCALER: loss_scaler,
                DYNAMIC_LOSS_SCALE: dynamic_loss_scale,
                OVERFLOW: overflow,
                PARTITION_COUNT: partition_count
            }
            
            ds_config, ds_version = _get_ds_config(model_path, tag)

            save_state = {'optimizer_state_dict': inside_state, DS_CONFIG: ds_config, DS_VERSION: ds_version}
            checkpoint_name = f"bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt"
            TorchCheckpointEngine().save(save_state, os.path.join(new_optimizer_path, checkpoint_name))

    elif model_type == "mistral":
        print("mistral model is not supported yet")

    elif tie_word_embeddings == False:
        print("tie_word_embeddings is False")
        for i in range(num_gpus):
            num_groups = 2 * num_hidden_layers + 3

            new_state_dict = {}
            new_fp32_flat_groups = [None] * num_groups
            new_param_groups_value = [None] * num_groups
            
            zero_stage = newest_model_data['states'][i][ZERO_STAGE]
            loss_scaler = newest_model_data['states'][i][LOSS_SCALER]
            dynamic_loss_scale = newest_model_data['states'][i][DYNAMIC_LOSS_SCALE]
            overflow = newest_model_data['states'][i][OVERFLOW]
            partition_count = newest_model_data['states'][i][PARTITION_COUNT]
            
            # Process optimizer states for each GPU
            for model_path, model_data in optimizer_states.items():
                tag = get_tag_from_path(model_path)
                loaded_state_dict = model_data['states'][i][OPTIMIZER_STATE_DICT]
                loaded_fp32_flat_groups = model_data['states'][i][FP32_FLAT_GROUPS]
                loaded_param_groups = model_data['states'][i][OPTIMIZER_STATE_DICT][PARAM_GROUPS]

                # Process each layer range for this model
                for layer_range in model_data['ranges']:
                    new_init_layer = layer_range[0][0]
                    new_end_layer = layer_range[0][1]
                    old_init_layer = layer_range[1][0]
                    old_end_layer = layer_range[1][1]
                    # print(new_init_layer, new_end_layer, old_init_layer, old_end_layer)

                    # add the token_embedded and norm_layer
                    if old_init_layer == old_end_layer and old_init_layer == 0:
                        new_state_dict[0] = loaded_state_dict['state'][0]
                        new_param_groups_value[0] = loaded_param_groups[0]
                        new_fp32_flat_groups[0] = loaded_fp32_flat_groups[0]

                    if old_init_layer == old_end_layer and old_init_layer == num_hidden_layers+1:
                        new_state_dict[num_hidden_layers+1] = loaded_state_dict['state'][num_hidden_layers+1]
                        new_param_groups_value[num_hidden_layers+1] = [g for g in loaded_param_groups if g["params"][0] == num_hidden_layers+1][0]
                        new_fp32_flat_groups[num_hidden_layers+1] = loaded_fp32_flat_groups[loaded_param_groups.index([g for g in loaded_param_groups if g["params"][0] == num_hidden_layers+1][0])]

                    if old_init_layer == old_end_layer and old_init_layer == num_hidden_layers+2:
                        new_state_dict[num_hidden_layers+2] = loaded_state_dict['state'][num_hidden_layers+2]
                        new_param_groups_value[num_hidden_layers+2] = [g for g in loaded_param_groups if g["params"][0] == num_hidden_layers+2][0]
                        new_fp32_flat_groups[num_hidden_layers+2] = loaded_fp32_flat_groups[loaded_param_groups.index([g for g in loaded_param_groups if g["params"][0] == num_hidden_layers+2][0])]
                    
                    if old_init_layer != old_end_layer:
                        for k in range(new_init_layer, new_end_layer):
                            # get the source index by the relative position
                            src_idx1 = k - new_init_layer + old_init_layer + 1
                            src_idx2 = k - new_init_layer + old_init_layer + num_hidden_layers + 3
                            # print(loaded_state_dict['state'])

                            # Add state for the current layer and its corresponding k+18 layer
                            new_state_dict[k+1] = loaded_state_dict['state'][src_idx1]
                            new_state_dict[k+num_hidden_layers+3] = loaded_state_dict['state'][src_idx2]

                            new_param_groups_value[k+1] = [g for g in loaded_param_groups if g["params"][0] == src_idx1][0]
                            new_param_groups_value[k+num_hidden_layers+3] = [g for g in loaded_param_groups if g["params"][0] == src_idx2][0]

                            index1 = loaded_param_groups.index([g for g in loaded_param_groups if g["params"][0] == src_idx1][0])
                            index2 = loaded_param_groups.index([g for g in loaded_param_groups if g["params"][0] == src_idx2][0])

                            new_fp32_flat_groups[k+1] = loaded_fp32_flat_groups[index1]
                            new_fp32_flat_groups[k+num_hidden_layers+3] = loaded_fp32_flat_groups[index2]

            # sort the new_state_dict by the key
            new_state_dict = dict(sorted(new_state_dict.items()))

            # Combine state dict with param groups
            new_optimizer = {
                'state': new_state_dict,
                'param_groups': new_param_groups_value
            }
            
            inside_state = {
                OPTIMIZER_STATE_DICT: new_optimizer,
                FP32_FLAT_GROUPS: new_fp32_flat_groups,
                ZERO_STAGE: zero_stage,
                LOSS_SCALER: loss_scaler,
                DYNAMIC_LOSS_SCALE: dynamic_loss_scale,
                OVERFLOW: overflow,
                PARTITION_COUNT: partition_count
            }
            
            ds_config, ds_version = _get_ds_config(model_path, tag)

            save_state = {'optimizer_state_dict': inside_state, DS_CONFIG: ds_config, DS_VERSION: ds_version}
            checkpoint_name = f"bf16_zero_pp_rank_{i}_mp_rank_00_optim_states.pt"
            TorchCheckpointEngine().save(save_state, os.path.join(new_optimizer_path, checkpoint_name))

    targets = MergePlanner(
        merge_config,
        arch_info,
        options=options,
        out_model_config=cfg_out,
    ).plan_to_disk(out_path=out_path)

    if options.multi_gpu:
        exec = MultiGPUExecutor(
            targets=targets,
            storage_device=None if options.low_cpu_memory else "cpu",
        )
    else:
        exec = Executor(
            targets=targets,
            math_device=options.device,
            storage_device=options.device if options.low_cpu_memory else "cpu",
        )

    tokenizer = None
    for _task, value in exec.run(quiet=options.quiet):
        if isinstance(value, TokenizerInfo):
            tokenizer = value.tokenizer

    if tokenizer:
        pad_to_multiple_of = None
        if merge_config.tokenizer and merge_config.tokenizer.pad_to_multiple_of:
            pad_to_multiple_of = merge_config.tokenizer.pad_to_multiple_of
        _update_config_vocab(
            cfg_out, arch_info, tokenizer, pad_to_multiple_of=pad_to_multiple_of
        )

    LOG.info("Saving config")
    cfg_out.save_pretrained(out_path)

    if options.write_model_card:
        if not config_source:
            config_source = merge_config.to_yaml()

        card_md = generate_card(
            config=merge_config,
            config_yaml=config_source,
            name=os.path.basename(out_path),
        )
        with open(os.path.join(out_path, "README.md"), "w", encoding="utf-8") as fp:
            fp.write(card_md)

        with open(
            os.path.join(out_path, "mergekit_config.yml"), "w", encoding="utf-8"
        ) as fp:
            fp.write(config_source)

    if tokenizer is not None:
        LOG.info("Saving tokenizer")
        _set_chat_template(tokenizer, merge_config)
        tokenizer.save_pretrained(out_path, safe_serialization=True)
    else:
        if options.copy_tokenizer:
            try:
                _copy_tokenizer(merge_config, out_path, options=options)
            except Exception as e:
                LOG.error(
                    "Failed to copy tokenizer. The merge was still successful, just copy it from somewhere else.",
                    exc_info=e,
                )
        elif merge_config.chat_template:
            LOG.warning(
                "Chat template specified but no tokenizer found. Chat template will not be saved."
            )

    _copy_tagalong_files(
        merge_config,
        out_path,
        files=arch_info.tagalong_files or [],
        options=options,
    )

    if getattr(arch_info, "post_fill_parameters", False):
        from mergekit.scripts.fill_missing_params import copy_and_fill_missing_params

        logging.info(
            f"Filling missing parameters from base model {arch_info.post_fill_parameters} into new directory"
        )
        copy_and_fill_missing_params(
            base_model_repo_id=arch_info.post_fill_parameters,
            sub_model_dir=out_path,
        )
        logging.info("Deleting initial merge directory: " + out_path)
        shutil.rmtree(out_path)


def _set_chat_template(
    tokenizer: transformers.PreTrainedTokenizerBase,
    merge_config: MergeConfiguration,
    trust_remote_code: bool = False,
):
    chat_template = merge_config.chat_template
    if not chat_template:
        return

    if chat_template == "auto":
        # see if there is a plurality chat template among the input models
        model_templates = []
        for model in merge_config.referenced_models():
            try:
                tok = transformers.AutoTokenizer.from_pretrained(
                    model.model.path,
                    revision=model.model.revision,
                    trust_remote_code=trust_remote_code,
                )
                template = tok.chat_template
                if isinstance(template, dict):
                    template = template.get("default", None)
                if template:
                    model_templates.append(template.strip())
            except Exception as e:
                LOG.warning(f"Unable to load tokenizer for {model}", exc_info=e)

        if not model_templates:
            return

        chat_template = Counter(model_templates).most_common(1)[0][0]
        LOG.info(f"Auto-selected chat template: {chat_template}")

    elif (
        t := importlib.resources.files(chat_templates).joinpath(
            chat_template + ".jinja"
        )
    ).is_file():
        chat_template = t.read_text()

    elif len(chat_template) < 20 or "{" not in chat_template:
        raise RuntimeError(f"Invalid chat template: {chat_template}")

    tokenizer.chat_template = chat_template


def _get_donor_model(
    merge_config: MergeConfiguration,
    options: MergeOptions,
) -> Tuple[ModelReference, str]:
    donor_model = merge_config.base_model or (merge_config.referenced_models()[0])
    donor_local_path = donor_model.merged(
        cache_dir=options.lora_merge_cache,
        trust_remote_code=options.trust_remote_code,
        lora_merge_dtype=options.lora_merge_dtype,
    ).local_path(cache_dir=options.transformers_cache)
    if not donor_local_path:
        raise RuntimeError(f"Unable to find local path for {donor_model}")
    return donor_model, donor_local_path


def _copy_tagalong_files(
    merge_config: MergeConfiguration,
    out_path: str,
    files: List[str],
    options: MergeOptions,
):
    donor_model, donor_local_path = _get_donor_model(merge_config, options=options)

    for file_name in files:
        fp = os.path.join(donor_local_path, file_name)
        if os.path.exists(fp):
            LOG.info(f"Copying {file_name} from {donor_model}")
            shutil.copy(
                fp,
                os.path.join(out_path, file_name),
            )

    return


def _copy_tokenizer(
    merge_config: MergeConfiguration, out_path: str, options: MergeOptions
):
    donor_model, donor_local_path = _get_donor_model(merge_config, options=options)

    if (
        (not merge_config.chat_template)
        and os.path.exists(os.path.join(donor_local_path, "tokenizer_config.json"))
        and (
            os.path.exists(os.path.join(donor_local_path, "tokenizer.json"))
            or os.path.exists(os.path.join(donor_local_path, "tokenizer.model"))
        )
    ):
        LOG.info(f"Copying tokenizer from {donor_model}")

        for file_name in [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "added_tokens.json",
            "merges.txt",
        ]:
            if os.path.exists(os.path.join(donor_local_path, file_name)):
                shutil.copy(
                    os.path.join(donor_local_path, file_name),
                    os.path.join(out_path, file_name),
                )

        return

    # fallback: try actually loading the tokenizer and saving it
    LOG.info(f"Reserializing tokenizer from {donor_model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        donor_model.model.path,
        revision=donor_model.model.revision,
        trust_remote_code=options.trust_remote_code,
    )
    _set_chat_template(tokenizer, merge_config)
    tokenizer.save_pretrained(out_path, safe_serialization=True)


def _model_out_config(
    config: MergeConfiguration,
    arch_info: ModelArchitecture,
    trust_remote_code: bool = False,
) -> transformers.PretrainedConfig:
    """Return a configuration for the resulting model."""
    if config.base_model:
        res = config.base_model.config(trust_remote_code=trust_remote_code)
    else:
        res = config.referenced_models()[0].config(trust_remote_code=trust_remote_code)
    if config.out_dtype:
        res.torch_dtype = config.out_dtype
    elif config.dtype:
        res.torch_dtype = config.dtype

    module_layers = {}
    for module_name in arch_info.modules:
        if config.modules and module_name in config.modules:
            module_def = config.modules.get(module_name)
            if module_def and module_def.slices:
                module_layers[module_name] = sum(
                    [
                        s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
                        for s in module_def.slices
                    ]
                )
        elif config.slices:
            module_layers[module_name] = sum(
                [
                    s.sources[0].layer_range[1] - s.sources[0].layer_range[0]
                    for s in config.slices
                ]
            )

    if module_layers:
        for module_name in module_layers:
            if module_name not in arch_info.modules:
                LOG.warning(
                    f"Module {module_name} in config but not in architecture info"
                )
                continue
            module_info = arch_info.modules[module_name]
            cfg_key = module_info.architecture.num_layers_config_key()
            if not cfg_key:
                if module_layers[module_name] > 0:
                    LOG.warning(
                        f"Module {module_name} has no configuration key for number of layers, "
                        "but the number of layers is not zero."
                    )
                continue
            try:
                set_config_value(res, cfg_key, module_layers[module_name])
            except Exception as e:
                LOG.warning(
                    f"Unable to set number of layers for module {module_name} in output config "
                    "- you may need to manually correct it.",
                    exc_info=e,
                )

    return res


def _update_config_vocab(
    config: transformers.PretrainedConfig,
    arch_info: ModelArchitecture,
    tokenizer: transformers.PreTrainedTokenizerBase,
    pad_to_multiple_of: Optional[int] = None,
):
    vocab_size = len(tokenizer.get_vocab())
    if pad_to_multiple_of and vocab_size % pad_to_multiple_of:
        vocab_size = vocab_size + pad_to_multiple_of - (vocab_size % pad_to_multiple_of)
    try:
        set_config_value(
            config, arch_info.vocab_size_config_key or "vocab_size", vocab_size
        )
    except Exception as e:
        LOG.warning(
            "Unable to set vocabulary size in output config - you may need to manually correct it.",
            exc_info=e,
        )


__all__ = ["MergeOptions", "run_merge"]
