# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import gc
import logging
import tempfile
from typing import Optional, Union

import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface
import lm_eval.tasks
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch
import transformers

try:
    import vllm
except ImportError:
    vllm = None


from mergekit.architecture import ConfiguredArchitectureInfo, get_architecture_info
from mergekit.config import MergeConfiguration
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.genome import ModelGenome
from mergekit.evo.helpers import _eval_model, evaluate_model, merge_model
from mergekit.evo.monkeypatch import NoInit, monkeypatch_lmeval_shuffle
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache, ReturnTensor
from mergekit.merge import _model_out_config
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner


class MergeActorBase:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        batch_size: Optional[int] = None,
        task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.cache = LoaderCache()
        self.cache.setup(merge_options)
        self.model_storage_path = model_storage_path
        self.vllm = vllm
        self.batch_size = batch_size
        self.task_manager = task_manager

        if config.shuffle:
            monkeypatch_lmeval_shuffle()

        # monkeypatch_tqdm()


@ray.remote(num_cpus=1, num_gpus=1.0)
class OnDiskMergeEvaluator(MergeActorBase):
    """
    Merges models to disk then evaluates them in a separate process.

    Maximum compatibility and potential for parallelism, but higher overhead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> float:
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Merging model")
        merged_path = merge_model(
            genotype, self.genome, self.model_storage_path, self.merge_options
        )
        logging.info(f"Model merged to {merged_path}")
        return evaluate_model(
            merged_path,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
            vllm=self.vllm,
            batch_size=self.batch_size,
            task_manager=self.task_manager,
        )


@ray.remote(num_cpus=1, num_gpus=1)
class InMemoryMergeEvaluator(MergeActorBase):
    """
    Performs merges in memory, using a single model instance.

    This reduces overhead from disk I/O and model loading, but prevents
    parallelism and may be slower for large models.

    Implementation is dark sorcery tampering with the internals of lm-eval,
    transformers, and vLLM and may break at any time.
    """

    model: Union[
        lm_eval.models.huggingface.HFLM, lm_eval.models.vllm_causallms.VLLM, None
    ] = None
    arch_info: Optional[ConfiguredArchitectureInfo] = None

    def __init__(
        self,
        *args,
        vllm: bool = False,
        **kwargs,
    ):
        # assert not vllm, "VLLM is not supported for in-memory merging"
        super().__init__(*args, vllm=vllm, **kwargs)

    def _maybe_init_model(self, config: MergeConfiguration):
        ai = get_architecture_info(self.genome._input_config_example)
        cfg_out = _model_out_config(
            config,
            ai,
            trust_remote_code=self.merge_options.trust_remote_code,
        )
        cfg_out.use_cache = True
        cfg_out.torch_dtype = torch.bfloat16

        if self.arch_info is not None:
            different = False
            for key in cfg_out.to_diff_dict():
                if key in ["architectures", "model_type"]:
                    # to get to here we must have --allow-crimes set, so let it ride
                    continue
                elif key in ["use_cache", "torch_dtype"]:
                    continue
                elif key.endswith("_token_id"):
                    # update our config but don't fail if it's different
                    setattr(self.arch_info.config, key, getattr(cfg_out, key, None))
                    continue

                if getattr(cfg_out, key) != getattr(self.arch_info.config, key, None):
                    logging.warn(f"Config key {key} changed, reinitializing model")
                    different = True
                    break

            if not different:
                return

        with NoInit():
            inner_model = (
                transformers.AutoModelForCausalLM.from_config(
                    cfg_out,
                    trust_remote_code=self.merge_options.trust_remote_code,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                )
                .bfloat16()
                .cuda()
                .eval()
                .requires_grad_(False)
            )

        if self.vllm:
            # oh i hate this
            with tempfile.TemporaryDirectory(
                dir=self.model_storage_path, prefix="vllm"
            ) as tempdir:
                inner_model.save_pretrained(
                    tempdir, safe_serialization=True, out_shard_size=1_000_000_000_000
                )
                del inner_model
                tok = transformers.AutoTokenizer.from_pretrained(
                    self.genome.definition.base_model.model.path, use_fast=True
                )
                tok.save_pretrained(tempdir)

                max_model_len = None
                if (
                    seq_len := getattr(cfg_out, "max_position_embeddings", None)
                ) is not None:
                    max_model_len = seq_len
                if (window_sz := getattr(cfg_out, "sliding_window", None)) is not None:
                    max_model_len = min(max_model_len or 1024, window_sz)
                if max_model_len and max_model_len > 8192:
                    max_model_len = 8192
                    logging.warn(f"Clipping sequence length to {max_model_len}")

                self.model = lm_eval.models.vllm_causallms.VLLM(
                    pretrained=tempdir,
                    batch_size=self.batch_size or "auto",
                    max_model_len=max_model_len,
                    gpu_memory_utilization=0.7,  # can't do 0.9 because the merge will OOM
                    dtype="bfloat16",
                    device="cuda",
                    trust_remote_code=self.merge_options.trust_remote_code,
                )
        else:
            self.model = lm_eval.models.huggingface.HFLM(pretrained=inner_model)
        self.arch_info = ConfiguredArchitectureInfo(info=ai, config=cfg_out)
        logging.info("Model initialized")

    def evaluate(self, genotype: torch.Tensor) -> dict:
        config = self.genome.genotype_merge_config(genotype)
        self._maybe_init_model(config)

        planner = MergePlanner(
            config,
            self.arch_info.info,
            self.merge_options,
            self.arch_info.config,
        )

        tasks = planner.plan_in_memory()

        model = self.model.model
        if vllm is not None and isinstance(model, vllm.LLM):
            assert (
                model.llm_engine.parallel_config.world_size == 1
            ), "Must be single GPU"
            worker = model.llm_engine.driver_worker
            model = worker.model_runner.model
        param_dict = dict(model.named_parameters())

        stacked_mapping = {
            # mappings for Llama/Mistral attention weights to vLLM packed tensors
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
        }

        executor = Executor(tasks, math_device="cuda", storage_device="cuda")
        for tensor_task, value in executor.run(quiet=True):
            assert isinstance(tensor_task, ReturnTensor)
            name = tensor_task.weight_info.name

            if name in param_dict:
                param_dict[name].data.copy_(value, non_blocking=True)
            elif self.vllm:
                stacked = False
                for needle, (replacement, shard_id) in stacked_mapping.items():
                    if needle in name:
                        target = name.replace(needle, replacement)
                        param = param_dict[target]
                        weight_loader = param.weight_loader
                        weight_loader(param, value, shard_id)
                        stacked = True
                        break

                if not stacked:
                    raise ValueError(f"Unknown parameter {name}")
            else:
                raise ValueError(f"Unknown parameter {name}")

            del value

        return _eval_model(
            self.model,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
            task_manager=self.task_manager,
        )

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> float:
        return self.evaluate(genotype)
