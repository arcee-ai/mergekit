# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

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
from transformers.utils import is_flash_attn_2_available

from mergekit.architecture.base import ConfiguredModelArchitecture

try:
    import vllm
except ImportError:
    vllm = None


from mergekit.architecture import arch_info_for_config
from mergekit.config import MergeConfiguration
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.genome import InvalidGenotypeError, ModelGenome
from mergekit.evo.helpers import _eval_model, evaluate_model, merge_model
from mergekit.evo.monkeypatch import (
    NoInit,
    monkeypatch_lmeval_shuffle,
    monkeypatch_lmeval_vllm,
)
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache, ReturnTensor
from mergekit.merge import _model_out_config
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner

LOG = logging.getLogger(__name__)


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
        quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
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
        self.quantization_config = quantization_config

        if config.shuffle:
            monkeypatch_lmeval_shuffle()

        # monkeypatch_tqdm()
        monkeypatch_lmeval_vllm()


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
    ) -> dict:
        gc.collect()
        torch.cuda.empty_cache()
        LOG.info("Merging model")
        merged_path = merge_model(
            genotype, self.genome, self.model_storage_path, self.merge_options
        )
        if not merged_path:
            LOG.error("Model merge failed")
            return {"score": None, "results": None}

        model_kwargs = {}
        if self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config
        LOG.info(f"Model merged to {merged_path}")
        return evaluate_model(
            merged_path,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
            vllm=self.vllm,
            batch_size=self.batch_size,
            task_manager=self.task_manager,
            apply_chat_template=self.config.apply_chat_template,
            fewshot_as_multiturn=self.config.fewshot_as_multiturn,
            model_kwargs=model_kwargs,
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
    arch_info: Optional[ConfiguredModelArchitecture] = None

    def __init__(
        self,
        *args,
        vllm: bool = False,
        **kwargs,
    ):
        # assert not vllm, "VLLM is not supported for in-memory merging"
        super().__init__(*args, vllm=vllm, **kwargs)

    def _maybe_init_model(self, config: MergeConfiguration):
        ai = arch_info_for_config(self.genome._input_config_example)
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
                    LOG.warning(f"Config key {key} changed, reinitializing model")
                    different = True
                    break

            if not different:
                return

        self.inner_model = None

        model_kwargs = {
            "trust_remote_code": self.merge_options.trust_remote_code,
            "torch_dtype": torch.bfloat16,
        }
        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        with NoInit():
            inner_model = (
                transformers.AutoModelForCausalLM.from_config(
                    cfg_out,
                    **model_kwargs,
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
                tokenizer_donor = self.genome.definition.base_model
                if tokenizer_donor is None:
                    LOG.warning(
                        "Base model not set, using tokenizer from first model in genome"
                    )
                    tokenizer_donor = self.genome.definition.models[0]
                tok = transformers.AutoTokenizer.from_pretrained(
                    tokenizer_donor.model.path, use_fast=True
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
                    LOG.warning(f"Clipping sequence length to {max_model_len}")

                mem_util = (
                    0.7 if self.merge_options.cuda else 0.9
                )  # reduce memory usage if we're also using cuda for the merge
                self.model = lm_eval.models.vllm_causallms.VLLM(
                    pretrained=tempdir,
                    batch_size=self.batch_size or "auto",
                    max_model_len=max_model_len,
                    gpu_memory_utilization=mem_util,
                    dtype="bfloat16",
                    device="cuda",
                    trust_remote_code=self.merge_options.trust_remote_code,
                )
        else:
            self.model = lm_eval.models.huggingface.HFLM(pretrained=inner_model)
        self.arch_info = (
            ConfiguredModelArchitecture(
                info=ai,
                config=cfg_out,
            )
            if ai
            else None
        )
        LOG.info("Model initialized")

    def evaluate(self, genotype: torch.Tensor) -> dict:
        try:
            config = self.genome.genotype_merge_config(genotype)
        except InvalidGenotypeError as e:
            LOG.error("Invalid genotype", exc_info=e)
            return {"score": None, "results": None}

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
            engine = model.llm_engine
            if hasattr(engine, "model_executor"):
                worker = engine.model_executor.worker
            elif hasattr(engine, "driver_worker"):
                worker = engine.driver_worker
            else:
                raise ValueError("Unknown LLM engine type")
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

        executor = Executor(
            tasks,
            math_device="cuda" if self.merge_options.cuda else "cpu",
            storage_device="cuda" if self.merge_options.cuda else "cpu",
        )
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
            batch_size=self.batch_size,
            apply_chat_template=self.config.apply_chat_template,
            fewshot_as_multiturn=self.config.fewshot_as_multiturn,
        )

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> dict:
        return self.evaluate(genotype)
