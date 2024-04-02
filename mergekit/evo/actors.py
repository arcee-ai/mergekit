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

import shutil
import tempfile
from typing import Optional

import lm_eval
import lm_eval.models.huggingface
import ray
import torch
import transformers

from mergekit.architecture import ConfiguredArchitectureInfo, get_architecture_info
from mergekit.config import MergeConfiguration
from mergekit.evo.genome import EvolMergeConfiguration, ModelGenome
from mergekit.evo.monkeypatch import NoInit, monkeypatch_lmeval_shuffle
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache, ReturnTensor
from mergekit.merge import _model_out_config, run_merge
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner


@ray.remote(num_cpus=1, num_gpus=1.0)
class OnDiskMergeEvaluator:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.cache = LoaderCache()
        self.cache.setup(merge_options)
        self.model_storage_path = model_storage_path
        self.vllm = vllm

        if config.shuffle:
            monkeypatch_lmeval_shuffle()

    async def _merge_model(
        self,
        genotype: torch.Tensor,
    ) -> str:
        cfg = self.genome.genotype_merge_config(genotype)
        res = tempfile.mkdtemp(prefix="merged", dir=self.model_storage_path)
        run_merge(cfg, out_path=res, options=self.merge_options)
        return res

    async def _evaluate_model(
        self,
        merged_path: str,
    ) -> float:
        model_args = {
            "pretrained": merged_path,
            "dtype": "bfloat16",
        }
        if self.vllm:
            model_args["gpu_memory_utilization"] = 0.8
            model_args["tensor_parallel_size"] = 1
            model_args["batch_size"] = "auto"
        else:
            model_args["use_cache"] = True

        results = lm_eval.evaluator.simple_evaluate(
            model="vllm" if self.vllm else "hf",
            model_args=model_args,
            tasks=[self.config.task],
            log_samples=False,
            verbosity="WARNING",
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
        )

        shutil.rmtree(merged_path)

        print(results["results"][self.config.task])
        return -results["results"][self.config.task]["acc,none"]

    async def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> float:
        merged_path = await self._merge_model(genotype)
        return await self._evaluate_model(merged_path)


@ray.remote(num_cpus=1, num_gpus=1)
class InMemoryMergeEvaluator:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.cache = LoaderCache()
        self.cache.setup(merge_options)
        self.model_storage_path = model_storage_path

        assert not vllm, "VLLM is not supported for in-memory merging"

        self.arch_info = None
        if config.shuffle:
            monkeypatch_lmeval_shuffle()

    def _maybe_init_model(self, config: MergeConfiguration):
        ai = get_architecture_info(self.genome._input_config_example)
        cfg_out = _model_out_config(
            config,
            ai,
            trust_remote_code=self.merge_options.trust_remote_code,
        )
        if self.arch_info is not None and self.arch_info.config == cfg_out:
            return

        with NoInit():
            self.model = (
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

        self.arch_info = ConfiguredArchitectureInfo(info=ai, config=cfg_out)
        print("Model initialized")

    def evaluate(self, genotype: torch.Tensor) -> float:
        config = self.genome.genotype_merge_config(genotype)
        self._maybe_init_model(config)

        planner = MergePlanner(
            config,
            self.arch_info.info,
            self.merge_options,
            self.arch_info.config,
        )

        tasks = planner.plan_in_memory()
        executor = Executor(tasks, math_device="cuda", storage_device="cuda")
        for tensor_task, value in executor.run(quiet=True):
            assert isinstance(tensor_task, ReturnTensor)
            if self.model.load_state_dict(
                {tensor_task.weight_info.name: value}, strict=False, assign=False
            ).unexpected_keys:
                raise RuntimeError(f"Unexpected keys in tensor {tensor_task}")

            del value

        model = lm_eval.models.huggingface.HFLM(pretrained=self.model)
        results = lm_eval.evaluator.simple_evaluate(
            model=model,
            tasks=[self.config.task],
            log_samples=False,
            verbosity="WARNING",
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
        )

        print(results["results"][self.config.task])
        return -results["results"][self.config.task]["acc,none"]

    async def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> float:
        return self.evaluate(genotype)
