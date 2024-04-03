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

import asyncio
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface
import numpy as np
import ray
import ray.util.queue
import torch
import transformers

from mergekit.architecture import ConfiguredArchitectureInfo, get_architecture_info
from mergekit.config import MergeConfiguration
from mergekit.evo.config import EvolMergeConfiguration, TaskConfiguration
from mergekit.evo.genome import ModelGenome
from mergekit.evo.monkeypatch import (
    NoInit,
    monkeypatch_lmeval_shuffle,
    monkeypatch_tqdm,
)
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache, ReturnTensor
from mergekit.merge import _model_out_config, run_merge
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner


class EvaluationStrategyBase(ABC):
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        num_gpus: Optional[int] = None,
        merge_kwargs: Optional[Dict[str, Any]] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.merge_kwargs = merge_kwargs
        self.eval_kwargs = eval_kwargs

    @abstractmethod
    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[float]:
        pass

    @abstractmethod
    def evaluate_genotype(self, genotype: np.ndarray) -> float:
        pass


def _eval_model(
    model: Union[str, lm_eval.api.model.LM],
    tasks: List[TaskConfiguration],
    model_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    results = lm_eval.evaluator.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=[task.name for task in tasks],
        log_samples=False,
        verbosity="WARNING",
        **kwargs,
    )

    logging.info(results["results"])
    res = 0
    for task in tasks:
        res += results["results"][task.name][task.metric] * task.weight
    return res


def evaluate_model(
    merged_path: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    vllm: bool,
) -> float:
    monkeypatch_tqdm()
    try:
        model_args = {
            "pretrained": merged_path,
            "dtype": "bfloat16",
        }
        if vllm:
            model_args["gpu_memory_utilization"] = 0.8
            model_args["tensor_parallel_size"] = 1
            model_args["batch_size"] = "auto"
        else:
            model_args["use_cache"] = True
            model_args["batch_size"] = 32

        res = _eval_model(
            "vllm" if vllm else "huggingface",
            tasks,
            model_args,
            num_fewshot=num_fewshot,
            limit=limit,
        )
        return res
    finally:
        shutil.rmtree(merged_path)


evaluate_model_ray = ray.remote(num_cpus=1, num_gpus=0.9)(evaluate_model)


def merge_model(
    genotype: torch.Tensor,
    genome: ModelGenome,
    model_storage_path: str,
    merge_options: MergeOptions,
) -> str:
    monkeypatch_tqdm()
    cfg = genome.genotype_merge_config(genotype)
    res = tempfile.mkdtemp(prefix="merged", dir=model_storage_path)
    run_merge(cfg, out_path=res, options=merge_options)
    return res


merge_model_ray = ray.remote(
    num_cpus=1,
    num_gpus=0.1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model)


class MergeActorBase:
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

        monkeypatch_tqdm()


@ray.remote(num_cpus=1, num_gpus=1.0)
class OnDiskMergeEvaluator(MergeActorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _merge_model(
        self,
        genotype: torch.Tensor,
    ) -> str:
        return merge_model(
            genotype, self.genome, self.model_storage_path, self.merge_options
        )

    async def _evaluate_model(
        self,
        merged_path: str,
    ) -> float:
        return evaluate_model(
            merged_path,
            self.config.tasks,
            self.config.num_fewshot,
            self.config.limit,
            self.vllm,
        )

    async def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> float:
        merged_path = await self._merge_model(genotype)
        return await self._evaluate_model(merged_path)


@ray.remote(num_cpus=1, num_gpus=1)
class InMemoryMergeEvaluator(MergeActorBase):
    model: Optional[lm_eval.models.huggingface.HFLM] = None
    arch_info: Optional[ConfiguredArchitectureInfo] = None

    def __init__(
        self,
        *args,
        vllm: bool = False,
        **kwargs,
    ):
        assert not vllm, "VLLM is not supported for in-memory merging"
        super().__init__(*args, vllm=False, **kwargs)

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

        self.model = lm_eval.models.huggingface.HFLM(pretrained=inner_model)
        self.arch_info = ConfiguredArchitectureInfo(info=ai, config=cfg_out)
        logging.info("Model initialized")

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
            if self.model.model.load_state_dict(
                {tensor_task.weight_info.name: value}, strict=False, assign=False
            ).unexpected_keys:
                raise RuntimeError(f"Unexpected keys in tensor {tensor_task}")

            del value

        return _eval_model(
            self.model,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
        )

    async def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> float:
        return self.evaluate(genotype)


class ActorPoolEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        in_memory: bool = False,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if in_memory:
            self.actor_cls = InMemoryMergeEvaluator
        else:
            self.actor_cls = OnDiskMergeEvaluator

        self.actor_pool = ray.util.ActorPool(
            [
                self.actor_cls.remote(
                    self.config,
                    self.genome,
                    self.merge_options,
                    model_storage_path=model_storage_path,
                    vllm=vllm,
                )
                for _ in range(self.num_gpus)
            ]
        )

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[float]:
        return list(
            self.actor_pool.map(
                lambda a, x: a.evaluate_genotype.remote(x),
                genotypes,
            )
        )

    def evaluate_genotype(self, genotype: np.ndarray) -> float:
        return self.evaluate_genotypes([genotype])[0]


@ray.remote
class BufferedRayEvaluationStrategyActor:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        num_gpus: Optional[int] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.model_storage_path = model_storage_path
        self.vllm = vllm
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.input_queue = []
        self.shutdown = False

    async def evaluate_genotype(self, genotype: np.ndarray):
        future_result = asyncio.Future()
        self.input_queue.append((genotype, future_result))
        return await future_result

    async def process_queue(self):
        merging: Dict[ray.ObjectRef, asyncio.Future] = {}
        merged: List[Tuple[asyncio.Future, ray.ObjectRef]] = []
        evaluating: Dict[ray.ObjectRef, asyncio.Future] = {}

        logging.info("Starting processing loop")

        try:
            while not self.shutdown:
                while self.input_queue and (len(merging) + len(merged) < self.num_gpus):
                    genotype, future_result = self.input_queue.pop(0)
                    merging[
                        merge_model_ray.remote(
                            genotype,
                            self.genome,
                            self.model_storage_path,
                            self.merge_options,
                        )
                    ] = future_result

                while merged and len(evaluating) < self.num_gpus:
                    future_result, merged_path = merged.pop()
                    evaluating[
                        evaluate_model_ray.remote(
                            merged_path,
                            self.config.tasks,
                            self.config.num_fewshot,
                            self.config.limit,
                            self.vllm,
                        )
                    ] = future_result

                ready, _ = ray.wait(
                    list(merging.keys()) + list(evaluating.keys()),
                    num_returns=1,
                    fetch_local=False,
                    timeout=1,
                )
                for r in ready:
                    if r in merging:
                        future_result = merging.pop(r)
                        merged.append((future_result, r))
                    elif r in evaluating:
                        future_result = evaluating.pop(r)
                        future_result.set_result(await r)

                if (
                    not self.input_queue
                    and not merging
                    and not merged
                    and not evaluating
                ):
                    await asyncio.sleep(1)
        except Exception as e:
            logging.error("Error in processing loop", exc_info=e)
            raise

    async def shutdown(self):
        self.shutdown = True


class BufferedRayEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.actor = BufferedRayEvaluationStrategyActor.options(
            max_concurrency=1000
        ).remote(
            self.config,
            self.genome,
            self.merge_options,
            model_storage_path=model_storage_path,
            vllm=vllm,
            num_gpus=self.num_gpus,
        )
        self.actor.process_queue.remote()

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[float]:
        return ray.get([self.actor.evaluate_genotype.remote(x) for x in genotypes])

    def evaluate_genotype(self, genotype: np.ndarray) -> float:
        return ray.get(self.actor.evaluate_genotype.remote(genotype))
