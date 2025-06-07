# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import lm_eval.tasks
import numpy as np
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch
import transformers

from mergekit.common import get_torch_accelerator_count
from mergekit.evo.actors import InMemoryMergeEvaluator, OnDiskMergeEvaluator
from mergekit.evo.config import EvolMergeConfiguration
from mergekit.evo.genome import ModelGenome
from mergekit.evo.helpers import evaluate_model_ray, merge_model_ray
from mergekit.options import MergeOptions


class EvaluationStrategyBase(ABC):
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        num_gpus: Optional[int] = None,
        batch_size: Optional[int] = None,
        task_search_path: Union[str, List[str], None] = None,
        model_storage_path: Optional[str] = None,
        quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.num_gpus = num_gpus or get_torch_accelerator_count(
            self.merge_options.device
        )
        self.batch_size = batch_size
        self.task_manager = lm_eval.tasks.TaskManager(include_path=task_search_path)
        self.model_storage_path = model_storage_path
        self.quantization_config = quantization_config
        if self.model_storage_path:
            os.makedirs(self.model_storage_path, exist_ok=True)

    @abstractmethod
    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        pass

    @abstractmethod
    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        pass


class ActorPoolEvaluationStrategy(EvaluationStrategyBase):
    """
    Uses a fixed-size pool of actors to evaluate genotypes in parallel.
    """

    def __init__(
        self,
        *args,
        in_memory: bool = False,
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
                    model_storage_path=self.model_storage_path,
                    vllm=vllm,
                    batch_size=self.batch_size,
                    task_manager=self.task_manager,
                    quantization_config=self.quantization_config,
                )
                for _ in range(self.num_gpus)
            ]
        )

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        return list(
            self.actor_pool.map(
                lambda a, x: a.evaluate_genotype.remote(x),
                genotypes,
            )
        )

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return self.evaluate_genotypes([genotype])[0]


@ray.remote
class BufferedRayEvaluationStrategyActor:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        vllm: bool = False,
        num_gpus: Optional[int] = None,
        batch_size: Optional[int] = None,
        task_manager: Optional[lm_eval.tasks.TaskManager] = None,
        model_storage_path: Optional[str] = None,
        quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.vllm = vllm
        self.num_gpus = num_gpus or get_torch_accelerator_count(
            self.merge_options.device
        )
        self.input_queue = []
        self.batch_size = batch_size
        self.task_manager = task_manager
        self.model_storage_path = model_storage_path
        self.quantization_config = quantization_config
        self._shutdown = False

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
            while not self._shutdown:
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
                    kwargs = {}
                    if self.quantization_config is not None:
                        kwargs["quantization_config"] = self.quantization_config
                    evaluating[
                        evaluate_model_ray.remote(
                            merged_path,
                            self.config.tasks,
                            num_fewshot=self.config.num_fewshot,
                            limit=self.config.limit,
                            vllm=self.vllm,
                            batch_size=self.batch_size,
                            task_manager=self.task_manager,
                            apply_chat_template=self.config.apply_chat_template,
                            fewshot_as_multiturn=self.config.fewshot_as_multiturn,
                            **kwargs,
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
        self._shutdown = True


class BufferedRayEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        vllm: bool = False,
        in_memory: bool = False,
        **kwargs,
    ):
        if in_memory:
            raise ValueError("In-memory evaluation is not supported for buffered mode")

        super().__init__(*args, **kwargs)
        self.actor = BufferedRayEvaluationStrategyActor.options(
            max_concurrency=1000
        ).remote(
            self.config,
            self.genome,
            self.merge_options,
            model_storage_path=self.model_storage_path,
            vllm=vllm,
            num_gpus=self.num_gpus,
            task_manager=self.task_manager,
            batch_size=self.batch_size,
            quantization_config=self.quantization_config,
        )
        self.actor.process_queue.remote()

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        return ray.get([self.actor.evaluate_genotype.remote(x) for x in genotypes])

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return ray.get(self.actor.evaluate_genotype.remote(genotype))


@ray.remote
def evaluate_genotype_serial(
    genotype: np.ndarray,
    config: EvolMergeConfiguration,
    genome: ModelGenome,
    merge_options: MergeOptions,
    model_storage_path: Optional[str] = None,
    vllm: bool = False,
    batch_size: Optional[int] = None,
    task_manager: Optional[lm_eval.tasks.TaskManager] = None,
    quantization_config: Optional[transformers.BitsAndBytesConfig] = None,
):
    pg = ray.util.placement_group([{"CPU": 1, "GPU": 1}], strategy="STRICT_PACK")
    strat = ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
        placement_group=pg
    )
    merged_path = merge_model_ray.options(scheduling_strategy=strat).remote(
        genotype, genome, model_storage_path, merge_options
    )
    if not merged_path:
        return {"score": None, "results": None}
    kwargs = {}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    res = ray.get(
        evaluate_model_ray.options(scheduling_strategy=strat).remote(
            merged_path,
            config.tasks,
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            vllm=vllm,
            batch_size=batch_size,
            task_manager=task_manager,
            apply_chat_template=config.apply_chat_template,
            fewshot_as_multiturn=config.fewshot_as_multiturn,
            **kwargs,
        )
    )
    ray.util.remove_placement_group(pg)
    return res


class SerialEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        vllm: bool = False,
        in_memory: bool = False,
        **kwargs,
    ):
        self.vllm = vllm
        if in_memory:
            raise ValueError("In-memory evaluation is not supported for serial mode")
        super().__init__(*args, **kwargs)

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[dict]:
        return ray.get(
            [
                evaluate_genotype_serial.remote(
                    x,
                    self.config,
                    self.genome,
                    self.merge_options,
                    model_storage_path=self.model_storage_path,
                    vllm=self.vllm,
                    batch_size=self.batch_size,
                    task_manager=self.task_manager,
                    quantization_config=self.quantization_config,
                )
                for x in genotypes
            ]
        )

    def evaluate_genotype(self, genotype: np.ndarray) -> dict:
        return self.evaluate_genotypes([genotype])[0]
