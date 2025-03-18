# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import enum
import logging
from typing import Dict, List, Optional, Tuple

import click
import torch
import transformers
from pydantic import BaseModel

from mergekit.architecture import (
    ConfiguredModelArchitecture,
    WeightInfo,
    arch_info_for_config,
)
from mergekit.common import ModelReference, set_config_value
from mergekit.graph import Executor, Task
from mergekit.io.tasks import (
    FinalizeModel,
    LoaderCache,
    LoadTensor,
    SaveTensor,
    TensorWriterTask,
)
from mergekit.multigpu_executor import MultiGPUExecutor
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options
from mergekit.tokenizer.normalization import (
    NormalizedToken,
    TokenMarker,
    normalized_vocabulary,
    token_prefixes,
)


class TokenAssignmentStats(BaseModel):
    exact_match: int = 0
    byte_match: int = 0
    prefix_match: int = 0
    to_approximate: int = 0

    def pretty_print(self):
        chunks = ["Token Breakdown:"]
        if self.exact_match:
            chunks.append(f"  Exact matches: {self.exact_match}")
        if self.byte_match:
            chunks.append(f"  Byte matches: {self.byte_match}")
        if self.prefix_match:
            chunks.append(f"  Prefix matches: {self.prefix_match}")
        if self.to_approximate:
            chunks.append(f"  Tokens to approximate: {self.to_approximate}")
        chunks.append(
            f"  Total: {self.exact_match + self.byte_match + self.prefix_match + self.to_approximate}"
        )
        return "\n".join(chunks)


class TokenizerCache:
    loaded: Dict[ModelReference, transformers.PreTrainedTokenizerBase]
    trust_remote_code: bool = False
    _instance: Optional["TokenizerCache"] = None

    def __new__(cls) -> "TokenizerCache":
        if cls._instance is None:
            res = super(TokenizerCache, cls).__new__(cls)
            res.loaded = {}
            cls._instance = res
        return cls._instance

    def get(self, model: ModelReference) -> transformers.PreTrainedTokenizerBase:
        if model not in self.loaded:
            self.loaded[model] = transformers.AutoTokenizer.from_pretrained(
                model.model.path,
                revision=model.model.revision,
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
            )
        return self.loaded[model]


class EmbeddingKnnTask(Task[Tuple[torch.Tensor, torch.Tensor]]):
    target_tensor: Task
    common_embedding: Task
    k: int
    cosine_similarity: bool = False

    def arguments(self):
        return {
            "target": self.target_tensor,
            "common_embeddings": self.common_embedding,
        }

    def uses_accelerator(self):
        return True

    def execute(self, target: torch.Tensor, common_embeddings: torch.Tensor):
        if self.cosine_similarity:
            distances = 1 - torch.nn.functional.cosine_similarity(
                target.unsqueeze(0), common_embeddings, dim=1
            )
        else:
            distances = torch.cdist(
                target.unsqueeze(0), common_embeddings, p=2
            ).squeeze()
        distances, indices = torch.topk(distances, self.k, largest=False)
        knn_embeddings = common_embeddings[indices]
        return distances, knn_embeddings


class BarycentricWeightsTask(Task[torch.Tensor]):
    target_tensor: Task  # [torch.Tensor]
    knn_task: Task  # [Tuple[torch.Tensor, torch.Tensor]]

    def arguments(self):
        return {
            "target": self.target_tensor,
            "knn": self.knn_task,
        }

    def uses_accelerator(self):
        return True

    def execute(self, target: torch.Tensor, knn: Tuple[torch.Tensor, torch.Tensor]):
        distances, knn_embeddings = knn

        # Find least squares barycentric weights
        # Constrain sum of weights to 1 by adding a row of 1s
        constraint_row = torch.ones(
            (1, knn_embeddings.shape[0]), device=target.device
        )  # (1, k)
        knn_e_c = torch.cat([knn_embeddings.T, constraint_row], dim=0)
        e_c = torch.cat(
            [
                target,
                torch.tensor([1.0], device=target.device, dtype=target.dtype),
            ]
        ).unsqueeze(-1)
        # torch.linalg.lstsq doesn't work for rank-deficient matrices on CUDA
        # despite it being explicitly recommended for this use case in the docs
        # so pinv instead
        weights = torch.linalg.pinv(knn_e_c, rcond=1e-6) @ e_c
        return weights[:-1]


class DistanceWeightsTask(Task[torch.Tensor]):
    knn_task: Task  # [Tuple[torch.Tensor, torch.Tensor]]

    def arguments(self):
        return {
            "knn": self.knn_task,
        }

    def uses_accelerator(self):
        return True

    def execute(self, knn: Tuple[torch.Tensor, torch.Tensor]):
        distances, _ = knn
        return torch.nn.functional.softmin(distances, dim=0)


class ReconstructedEmbeddingTask(Task[torch.Tensor]):
    weights_task: Task  # [torch.Tensor]
    knn_task: Task  # [Tuple[torch.Tensor, torch.Tensor]]
    embeddings_task: Task  # [torch.Tensor]

    def arguments(self):
        return {
            "weights": self.weights_task,
            "knn": self.knn_task,
            "embeddings": self.embeddings_task,
        }

    def uses_accelerator(self):
        return True

    def execute(
        self,
        weights: torch.Tensor,
        knn: Tuple[torch.Tensor, torch.Tensor],
        embeddings: torch.Tensor,
    ):
        knn_indices, _ = knn
        return torch.sum(weights * embeddings[knn_indices], dim=0)


class SubwordEmbeddingSumTask(Task[torch.Tensor]):
    text: str
    tokenizer_from: ModelReference
    embeddings: Task
    average: bool = True

    def arguments(self):
        return {"embeddings": self.embeddings}

    def uses_accelerator(self):
        return True

    def execute(self, embeddings: torch.Tensor):
        tokenizer = TokenizerCache().get(self.tokenizer_from)
        tokenized = tokenizer(self.text)["input_ids"]
        res = torch.zeros_like(embeddings[0])
        for token in tokenized:
            res += embeddings[token]
        if self.average and len(tokenized) > 1:
            res /= len(tokenized)
        return res


class EmbeddingMeanTask(Task[torch.Tensor]):
    embeddings: Task

    def arguments(self):
        return {"embeddings": self.embeddings}

    def uses_accelerator(self):
        return True

    def execute(self, embeddings: torch.Tensor):
        return embeddings.mean(dim=0)


class IndexedEmbeddingTask(Task[torch.Tensor]):
    embeddings: Task
    index: int

    def arguments(self):
        return {"embeddings": self.embeddings}

    def uses_accelerator(self):
        return True

    def execute(self, embeddings: torch.Tensor):
        return embeddings[self.index]


class MultiIndexedEmbeddingTask(Task[torch.Tensor]):
    embeddings: Task
    indices: Tuple[int, ...]

    def arguments(self):
        return {"embeddings": self.embeddings}

    def uses_accelerator(self):
        return True

    def execute(self, embeddings: torch.Tensor):
        return torch.stack([embeddings[i] for i in self.indices], dim=0)

    def main_thread_only(self):
        return True

    def __hash__(self):
        # fun fact: hashing a tuple of 100k ints is very very slow
        # so just hash the embeddings task and let __eq__ sort it out
        return hash(("MultiIndexedEmbeddingTask", self.embeddings))

    def __eq__(self, other):
        if not isinstance(other, MultiIndexedEmbeddingTask):
            return False
        return self.indices == other.indices and self.embeddings == other.embeddings


class ZeroTensorTask(Task[torch.Tensor]):
    shape: Tuple[int, ...]

    def execute(self):
        return torch.zeros(self.shape)


class AssembleEmbeddingsTask(Task[torch.Tensor]):
    name: str
    embeddings: Tuple[Task, ...]

    def arguments(self):
        return {f"_e_{i}": task for i, task in enumerate(self.embeddings)}

    def uses_accelerator(self):
        return True

    def execute(self, **embeddings):
        return torch.stack(
            [embeddings[f"_e_{i}"] for i in range(len(self.embeddings))], dim=0
        )

    def main_thread_only(self):
        return True

    def __hash__(self):
        return hash(("AssembleEmbeddingsTask", self.name))

    def __eq__(self, other):
        if not isinstance(other, AssembleEmbeddingsTask):
            return False
        return self.name == other.name and self.embeddings == other.embeddings


class ApproximationMethod(enum.Enum):
    KNN_INTERPOLATION = "knn_interpolation"
    SUBWORD = "subword"
    MEAN = "mean"
    ZERO = "zero"


class TokenSurgeonOptions(BaseModel):
    model: ModelReference
    donor: ModelReference
    out_path: str
    method: ApproximationMethod = ApproximationMethod.KNN_INTERPOLATION
    k: int = 8
    cosine_similarity: bool = False
    barycentric: bool = False
    average: bool = True


def get_embedding_info(
    model: ModelReference, options: MergeOptions
) -> Tuple[WeightInfo, WeightInfo]:
    """Get WeightInfo for the input and output embeddings of a model."""
    cfg = model.config(trust_remote_code=options.trust_remote_code)
    arch_info = arch_info_for_config(cfg)

    if len(arch_info.modules) != 1:
        raise RuntimeError("Model has multiple modules - not supported by tokensurgeon")
    module_def = next(iter(arch_info.modules.values()))

    embed, lm_head = None, None
    for weight_info in module_def.architecture.pre_weights(cfg):
        if weight_info.is_embed:
            if embed is not None:
                raise RuntimeError("Multiple input embeddings found")
            embed = weight_info

    for weight_info in module_def.architecture.post_weights(cfg):
        if weight_info.is_embed:
            if lm_head is not None:
                raise RuntimeError("Multiple output embeddings found")
            lm_head = weight_info
    return embed, lm_head


def maybe_aliases(weight_info: WeightInfo, tied: bool) -> Tuple[str, ...]:
    return tuple(
        list(weight_info.aliases or [])
        + list((weight_info.tied_names or []) if tied else [])
    )


def get_stuff(
    model: ModelReference, options: MergeOptions, get_tied: bool = False
) -> Tuple[Dict[NormalizedToken, int], Optional[torch.Tensor], Optional[torch.Tensor]]:
    tokenizer = TokenizerCache().get(model)
    vocab = normalized_vocabulary(tokenizer)
    embed_wi, lm_head_wi = get_embedding_info(model, options)
    loader = LoaderCache().get(model)
    embed = loader.get_tensor(
        embed_wi.name,
        aliases=maybe_aliases(embed_wi, get_tied),
        raise_on_missing=not embed_wi.optional,
    )
    lm_head = loader.get_tensor(
        lm_head_wi.name,
        aliases=maybe_aliases(lm_head_wi, get_tied),
        raise_on_missing=not lm_head_wi.optional,
    )
    return vocab, embed, lm_head


def match_byte_token(
    token: NormalizedToken, original_vocab: Dict[NormalizedToken, int]
) -> Optional[int]:
    if not isinstance(token, str):
        return None
    if len(token) == 1 and ord(token) < 256:
        # check for matching byte tokens
        byte_tok = f"<0x{ord(token):02X}>"
        if byte_tok in original_vocab:
            return original_vocab[byte_tok]
    elif token.startswith("<0x") and token.endswith(">") and len(token) == 6:
        # check for character tokens matching byte tokens
        try:
            byte = int(token[3:-1], 16)
        except ValueError:
            pass
        else:
            if chr(byte) in original_vocab:
                return original_vocab[chr(byte)]
    return None


def match_prefix(
    token: NormalizedToken, original_vocab: Dict[NormalizedToken, int]
) -> Optional[int]:
    for prefix in token_prefixes(token):
        if prefix in original_vocab:
            return original_vocab[prefix]
    return None


def unnormalize_token(token: NormalizedToken) -> str:
    if isinstance(token, tuple):
        if token[0] == TokenMarker.WORD_START:
            return " " + token[1]
        return token[1]
    return token


def plan_embedding(
    options: TokenSurgeonOptions,
    original_vocab: Dict[NormalizedToken, int],
    donor_vocab: Dict[NormalizedToken, int],
    common_tokens: List[str],
    hidden_size: int,
    weight_info: WeightInfo,
    allow_prefix_match: bool = False,
) -> Tuple[Task[torch.Tensor], TokenAssignmentStats]:
    logging.info(f"Planning embedding surgery for {weight_info.name}")
    t_original_embed = LoadTensor(
        model=options.model,
        tensor=weight_info.name,
        optional=weight_info.optional,
        aliases=weight_info.aliases,
        tied_names=weight_info.tied_names,
        force_main_thread=True,
    )
    t_donor_embed = LoadTensor(
        model=options.donor,
        tensor=weight_info.name,
        optional=weight_info.optional,
        aliases=weight_info.aliases,
        tied_names=weight_info.tied_names,
        force_main_thread=True,
    )
    # e_c_0 = torch.stack(
    #     [original_embed[original_vocab[token]] for token in common_tokens]
    # )
    # e_c_1 = torch.stack([donor_embed[donor_vocab[token]] for token in common_tokens])
    t_e_c_0 = MultiIndexedEmbeddingTask(
        embeddings=t_original_embed,
        indices=tuple(original_vocab[token] for token in common_tokens),
    )
    t_e_c_1 = MultiIndexedEmbeddingTask(
        embeddings=t_donor_embed,
        indices=tuple(donor_vocab[token] for token in common_tokens),
    )
    mean_donor_embed_task = EmbeddingMeanTask(embeddings=t_donor_embed)

    stats = TokenAssignmentStats()
    embedding_tasks = []
    for tok_out in donor_vocab:
        tok_embedding_task = None
        idx_out = donor_vocab[tok_out]
        if tok_out in original_vocab:
            idx_in = original_vocab[tok_out]
            tok_embedding_task = IndexedEmbeddingTask(
                embeddings=t_original_embed, index=idx_in
            )
            stats.exact_match += 1
        elif byte_idx := match_byte_token(tok_out, original_vocab):
            tok_embedding_task = IndexedEmbeddingTask(
                embeddings=t_original_embed, index=byte_idx
            )
            stats.byte_match += 1
        elif allow_prefix_match and (
            prefix_idx := match_prefix(tok_out, original_vocab)
        ):
            tok_embedding_task = IndexedEmbeddingTask(
                embeddings=t_original_embed, index=prefix_idx
            )
            stats.prefix_match += 1
        else:
            # gotta approximate
            stats.to_approximate += 1
            if options.method == ApproximationMethod.KNN_INTERPOLATION:
                knn_task = EmbeddingKnnTask(
                    target_tensor=IndexedEmbeddingTask(
                        embeddings=t_donor_embed, index=idx_out
                    ),
                    common_embedding=t_e_c_1,
                    k=options.k,
                    cosine_similarity=options.cosine_similarity,
                )
                if options.barycentric:
                    weights_task = BarycentricWeightsTask(
                        target_tensor=t_donor_embed, knn_task=knn_task
                    )
                else:
                    weights_task = DistanceWeightsTask(knn_task=knn_task)
                reconstructed_task = ReconstructedEmbeddingTask(
                    weights_task=weights_task,
                    knn_task=knn_task,
                    embeddings_task=t_e_c_0,
                )
                tok_embedding_task = reconstructed_task
            elif options.method == ApproximationMethod.SUBWORD:
                tok_embedding_task = SubwordEmbeddingSumTask(
                    text=unnormalize_token(tok_out),
                    tokenizer_from=options.model,
                    embeddings=t_original_embed,
                    average=options.average,
                )
            elif options.method == ApproximationMethod.MEAN:
                tok_embedding_task = mean_donor_embed_task
            elif options.method == ApproximationMethod.ZERO:
                tok_embedding_task = ZeroTensorTask(shape=(hidden_size,))
            else:
                raise RuntimeError(f"Unknown approximation method: {options.method}")

        if tok_embedding_task is None:
            raise RuntimeError(f"Failed to create task for token: {tok_out}")
        embedding_tasks.append(tok_embedding_task)

    assemble_task = AssembleEmbeddingsTask(
        embeddings=embedding_tasks, name=weight_info.name
    )
    logging.info(stats.pretty_print())

    pct_approx = stats.to_approximate / len(donor_vocab) * 100
    logging.info(f"Approximation rate: {pct_approx:.2f}%")
    if pct_approx > 10:
        # encourage best practices
        logging.warning(
            f"Large number of tokens ({pct_approx}%) could not be exactly "
            "matched - be sure to fine tune this sucker!"
        )
    return assemble_task


def get_out_arch_info(
    model: ModelReference,
    donor: ModelReference,
    new_vocab_size: int,
    common_options: MergeOptions,
) -> ConfiguredModelArchitecture:
    cfg_donor = donor.config(trust_remote_code=common_options.trust_remote_code)
    cfg_out = model.config(trust_remote_code=common_options.trust_remote_code)
    arch_info_out = arch_info_for_config(cfg_out)
    set_config_value(
        cfg_out, arch_info_out.vocab_size_config_key or "vocab_size", new_vocab_size
    )
    for key in [
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
        "unk_token_id",
        "mask_token_id",
        "padding_side",
    ]:
        if hasattr(cfg_donor, key):
            set_config_value(cfg_out, key, getattr(cfg_donor, key))
    return ConfiguredModelArchitecture(info=arch_info_out, config=cfg_out)


def plan_surgery(
    options: TokenSurgeonOptions,
    common_options: MergeOptions,
) -> Tuple[List[Task], transformers.PretrainedConfig]:
    embed_wi, lm_head_wi = get_embedding_info(options.model, common_options)
    old_vocab, old_embed, old_lm_head = get_stuff(options.model, common_options)
    new_vocab, donor_embed, donor_lm_head = get_stuff(
        options.donor, common_options, get_tied=True
    )
    common_tokens = list(set(old_vocab.keys()) & set(new_vocab.keys()))

    writer_task = TensorWriterTask(
        out_path=options.out_path,
        max_shard_size=common_options.out_shard_size,
        safe_serialization=common_options.safe_serialization,
    )
    save_tasks = []
    if old_embed is not None:
        assert (
            donor_embed is not None
        ), "Donor model does not have an input embedding, but the target model does"
        embed_task = plan_embedding(
            options,
            old_vocab,
            new_vocab,
            common_tokens,
            hidden_size=old_embed.shape[1],
            weight_info=embed_wi,
            allow_prefix_match=False,
        )
        save_tasks.append(
            SaveTensor(
                tensor_name=embed_wi.name,
                tensor_task=embed_task,
                writer_task=writer_task,
                clone=False,
                force_main_thread=True,
            )
        )

    if old_lm_head is not None:
        assert (
            donor_lm_head is not None
        ), "Donor model does not have an output embedding, but the target model does"
        lm_head_task = plan_embedding(
            options,
            old_vocab,
            new_vocab,
            common_tokens,
            hidden_size=old_lm_head.shape[1],
            weight_info=lm_head_wi,
            allow_prefix_match=True,
        )
        save_tasks.append(
            SaveTensor(
                tensor_name=lm_head_wi.name,
                tensor_task=lm_head_task,
                writer_task=writer_task,
                clone=False,
                force_main_thread=True,
            )
        )

    arch_info_out = get_out_arch_info(
        options.model, options.donor, len(new_vocab), common_options
    )
    for weight in arch_info_out.all_weights():
        if weight.name not in {embed_wi.name, lm_head_wi.name}:
            load_task = LoadTensor(
                model=options.model,
                tensor=weight.name,
                optional=weight.optional,
                aliases=weight.aliases,
                tied_names=weight.tied_names,
            )
            save_tasks.append(
                SaveTensor(
                    tensor_name=weight.name,
                    tensor_task=load_task,
                    writer_task=writer_task,
                    clone=False,
                    optional=weight.optional,
                    dtype=weight.force_dtype,
                    force_main_thread=False,
                )
            )
    finalize = FinalizeModel(
        tensor_save_tasks=save_tasks,
        writer_task=writer_task,
    )
    return [finalize], arch_info_out.config


@click.command("mergekit-tokensurgeon", cls=PrettyPrintHelp)
@click.argument("model", type=str)
@click.argument("donor", type=str)
@click.argument("out_path", type=str)
@click.option(
    "--k",
    "-k",
    type=int,
    default=8,
    help="Number of nearest neighbours to use for embedding interpolation",
    show_default=True,
)
@click.option(
    "--barycentric/--no-barycentric",
    "-b/-nb",
    is_flag=True,
    default=False,
    help="Use barycentric interpolation instead of distance weighting",
    show_default=True,
)
@click.option(
    "--cosine-similarity/--no-cosine-similarity",
    "-c/-nc",
    is_flag=True,
    default=False,
    help="Use cosine similarity for nearest neighbour search",
    show_default=True,
)
@click.option(
    "--approximation-method",
    "-a",
    type=click.Choice([m.value for m in ApproximationMethod]),
    default=ApproximationMethod.KNN_INTERPOLATION.value,
    help="Method for approximating missing tokens",
    show_default=True,
)
@click.option(
    "--average/--no-average",
    is_flag=True,
    default=True,
    help="Use average instead of sum for subword embedding approximation",
    show_default=True,
)
@add_merge_options
def main(
    model: str,
    donor: str,
    out_path: str,
    k: int,
    barycentric: bool,
    cosine_similarity: bool,
    approximation_method: str,
    average: bool,
    merge_options: MergeOptions,
):
    # merge_options.apply_global_options()
    logging.basicConfig(level=logging.DEBUG)
    options = TokenSurgeonOptions(
        model=ModelReference.model_validate(model),
        donor=ModelReference.model_validate(donor),
        out_path=out_path,
        k=k,
        cosine_similarity=cosine_similarity,
        barycentric=barycentric,
        method=ApproximationMethod(approximation_method),
        average=average,
    )
    logging.info("Planning surgery...")
    tasks, out_config = plan_surgery(options, merge_options)

    logging.info(f"Writing config and tokenizer to {out_path}")
    out_config.save_pretrained(out_path)
    TokenizerCache().get(options.donor).save_pretrained(out_path)
    logging.info("Wrote them.")

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

    logging.info("Executing surgery...")
    executor.execute()
    logging.info("Done!")
