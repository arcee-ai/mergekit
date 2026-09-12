# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from typing import Dict, Optional

import torch

from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.tokenizer.build import BuildTokenizer, TokenizerInfo
from mergekit.tokenizer.config import (
    ModelTokenEmbedding,
    TokenEmbeddingConfig,
    ZeroEmbedding,
)
from mergekit.vocabulary import vocabulary_views


class PermutedVocabulary(Task[Dict[ModelReference, torch.Tensor]]):
    """Align token-indexed slices, preserving every non-vocabulary dimension."""

    gather_tensors: GatherTensors
    tokenizer_task: BuildTokenizer
    tokens: Optional[ImmutableMap[str, TokenEmbeddingConfig]]
    pad_to_multiple_of: Optional[int]
    base_model: Optional[ModelReference]
    vocabulary_axis: int

    def arguments(self) -> Dict[str, Task]:
        return {"tokenizer_info": self.tokenizer_task, "tensors": self.gather_tensors}

    def execute(
        self, tokenizer_info: TokenizerInfo, tensors: Dict[ModelReference, torch.Tensor]
    ) -> Dict[ModelReference, torch.Tensor]:
        tokenizer = tokenizer_info.tokenizer
        permutations = tokenizer_info.permutations

        if not tensors:
            return {}
        models = list(tensors)
        views = vocabulary_views(list(tensors.values()), self.vocabulary_axis)
        tensors = dict(zip(models, views))

        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)
        if self.pad_to_multiple_of and vocab_size % self.pad_to_multiple_of:
            vocab_size = (
                vocab_size // self.pad_to_multiple_of + 1
            ) * self.pad_to_multiple_of
        token_shape = views[0].shape[1:]

        dtype = tensors[models[0]].dtype
        for tensor in tensors.values():
            dtype = torch.promote_types(dtype, tensor.dtype)
        device = tensors[models[0]].device

        token_configs = dict(**(self.tokens or {}))
        tokens_to_average = self.assign_embedding_sources(
            permutations, models, vocab, token_configs
        )

        default_embeds = {}
        for token, token_id in vocab.items():
            cfg = token_configs.get(token)
            if not (cfg and cfg.force) and all(
                permutations[model][token_id] >= 0 for model in models
            ):
                continue
            embed = torch.zeros(token_shape, dtype=dtype, device=device)
            if token in tokens_to_average:
                count = 0
                for model in models:
                    p = permutations[model]
                    if p[token_id] < 0:
                        continue
                    embed = embed + tensors[model][p[token_id]]
                    count += 1
                embed /= count
            elif cfg is not None:
                embed = self.compute_default_embedding(
                    tokenizer_info, tensors, permutations, token, token_id, cfg
                )
            else:
                continue
            default_embeds[token] = embed

        result = {}
        for model in models:
            p = permutations[model]
            old_embed = tensors[model]
            new_embed = torch.zeros(
                (vocab_size, *token_shape), dtype=dtype, device=device
            )
            for token, token_id in vocab.items():
                force = False
                if token in token_configs:
                    force = token_configs[token].force

                if p[token_id] >= 0 and not force:
                    new_embed[token_id] = old_embed[p[token_id]]
                elif token in default_embeds:
                    new_embed[token_id] = default_embeds[token]
                else:
                    logging.error(
                        f"No embedding for token {repr(token)} in model {model}!"
                    )

            if vocab_size > len(vocab):
                # as suggested by https://nlp.stanford.edu/~johnhew/vocab-expansion.html
                avg_embed = torch.mean(new_embed[: len(vocab)], dim=0)
                new_embed[len(vocab) :] = avg_embed
            result[model] = new_embed.movedim(0, self.vocabulary_axis)

        return result

    def assign_embedding_sources(
        self,
        permutations: Dict[ModelReference, Dict[int, int]],
        models: list[ModelReference],
        vocab: Dict[str, int],
        token_configs: Dict[str, TokenEmbeddingConfig],
    ):
        permutation_list = [permutations[model] for model in models]

        tokens_to_average = set()
        # find tokens that are only present in one model
        for token, token_id in vocab.items():
            if token in token_configs:
                continue

            has_token = [p[token_id] >= 0 for p in permutation_list]
            num_present = sum(int(x) for x in has_token)
            if num_present == 1:
                donor_model = models[has_token.index(True)]
                token_configs[token] = TokenEmbeddingConfig(source=donor_model)
                continue

            if num_present == 0:
                token_configs[token] = TokenEmbeddingConfig(
                    source=ZeroEmbedding(kind="zero")
                )
                logging.warning(f"Token {repr(token)} not found in any model")
                continue

            if num_present > 0 and self.base_model in models:
                if permutations[self.base_model][token_id] >= 0:
                    token_configs[token] = TokenEmbeddingConfig(source=self.base_model)
                    continue

            tokens_to_average.add(token)
        return tokens_to_average

    def compute_default_embedding(
        self,
        tokenizer_info: TokenizerInfo,
        tensors: Dict[ModelReference, torch.Tensor],
        permutations: Dict[ModelReference, Dict[int, int]],
        token: str,
        token_id: int,
        cfg: TokenEmbeddingConfig,
    ) -> torch.Tensor:
        if isinstance(cfg.source, ZeroEmbedding):
            tensor = next(iter(tensors.values()))
            return tensor.new_zeros(tensor.shape[1:])
        model = (
            cfg.source.model
            if isinstance(cfg.source, ModelTokenEmbedding)
            else cfg.source
        )
        if not isinstance(model, ModelReference):
            raise NotImplementedError(cfg)
        if model not in tensors:
            raise ValueError(
                f"Token {token!r} requires a vocabulary weight from model {model}, "
                "but that weight is missing"
            )
        if isinstance(cfg.source, ModelTokenEmbedding):
            src_token_id = cfg.source.token_id
            if src_token_id is None:
                src_token = cfg.source.token
                assert src_token in tokenizer_info.original_vocabs[model], (
                    f"Token {repr(src_token)} not found in model {model}"
                )
                src_token_id = tokenizer_info.original_vocabs[model][src_token]
            assert src_token_id >= 0 and src_token_id < tensors[model].shape[0], (
                f"Token ID {src_token_id} out of range for model {model}"
            )
            embed = tensors[model][src_token_id]
        else:
            p = permutations[model]
            assert p[token_id] >= 0, f"Token {repr(token)} not found in model {model}"
            embed = tensors[model][p[token_id]]
        return embed
