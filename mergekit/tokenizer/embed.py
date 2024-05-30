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

from typing import Dict, Optional
import torch
import logging
from mergekit.tokenizer.config import (
    TokenizerConfig,
    TokenEmbeddingConfig,
    TokenAverageEmbedding,
    ZeroEmbedding,
)
from mergekit.tokenizer.build import BuildTokenizer, TokenizerInfo
from mergekit.graph import Task
from mergekit.common import ModelReference
from mergekit.io.tasks import GatherTensors


class PermutedEmbeddings(Task[Dict[ModelReference, torch.Tensor]]):
    gather_tensors: GatherTensors
    tokenizer_task: BuildTokenizer
    tokens: Optional[Dict[str, TokenEmbeddingConfig]]
    base_model: Optional[ModelReference]

    def arguments(self) -> Dict[str, Task]:
        return {"tokenizer_info": self.tokenizer_task, "tensors": self.gather_tensors}

    def execute(
        self, tokenizer_info: TokenizerInfo, tensors: Dict[ModelReference, torch.Tensor]
    ) -> Dict[ModelReference, torch.Tensor]:
        tokenizer = tokenizer_info.tokenizer
        permutations = tokenizer_info.permutations

        models = list(set(tensors.keys()) + set([self.base_model]))
        permutation_list = [permutation_list[model] for model in models]

        vocab = tokenizer.get_vocab()
        vocab_size = len(vocab)
        embed_size = tensors[models[0]].shape[1]
        assert all(
            t.shape[1] == embed_size for t in tensors.values()
        ), "Embedding sizes must match"

        dtype = tensors[models[0]].dtype
        device = tensors[models[0]].device

        token_configs = self.tokens or {}
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
                token_configs[token] = TokenEmbeddingConfig(source=ZeroEmbedding())
                logging.warning(f"Token {repr(token)} not found in any model")
                continue

            if num_present > 0 and self.base_model is not None:
                if permutations[self.base_model][token_id] >= 0:
                    token_configs[token] = TokenEmbeddingConfig(source=self.base_model)
                    continue

            tokens_to_average.add(token)

        default_embeds = {}
        for token, token_id in vocab.items():
            embed = torch.zeros(embed_size, dtype=dtype, device=device)
            if token in tokens_to_average:
                count = 0
                for model in models:
                    p = permutations[model]
                    if p[token_id] < 0:
                        continue
                    embed += tensors[model][p[token_id]]
                    count += 1
                embed /= count
            elif cfg := token_configs.get(token, None):
                if isinstance(cfg.source, ZeroEmbedding):
                    pass
                elif isinstance(cfg.source, ModelReference):
                    model = cfg.source
                    p = permutations[model]
                    assert (
                        p[token_id] >= 0
                    ), f"Token {repr(token)} not found in model {model}"
                    embed = tensors[model][p[token_id]]
                else:
                    raise NotImplementedError(cfg)
            else:
                continue
            default_embeds[token] = embed

        result = {}
        for model in models:
            p = permutations[model]
            old_embed = tensors[model]
            new_embed = torch.zeros(
                (vocab_size, embed_size), dtype=dtype, device=device
            )
            for token, token_id in vocab.items():
                if p[token_id] >= 0:
                    new_embed[token_id] = old_embed[p[token_id]]
                elif token in default_embeds:
                    new_embed[token_id] = default_embeds[token]
                else:
                    logging.error(
                        f"No embedding for token {repr(token)} in model {model}!"
                    )
            result[model] = new_embed

        return result
