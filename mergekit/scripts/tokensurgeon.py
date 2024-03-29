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

import logging
import sys
import unicodedata
from typing import Dict, List, Tuple

import click
import torch
import tqdm
import transformers

from mergekit.architecture import (
    ConfiguredArchitectureInfo,
    WeightInfo,
    get_architecture_info,
)
from mergekit.common import ModelReference
from mergekit.io import TensorWriter
from mergekit.io.tasks import LoaderCache
from mergekit.options import MergeOptions, add_merge_options


@click.command("mergekit-tokensurgeon")
@click.argument("model", type=str)
@click.argument("donor", type=str)
@click.argument("out_path", type=str)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.option(
    "-k",
    type=int,
    default=5,
    help="Number of nearest neighbours to use for embedding interpolation",
)
@click.option(
    "--barycentric/--no-barycentric",
    is_flag=True,
    default=False,
    help="Use barycentric interpolation instead of distance-weighted",
)
@click.option(
    "--cosine-similarity/--no-cosine-similarity",
    is_flag=True,
    default=False,
    help="Use cosine similarity for nearest neighbour search",
)
@add_merge_options
def main(
    model: str,
    donor: str,
    out_path: str,
    verbose: bool,
    k: int,
    barycentric: bool,
    cosine_similarity: bool,
    merge_options: MergeOptions,
):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    model_ref = ModelReference.model_validate(model)
    donor_ref = ModelReference.model_validate(donor)
    replace_tokenizer(
        model=model_ref,
        donor=donor_ref,
        out_path=out_path,
        k=k,
        barycentric=barycentric,
        cosine_similarity=cosine_similarity,
        options=merge_options,
    )


def replace_tokenizer(
    model: ModelReference,
    donor: ModelReference,
    out_path: str,
    k: int,
    barycentric: bool,
    cosine_similarity: bool,
    options: MergeOptions,
):
    cache = LoaderCache()
    cache.setup(options=options)

    device = "cuda" if options.cuda else "cpu"

    arch_info = validate_architecture(model, donor, options)
    embed_info, lm_head_info = get_embedding_info(model, options)

    _, old_vocab = load_tokenizer(model, options)
    tokenizer, new_vocab = load_tokenizer(donor, options)
    common_tokens = list(set(old_vocab.keys()) & set(new_vocab.keys()))

    old_embed = cache.get(model).get_tensor(
        embed_info.name, aliases=embed_info.aliases, device=device
    )
    donor_embed = cache.get(donor).get_tensor(
        embed_info.name, aliases=embed_info.aliases, device=device
    )

    (_, hidden_size_0) = old_embed.shape
    (_, hidden_size_1) = donor_embed.shape
    if hidden_size_1 != hidden_size_0:
        report_issue(
            f"Embedding sizes do not match: {hidden_size_0} vs {hidden_size_1}",
            error=not options.allow_crimes,
        )

    min_overlap = max(hidden_size_0, hidden_size_1)
    if len(common_tokens) < min_overlap:
        report_issue(
            f"Common tokens ({len(common_tokens)}) less than embedding size ({min_overlap})",
            error=not options.allow_crimes,
        )

    logging.info("Computing new embeddings")
    new_embed = lstsq_embeddings(
        old_embed,
        donor_embed,
        old_vocab,
        new_vocab,
        common_tokens,
        k=k,
        barycentric=barycentric,
        cosine_similarity=cosine_similarity,
    )

    old_lm_head = cache.get(model).get_tensor(
        lm_head_info.name, aliases=lm_head_info.aliases, device=device
    )
    donor_lm_head = cache.get(donor).get_tensor(
        lm_head_info.name, aliases=lm_head_info.aliases, device=device
    )

    logging.info("Computing new lm_head embeddings")
    new_lm_head = lstsq_embeddings(
        old_lm_head,
        donor_lm_head,
        old_vocab,
        new_vocab,
        common_tokens,
        accept_prefix=True,
        k=k,
        barycentric=barycentric,
        cosine_similarity=cosine_similarity,
    )

    # Save out the new model
    logging.info(f"Saving new model to {out_path}")
    writer = TensorWriter(
        out_path,
        max_shard_size=options.out_shard_size,
        safe_serialization=options.safe_serialization,
    )
    for weight_info in tqdm.tqdm(arch_info.all_weights(), desc="Saving weights"):
        if weight_info.name == embed_info.name:
            tensor = new_embed
        elif weight_info.name == lm_head_info.name:
            tensor = new_lm_head
        else:
            tensor = cache.get(model).get_tensor(
                weight_info.name, aliases=weight_info.aliases
            )
        writer.save_tensor(weight_info.name, tensor, clone=options.clone_tensors)
    writer.finalize()

    tokenizer.save_pretrained(out_path)
    cfg_out = arch_info.config
    try:
        cfg_out.vocab_size = tokenizer.vocab_size
    except AttributeError:
        logging.error(
            "Could not set vocab size in config.json - you may need to update it manually."
        )
    cfg_out.save_pretrained(out_path)


def normalize_token(token: str) -> str:
    if token.startswith("Ġ"):
        return "▁" + token[1:]
    return unicodedata.normalize("NFKC", token)


def get_embedding_info(
    model: ModelReference, options: MergeOptions
) -> Tuple[WeightInfo, WeightInfo]:
    cfg = model.config(trust_remote_code=options.trust_remote_code)
    arch_info = get_architecture_info(cfg)

    embed, lm_head = None, None
    for weight_info in arch_info.pre_weights(cfg):
        if weight_info.is_embed:
            if embed is not None:
                raise RuntimeError("Multiple input embeddings found")
            embed = weight_info

    for weight_info in arch_info.post_weights(cfg):
        if weight_info.is_embed:
            if lm_head is not None:
                raise RuntimeError("Multiple output embeddings found")
            lm_head = weight_info

    return embed, lm_head


def report_issue(message: str, error: bool = False):
    if error:
        logging.error(message)
        sys.exit(1)
    else:
        logging.warning(message)


def lstsq_embeddings(
    embed_0: torch.Tensor,
    embed_1: torch.Tensor,
    vocab_0: Dict[str, int],
    vocab_1: Dict[str, int],
    common_tokens: List[str],
    accept_prefix: bool = False,
    k: int = 5,
    barycentric: bool = False,
    cosine_similarity: bool = False,
) -> torch.Tensor:
    hidden_size_0 = embed_0.shape[1]
    hidden_size_1 = embed_1.shape[1]

    e_c_0 = torch.empty(
        len(common_tokens), hidden_size_0, device=embed_0.device, dtype=embed_0.dtype
    )
    e_c_1 = torch.empty(
        len(common_tokens), hidden_size_1, device=embed_1.device, dtype=embed_1.dtype
    )
    for i, token in enumerate(common_tokens):
        idx_0 = vocab_0[token]
        idx_1 = vocab_1[token]
        e_c_0[i] = embed_0[idx_0]
        e_c_1[i] = embed_1[idx_1]

    exact_matches = 0
    prefix_matches = 0
    knn_matches = 0
    res = torch.zeros(
        max(vocab_1.values()) + 1,
        hidden_size_0,
        device=embed_0.device,
        dtype=embed_0.dtype,
    )

    knn_reconstruction_error = []
    for token in tqdm.tqdm(vocab_1, desc="Merging tokens"):
        idx_1 = vocab_1[token]
        if token in vocab_0:
            res[idx_1] = embed_0[vocab_0[token]]
            exact_matches += 1
            continue

        if accept_prefix:
            # For the LM head, we can accept prefix matches so long as the prefix is
            # not also in the new vocab - this is to avoid including the same embedding
            # vector multiple times
            found_prefix = False
            for j in range(len(token) - 1, 0, -1):
                prefix = token[:j]
                if prefix in vocab_0 and prefix not in vocab_1:
                    res[idx_1] = embed_0[vocab_0[prefix]]
                    found_prefix = True
                    break

            if found_prefix:
                prefix_matches += 1
                continue

        # If we can't find a prefix match, approximate from k nearest neighbours
        token_embedding = embed_1[idx_1]
        if cosine_similarity:
            cos_similarities = torch.nn.functional.cosine_similarity(
                token_embedding.unsqueeze(0), e_c_1, dim=1
            )
            distances = 1 - cos_similarities
        else:
            # euclidean distance
            distances = torch.cdist(token_embedding.unsqueeze(0), e_c_1).squeeze()
        _, indices = torch.topk(distances, k, largest=False)
        knn_embeddings = e_c_1[indices]

        if barycentric:
            # Find least squares barycentric weights
            # Constrain sum of weights to 1 by adding a row of 1s
            constraint_row = torch.ones(
                (1, knn_embeddings.shape[0]), device=embed_0.device
            )
            knn_e_c = torch.cat([knn_embeddings.T, constraint_row], dim=0)
            e_c = torch.cat(
                [
                    token_embedding,
                    torch.tensor([1.0], device=e_c_0.device, dtype=e_c_0.dtype),
                ]
            ).unsqueeze(-1)
            weights = torch.linalg.lstsq(knn_e_c.float(), e_c.float()).solution.to(
                dtype=e_c_0.dtype
            )
        else:
            # Just weight by distance
            if cosine_similarity:
                weights = cos_similarities[indices].unsqueeze(-1).to(dtype=e_c_0.dtype)
            else:
                # weights = 1 / distances[indices].to(dtype=e_c_0.dtype).clamp(min=1e-6)
                weights = torch.nn.functional.softmin(
                    distances[indices].to(dtype=e_c_0.dtype), dim=0
                )
            weights /= weights.sum()

        # compute reconstruction error in embed_1 space
        knn_reconstruction_error.append(
            torch.nn.functional.mse_loss(
                (knn_embeddings.T.to(weights.dtype) @ weights).squeeze(),
                token_embedding,
            ).item()
        )

        # Reconstruct the embedding in embed_0 space
        res[idx_1] = (e_c_0[indices].T @ weights).squeeze()
        knn_matches += 1

    logging.info("Token breakdown:")
    logging.info(f"\tExact matches: {exact_matches}")
    if prefix_matches:
        logging.info(f"\tPrefix matches: {prefix_matches}")
    logging.info(f"\tKNN solutions: {knn_matches}")
    if knn_reconstruction_error:
        knn_err = torch.tensor(
            knn_reconstruction_error, device=embed_0.device, dtype=torch.float32
        )
        logging.info("KNN reconstruction error:")
        logging.info(f"\tMean: {knn_err.mean().item()}")
        logging.info(f"\tMedian: {knn_err.median().item()}")
        logging.info(f"\tMax: {knn_err.max().item()}")
        logging.info(f"\tMin: {knn_err.min().item()}")
        logging.info(f"\tStddev: {knn_err.std().item()}")

    return res


def load_tokenizer(
    model: ModelReference, options: MergeOptions
) -> Tuple[transformers.PreTrainedTokenizerBase, Dict[str, int]]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.model.path,
        revision=model.model.revision,
        trust_remote_code=options.trust_remote_code,
    )
    return tokenizer, {
        normalize_token(token): i for token, i in tokenizer.get_vocab().items()
    }


def validate_architecture(
    model: ModelReference, donor: ModelReference, options: MergeOptions
) -> ConfiguredArchitectureInfo:
    model_cfg = model.config(trust_remote_code=options.trust_remote_code)
    model_arch_info = get_architecture_info(model_cfg)
    donor_arch_info = get_architecture_info(
        donor.config(trust_remote_code=options.trust_remote_code)
    )
    if donor_arch_info != model_arch_info:
        report_issue(
            f"Model architectures do not match: {model_arch_info} vs {donor_arch_info}",
            error=not options.allow_crimes,
        )

    return ConfiguredArchitectureInfo(info=model_arch_info, config=model_cfg)


if __name__ == "__main__":
    with torch.no_grad():
        main()
