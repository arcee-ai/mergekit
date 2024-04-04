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

import enum
import logging
import sys
from typing import Dict, Generator, List, Optional, Tuple, Union

import click
import torch
import tqdm
import transformers
from typing_extensions import TypeAlias

from mergekit.architecture import (
    ConfiguredArchitectureInfo,
    WeightInfo,
    get_architecture_info,
)
from mergekit.common import ModelReference
from mergekit.io import TensorWriter
from mergekit.io.tasks import LoaderCache
from mergekit.options import MergeOptions, add_merge_options

LOG = logging.getLogger(__name__)


@click.command("mergekit-tokensurgeon")
@click.argument("model", type=str)
@click.argument("donor", type=str)
@click.argument("out_path", type=str)
@click.option(
    "-v", "verbosity", count=True, help="Verbose logging", default=0, show_default=False
)
@click.option(
    "-k",
    type=int,
    default=8,
    help="Number of nearest neighbours to use for embedding interpolation",
)
@click.option(
    "--barycentric/--no-barycentric",
    "-b/-nb",
    is_flag=True,
    default=False,
    help="Use barycentric interpolation instead of distance weighting",
)
@click.option(
    "--cosine-similarity/--no-cosine-similarity",
    "-c/-nc",
    is_flag=True,
    default=False,
    help="Use cosine similarity for nearest neighbour search",
)
@add_merge_options
def main(
    model: str,
    donor: str,
    out_path: str,
    verbosity: int,
    k: int,
    barycentric: bool,
    cosine_similarity: bool,
    merge_options: MergeOptions,
):
    """
    Replace the tokenizer of a model with that of a donor model. Attempts to
    approximate embeddings for tokens that are in the donor model but not the
    original model.

    This greatly reduces the amount of training required to settle in the new
    embeddings, and potentially removes the need for fine-tuning entirely for
    tokenizers that are sufficiently similar.

    The model and donor model must have the same architecture.
    """
    log_level = logging.WARNING
    if verbosity == 1:
        log_level = logging.INFO
    elif verbosity > 1:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)
    LOG.warning("This tool is experimental and may produce unexpected results.")

    model = ModelReference.model_validate(model)
    donor = ModelReference.model_validate(donor)

    cache = LoaderCache()
    cache.setup(options=merge_options)

    device = "cuda" if merge_options.cuda else "cpu"

    arch_info, donor_cfg = validate_architecture(model, donor, merge_options)
    embed_info, lm_head_info = get_embedding_info(model, merge_options)
    donor_embed_info, donor_lm_head_info = get_embedding_info(donor, merge_options)

    _, old_vocab = load_tokenizer(model, merge_options)
    tokenizer, new_vocab = load_tokenizer(donor, merge_options)
    common_tokens = list(set(old_vocab.keys()) & set(new_vocab.keys()))

    old_embed = cache.get(model).get_tensor(
        embed_info.name, aliases=embed_info.aliases, device=device
    )
    donor_embed = cache.get(donor).get_tensor(
        donor_embed_info.name, aliases=donor_embed_info.aliases, device=device
    )

    (_, hidden_size_0) = old_embed.shape
    (_, hidden_size_1) = donor_embed.shape
    if hidden_size_1 != hidden_size_0:
        report_issue(
            f"Embedding sizes do not match: {hidden_size_0} vs {hidden_size_1}",
            error=not merge_options.allow_crimes,
        )

    min_overlap = max(hidden_size_0, hidden_size_1)
    if len(common_tokens) < min_overlap:
        report_issue(
            f"Common tokens ({len(common_tokens)}) less than embedding size ({min_overlap})",
            error=not merge_options.allow_crimes,
        )

    LOG.info("Computing new embeddings")
    new_embed = get_embeddings(
        old_embed,
        donor_embed,
        old_vocab,
        new_vocab,
        common_tokens,
        accept_prefix=False,
        k=k,
        barycentric=barycentric,
        cosine_similarity=cosine_similarity,
        name=embed_info.name,
    )

    if lm_head_info:
        old_lm_head = cache.get(model).get_tensor(
            lm_head_info.name, aliases=lm_head_info.aliases, device=device
        )
        donor_lm_head = cache.get(donor).get_tensor(
            donor_lm_head_info.name, aliases=donor_lm_head_info.aliases, device=device
        )

        LOG.info("Computing new lm_head embeddings")
        new_lm_head = get_embeddings(
            old_lm_head,
            donor_lm_head,
            old_vocab,
            new_vocab,
            common_tokens,
            accept_prefix=True,
            k=k,
            barycentric=barycentric,
            cosine_similarity=cosine_similarity,
            name=lm_head_info.name,
        )

    # Save out the new model
    LOG.info(f"Saving new model to {out_path}")
    writer = TensorWriter(
        out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )
    for weight_info in tqdm.tqdm(arch_info.all_weights(), desc="Saving weights"):
        if weight_info.name == embed_info.name:
            tensor = new_embed
        elif lm_head_info is not None and weight_info.name == lm_head_info.name:
            tensor = new_lm_head
        else:
            tensor = cache.get(model).get_tensor(
                weight_info.name, aliases=weight_info.aliases
            )
        writer.save_tensor(weight_info.name, tensor, clone=merge_options.clone_tensors)
    writer.finalize()

    tokenizer.save_pretrained(out_path)
    cfg_out = arch_info.config
    try:
        cfg_out.vocab_size = tokenizer.vocab_size
    except AttributeError:
        LOG.error(
            "Could not set vocab size in config.json - you may need to update it manually."
        )
    for key in [
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
        "unk_token_id",
        "mask_token_id",
        "padding_side",
    ]:
        if hasattr(donor_cfg, key) and (value := getattr(donor_cfg, key)) is not None:
            try:
                setattr(cfg_out, key, value)
            except AttributeError:
                LOG.error(f"Could not set {key}!")
    cfg_out.save_pretrained(out_path)


class TokenMarker(enum.Enum):
    SPECIAL = "special"
    WORD_START = "word_start"


NormalizedToken: TypeAlias = Union[str, Tuple[TokenMarker, str]]


def normalize_token(
    token: str,
    special_tokens_map: Dict[str, Union[str, List[str]]],
    word_start_prefix: str = "▁",
) -> NormalizedToken:
    """
    Normalize a token for comparison.
    """
    if token.startswith(word_start_prefix):
        return (TokenMarker.WORD_START, token[len(word_start_prefix) :])

    for special_token_type, values in special_tokens_map.items():
        if isinstance(values, str):
            values = [values]
        if token in values:
            return (TokenMarker.SPECIAL, special_token_type)
    return token


def token_prefixes(
    token: NormalizedToken, allow_whitespace: bool = False
) -> Generator[NormalizedToken, None, None]:
    """Yield potential prefixes of a token."""
    marker = None
    if isinstance(token, tuple):
        marker, token = token

    for i in range(len(token) - 1, 0, -1):
        prefix = token[:i]
        if not allow_whitespace and not prefix.strip():
            break
        if marker is not None:
            yield (marker, prefix)
        else:
            yield prefix


def get_embedding_info(
    model: ModelReference, options: MergeOptions
) -> Tuple[WeightInfo, WeightInfo]:
    """Get WeightInfo for the input and output embeddings of a model."""
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
    """Log an issue and exit if error is True."""
    if error:
        LOG.error(message)
        sys.exit(1)
    else:
        LOG.warning(message)


def get_embeddings(
    original_embed: torch.Tensor,
    donor_embed: torch.Tensor,
    original_vocab: Dict[NormalizedToken, int],
    donor_vocab: Dict[NormalizedToken, int],
    common_tokens: List[str],
    *,
    accept_prefix: bool = False,
    k: int = 8,
    barycentric: bool = False,
    cosine_similarity: bool = False,
    log_reconstruction_error: bool = True,
    log_statistics: bool = True,
    name: Optional[str] = None,
) -> torch.Tensor:
    """
    Generate embeddings for a target vocabulary.

    For tokens present in both vocabularies, the embedding from original_embed is
    directly copied. For tokens not present in the original vocabulary, the
    embedding is approximated using the k-nearest neighbours among the tokens that
    are present in both vocabularies. This can be done using either barycentric
    interpolation or distance weighted averaging.

    Args:
        original_embed (torch.Tensor): Embedding matrix for the original vocabulary.
        donor_embed (torch.Tensor): Embedding matrix for the new vocabulary.
        original_vocab (Dict[NormalizedToken, int]): Maps tokens to indices in
            original_embed.
        donor_vocab (Dict[NormalizedToken, int]): Maps tokens to indices in
            donor_embed.
        common_tokens (List[str]): Tokens that are common to both vocabularies.
        accept_prefix (bool): If True, allows using prefix matches for tokens when
            an exact match is not found.
        k (int): Number of nearest neighbours to use for embedding interpolation.
        barycentric (bool): If True, uses barycentric interpolation for embedding
            approximation. Otherwise, uses distance weighting.
        cosine_similarity (bool): If True, uses cosine similarity to find nearest
            neighbors. Otherwise, uses Euclidean distance.
        log_reconstruction_error (bool): If True, logs the mean squared error of
            the reconstructed embeddings.
        log_statistics (bool): If True, logs statistics about the embedding
            approximation process.
        name (Optional[str]): Name of the embedding matrix. Used for logging.

    Returns:
        torch.Tensor: Embedding matrix for the new vocabulary.
            Shape is (len(donor_vocab), original_embed.shape[1]).
    """
    hidden_size_0 = original_embed.shape[1]
    hidden_size_1 = donor_embed.shape[1]

    e_c_0 = torch.empty(
        len(common_tokens),
        hidden_size_0,
        device=original_embed.device,
        dtype=original_embed.dtype,
    )
    e_c_1 = torch.empty(
        len(common_tokens),
        hidden_size_1,
        device=donor_embed.device,
        dtype=donor_embed.dtype,
    )
    for i, token in enumerate(common_tokens):
        idx_0 = original_vocab[token]
        idx_1 = donor_vocab[token]
        e_c_0[i] = original_embed[idx_0]
        e_c_1[i] = donor_embed[idx_1]

    exact_matches = 0
    prefix_matches = 0
    knn_matches = 0
    res = torch.zeros(
        max(donor_vocab.values()) + 1,
        hidden_size_0,
        device=original_embed.device,
        dtype=original_embed.dtype,
    )

    # message for tqdm
    desc = "Computing embeddings"
    if name:
        desc += f" ({name})"

    knn_reconstruction_error = []
    for token in tqdm.tqdm(donor_vocab, desc=desc):
        idx_1 = donor_vocab[token]
        if token in original_vocab:
            res[idx_1] = original_embed[original_vocab[token]]
            exact_matches += 1
            continue

        if isinstance(token, str):
            if len(token) == 1 and ord(token) < 256:
                # check for matching byte tokens
                byte_tok = f"<0x{ord(token):02X}>"
                if byte_tok in original_vocab:
                    res[idx_1] = original_embed[original_vocab[byte_tok]]
                    exact_matches += 1
                    continue
            elif token.startswith("<0x") and token.endswith(">") and len(token) == 6:
                # check for character tokens matching byte tokens
                try:
                    byte = int(token[3:-1], 16)
                except ValueError:
                    pass
                else:
                    if chr(byte) in original_vocab:
                        res[idx_1] = original_embed[original_vocab[chr(byte)]]
                        exact_matches += 1
                        continue

        if accept_prefix:
            # For the LM head, we can accept prefix matches so long as the prefix is
            # not also in the new vocab - this is to avoid including the same embedding
            # vector multiple times
            found_prefix = False
            for prefix in token_prefixes(token, allow_whitespace=False):
                if prefix in original_vocab and prefix not in donor_vocab:
                    res[idx_1] = original_embed[original_vocab[prefix]]
                    found_prefix = True
                    break

            if found_prefix:
                prefix_matches += 1
                continue

        # If we can't find a prefix match, approximate from k nearest neighbours
        token_embedding = donor_embed[idx_1]
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
                (1, knn_embeddings.shape[0]), device=original_embed.device
            )
            knn_e_c = torch.cat([knn_embeddings.T, constraint_row], dim=0)
            e_c = torch.cat(
                [
                    token_embedding,
                    torch.tensor([1.0], device=e_c_0.device, dtype=e_c_0.dtype),
                ]
            ).unsqueeze(-1)
            weights = torch.linalg.lstsq(
                knn_e_c.float(), e_c.float(), rcond=1e-6
            ).solution.to(dtype=e_c_0.dtype)
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

        if log_reconstruction_error:
            # compute reconstruction error in donor_embed space
            knn_reconstruction_error.append(
                torch.nn.functional.mse_loss(
                    (knn_embeddings.T.to(weights.dtype) @ weights).squeeze(),
                    token_embedding,
                ).item()
            )

        # Reconstruct the embedding in original_embed space
        res[idx_1] = (e_c_0[indices].T @ weights).squeeze()
        knn_matches += 1

    if log_statistics:
        LOG.info("Token breakdown:")
        LOG.info(f"\tExact matches: {exact_matches}")
        if prefix_matches:
            LOG.info(f"\tPrefix matches: {prefix_matches}")
        LOG.info(f"\tKNN solutions: {knn_matches}")

        pct_approx = int((len(donor_vocab) - exact_matches) * 100 / len(donor_vocab))
        if pct_approx > 10:
            # encourage best practices
            LOG.warning(
                f"Large number of tokens ({pct_approx}%) could not be exactly "
                "matched - be sure to fine tune this sucker!"
            )

    if knn_reconstruction_error:
        knn_err = torch.tensor(
            knn_reconstruction_error, device=original_embed.device, dtype=torch.float32
        )
        LOG.info("KNN reconstruction error:")
        LOG.info(f"\tMean: {knn_err.mean().item()}")
        LOG.debug(f"\tMedian: {knn_err.median().item()}")
        LOG.debug(f"\tMax: {knn_err.max().item()}")
        LOG.debug(f"\tMin: {knn_err.min().item()}")
        LOG.debug(f"\tStddev: {knn_err.std().item()}")
        if knn_err.mean().isnan() or knn_err.mean().isinf():
            LOG.error(
                "NaN or infinite reconstruction error detected - output is "
                "definitely broken!"
            )
        if knn_err.mean().item() >= 0.01:
            LOG.warning("Unreasonably high reconstruction error - expect some issues!")

    return res


def load_tokenizer(
    model: ModelReference, options: MergeOptions
) -> Tuple[transformers.PreTrainedTokenizerBase, Dict[NormalizedToken, int]]:
    """Load a tokenizer from a model. Returns the tokenizer and a mapping of
    normalized tokens to indices."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.model.path,
        revision=model.model.revision,
        trust_remote_code=options.trust_remote_code,
    )

    gpt2_style = [
        transformers.GPT2Tokenizer,
        transformers.GPT2TokenizerFast,
        transformers.OpenAIGPTTokenizer,
        transformers.OpenAIGPTTokenizerFast,
    ]
    for candidate in ["Qwen2Tokenizer", "Qwen2TokenizerFast"]:
        if hasattr(transformers, candidate):
            gpt2_style.append(getattr(transformers, candidate))

    sp_style = [
        transformers.LlamaTokenizer,
        transformers.LlamaTokenizerFast,
        transformers.T5Tokenizer,
        transformers.T5TokenizerFast,
    ]
    for candidate in ["GemmaTokenizer", "GemmaTokenizerFast"]:
        if hasattr(transformers, candidate):
            sp_style.append(getattr(transformers, candidate))

    vocab = tokenizer.get_vocab()
    if isinstance(
        tokenizer,
        tuple(gpt2_style),
    ):
        word_start_prefix = "Ġ"
    elif isinstance(
        tokenizer,
        tuple(sp_style),
    ):
        if "Ġhello" in vocab:
            # dumb special case for deepseek's tokenizer
            word_start_prefix = "Ġ"
        else:
            word_start_prefix = "▁"
    else:
        LOG.warning("Unknown tokenizer type - assuming 'Ġ' word start prefix")
        word_start_prefix = "Ġ"

    tokenizer.all_special_tokens
    return tokenizer, {
        normalize_token(
            token,
            special_tokens_map=tokenizer.special_tokens_map,
            word_start_prefix=word_start_prefix,
        ): i
        for token, i in vocab.items()
    }


def validate_architecture(
    model: ModelReference, donor: ModelReference, options: MergeOptions
) -> Tuple[ConfiguredArchitectureInfo, transformers.PretrainedConfig]:
    """
    Validate that the architectures of two models match.

    Returns the architecture info for the model and the config for the donor.
    """
    model_cfg = model.config(trust_remote_code=options.trust_remote_code)
    donor_cfg = donor.config(trust_remote_code=options.trust_remote_code)
    model_arch_info = get_architecture_info(model_cfg)
    donor_arch_info = get_architecture_info(donor_cfg)
    if donor_arch_info != model_arch_info:
        report_issue(
            f"Model architectures do not match: {model_arch_info.name()} vs {donor_arch_info.name()}",
            error=not options.allow_crimes,
        )

    return ConfiguredArchitectureInfo(info=model_arch_info, config=model_cfg), donor_cfg


if __name__ == "__main__":
    with torch.no_grad():
        main()
