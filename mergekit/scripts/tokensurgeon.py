# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import enum
import logging
from typing import Dict, List, Optional, Tuple

import click
import torch
import torch.distributions.constraints
import tqdm
import transformers
from pydantic import BaseModel

from mergekit.architecture import (
    ConfiguredModelArchitecture,
    WeightInfo,
    arch_info_for_config,
)
from mergekit.common import ModelReference, set_config_value
from mergekit.io.tasks import (
    LoaderCache,
)
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options
from mergekit.tokenizer.normalization import (
    NormalizedToken,
    normalized_vocabulary,
    token_prefixes,
)
from mergekit.tokensurgeon import (
    SubwordMethod,
    WeightingScheme,
    batch_omp,
    common_interp_approximate,
    subword_approximate,
)
from mergekit.tokensurgeon.common_interpolation import DistanceMetric

LOG = logging.getLogger(__name__)


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


class ApproximationMethod(enum.Enum):
    COMMON_INTERPOLATION = "common_interpolation"
    SUBWORD = "subword"
    MEAN = "mean"
    ZERO = "zero"
    RANDN = "randn"
    JOHN_HEWITT = "random_matching_distribution"
    ORTHOGONAL_MATCHING_PURSUIT = "omp"


class TokenSurgeonOptions(BaseModel):
    model: ModelReference
    donor: ModelReference
    out_path: str
    method: ApproximationMethod = ApproximationMethod.COMMON_INTERPOLATION
    weight_scheme: WeightingScheme = WeightingScheme.DISTANCE_PROPORTIONAL
    k: int = 8
    knn: bool = True
    cosine_similarity: bool = False
    subword_method: SubwordMethod = SubwordMethod.MEAN
    batch_size: Optional[int] = None


def get_arch_info(
    model: ModelReference, options: MergeOptions
) -> ConfiguredModelArchitecture:
    cfg = model.config(trust_remote_code=options.trust_remote_code)
    arch_info = arch_info_for_config(cfg)
    return ConfiguredModelArchitecture(info=arch_info, config=cfg)


def get_embedding_info(
    arch_info: ConfiguredModelArchitecture,
) -> Tuple[WeightInfo, WeightInfo]:
    """Get WeightInfo for the input and output embeddings of a model."""

    if len(arch_info.info.modules) != 1:
        raise RuntimeError("Model has multiple modules - not supported by tokensurgeon")
    name = next(iter(arch_info.info.modules.keys()))
    module_def = arch_info.get_module(name)

    embed, lm_head = None, None
    for weight_info in module_def.pre_weights():
        if weight_info.is_embed:
            if embed is not None:
                raise RuntimeError("Multiple input embeddings found")
            embed = weight_info

    for weight_info in module_def.post_weights():
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
    model: ModelReference,
    options: MergeOptions,
    arch_info: Optional[ConfiguredModelArchitecture] = None,
    get_tied: bool = False,
    device: str = "cpu",
) -> Tuple[Dict[NormalizedToken, int], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if arch_info is None:
        arch_info = get_arch_info(model, options)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.model.path,
        revision=model.model.revision,
        trust_remote_code=options.trust_remote_code,
    )
    vocab = normalized_vocabulary(tokenizer)
    embed_wi, lm_head_wi = get_embedding_info(arch_info)
    loader = LoaderCache().get(model)
    embed = loader.get_tensor(
        embed_wi.name,
        device=device,
        aliases=maybe_aliases(embed_wi, get_tied),
        raise_on_missing=not embed_wi.optional,
    )
    lm_head = loader.get_tensor(
        lm_head_wi.name,
        device=device,
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


def john_hewitt_init(orig_embed: torch.Tensor, num_new_tokens: int) -> torch.Tensor:
    orig_embed_f32 = orig_embed.to(torch.float32)
    mean = orig_embed_f32.mean(dim=0)
    centered = orig_embed_f32 - mean
    covariance = centered.T @ centered / orig_embed_f32.shape[0]
    is_pd = torch.distributions.constraints.positive_definite.check(covariance).all()
    if not is_pd:
        LOG.warning(
            "Covariance matrix is not positive definite - falling back to small randn"
        )
        return (
            torch.randn(
                len(num_new_tokens),
                orig_embed.shape[1],
                device=orig_embed.device,
                dtype=orig_embed.dtype,
            )
            * 0.02
        )
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=mean,
        covariance_matrix=covariance,
    )
    new_embeds = dist.sample((num_new_tokens,))
    return new_embeds.to(orig_embed.dtype)


def debug_reconstruction_for_random_tokens(
    coeffs: torch.Tensor,
    donor_shared_embeds: torch.Tensor,
    indices: torch.LongTensor,
    donor_embed: torch.Tensor,
    target_tokens: List[NormalizedToken],
    donor_vocab: Dict[NormalizedToken, int],
    shared_vocab: List[NormalizedToken],
    options: TokenSurgeonOptions,
):
    import random

    reconstructed_in_donor = (
        torch.bmm(
            coeffs.unsqueeze(1).to(torch.float),
            donor_shared_embeds[indices].to(torch.float),
        )
        .squeeze(1)
        .to(donor_embed.dtype)
    )
    donor_tok = transformers.AutoTokenizer.from_pretrained(
        options.donor.model.path,
        revision=options.donor.model.revision,
        trust_remote_code=False,
    )
    for i in random.sample(range(len(target_tokens)), 10):
        tok_txt = donor_tok.decode([donor_vocab[target_tokens[i]]])
        comp_tokens = [
            repr(donor_tok.decode([donor_vocab[shared_vocab[j]]])) for j in indices[i]
        ]
        comp_coeffs = coeffs[i]
        print(
            repr(tok_txt)
            + "\\approx "
            + " + ".join(
                [
                    f"{comp_coeffs[j].item():.4f} * {comp_tokens[j]}"
                    for j in range(len(comp_tokens))
                ]
            )
        )
        donor_tok_embed = donor_embed[donor_vocab[target_tokens[i]]]
        reconstructed = reconstructed_in_donor[i]
        err_rms = (reconstructed - donor_tok_embed).norm()
        err_rel = err_rms / donor_tok_embed.norm().clamp_min(1e-6)
        print(f"  Original embed norm: {donor_tok_embed.norm().item():.4f}")
        print(f"  Reconstructed embed norm: {reconstructed.norm().item():.4f}")
        print(f"  RMS error: {err_rms.item():.4f} (rel: {err_rel.item():.4f})")
        cos_sim = torch.nn.functional.cosine_similarity(
            donor_tok_embed,
            reconstructed,
            dim=0,
        )
        print(f"  Cosine similarity: {cos_sim.item():.4f}")
        print()


def compute_new_embeddings(
    orig_embed: torch.Tensor,
    donor_embed: torch.Tensor,
    orig_vocab: Dict[NormalizedToken, int],
    donor_vocab: Dict[NormalizedToken, int],
    target_tokens: List[NormalizedToken],
    is_lm_head: bool,
    options: TokenSurgeonOptions,
) -> torch.Tensor:
    assert all(t in donor_vocab for t in target_tokens)
    if options.method == ApproximationMethod.MEAN:
        mean = orig_embed.mean(dim=0)
        return mean.unsqueeze(0).expand(len(target_tokens), -1)
    elif options.method == ApproximationMethod.ZERO:
        return torch.zeros(
            len(target_tokens),
            orig_embed.shape[1],
            device=orig_embed.device,
            dtype=orig_embed.dtype,
        )
    elif options.method == ApproximationMethod.RANDN:
        return torch.randn(
            len(target_tokens),
            orig_embed.shape[1],
            device=orig_embed.device,
            dtype=orig_embed.dtype,
        )
    elif options.method == ApproximationMethod.JOHN_HEWITT:
        return john_hewitt_init(orig_embed, len(target_tokens))
    elif options.method in (
        ApproximationMethod.COMMON_INTERPOLATION,
        ApproximationMethod.ORTHOGONAL_MATCHING_PURSUIT,
    ):
        shared_vocab = list(
            sorted(
                set(orig_vocab.keys()) & set(donor_vocab.keys()),
                key=lambda x: donor_vocab[x],
            )
        )
        donor_shared_embeds = donor_embed[
            torch.tensor([donor_vocab[t] for t in shared_vocab])
        ]
        orig_shared_embeds = orig_embed[
            torch.tensor([orig_vocab[t] for t in shared_vocab])
        ]
        targets = donor_embed[torch.tensor([donor_vocab[t] for t in target_tokens])]
        if options.method == ApproximationMethod.COMMON_INTERPOLATION:
            indices, coeffs = common_interp_approximate(
                targets,
                donor_shared_embeds,
                k=options.k,
                metric=(
                    DistanceMetric.COSINE
                    if options.cosine_similarity
                    else DistanceMetric.EUCLIDEAN
                ),
                weight_scheme=options.weight_scheme,
            )
        else:
            indices, coeffs = batch_omp(targets, donor_shared_embeds, options.k)

        # for paper: choose a few random tokens and print the shared tokens and coefficients for them
        debug_reconstruction_for_random_tokens(
            coeffs,
            donor_shared_embeds,
            indices,
            donor_embed,
            target_tokens,
            donor_vocab,
            shared_vocab,
            options,
        )

        res = (
            torch.bmm(coeffs.unsqueeze(1), orig_shared_embeds[indices].to(torch.float))
            .squeeze(1)
            .to(orig_embed.dtype)
        )
        return res
    elif options.method == ApproximationMethod.SUBWORD:
        return subword_approximate(orig_embed, target_tokens, is_lm_head, options)
    else:
        raise ValueError(f"Unknown approximation method: {options.method}")


def build_embedding_matrix(
    weight_info: WeightInfo,
    orig_embed: torch.Tensor,
    donor_embed: torch.Tensor,
    orig_vocab: Dict[NormalizedToken, int],
    donor_vocab: Dict[NormalizedToken, int],
    allow_prefix: bool,
    allow_byte: bool,
    is_lm_head: bool,
    options: TokenSurgeonOptions,
) -> torch.Tensor:
    LOG.info(f"Building new tensor for {weight_info.name}")
    stats = TokenAssignmentStats()
    out_vocab_size = max(len(donor_vocab), max(donor_vocab.values()) + 1)
    res = torch.zeros(
        out_vocab_size,
        orig_embed.shape[1],
        device=orig_embed.device,
        dtype=orig_embed.dtype,
    )
    new_tokens = []
    for token, donor_idx in donor_vocab.items():
        if token in orig_vocab:
            orig_idx = orig_vocab[token]
            res[donor_idx] = orig_embed[orig_idx]
            stats.exact_match += 1
        elif (
            allow_byte and (orig_idx := match_byte_token(token, orig_vocab)) is not None
        ):
            res[donor_idx] = orig_embed[orig_idx]
            stats.byte_match += 1
        elif allow_prefix and (orig_idx := match_prefix(token, orig_vocab)) is not None:
            res[donor_idx] = orig_embed[orig_idx]
            stats.prefix_match += 1
        else:
            new_tokens.append(token)
            stats.to_approximate += 1

    LOG.info(stats.pretty_print())
    if new_tokens:
        LOG.info(f"Approximating {len(new_tokens)} tokens")
        batch_size = options.batch_size or len(new_tokens)
        for base_idx in tqdm.tqdm(
            range(0, len(new_tokens), batch_size),
            desc="Approximating tokens",
        ):
            new_embeds = compute_new_embeddings(
                orig_embed,
                donor_embed,
                orig_vocab,
                donor_vocab,
                new_tokens[base_idx : base_idx + batch_size],
                is_lm_head,
                options,
            )
            for ne_idx, token in enumerate(
                new_tokens[base_idx : base_idx + batch_size]
            ):
                res[donor_vocab[token]] = new_embeds[ne_idx]
    return res


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
    "--knn/--no-knn",
    is_flag=True,
    default=True,
    help="Use KNN for common-vocabulary interpolation",
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
    default=ApproximationMethod.ORTHOGONAL_MATCHING_PURSUIT.value,
    help="Method for approximating missing tokens",
    show_default=True,
)
@click.option(
    "--weight-scheme",
    "-w",
    type=click.Choice([w.value for w in WeightingScheme]),
    default=WeightingScheme.DISTANCE_PROPORTIONAL.value,
    help="Weighting scheme for common-vocabulary interpolation",
    show_default=True,
)
@click.option(
    "--subword-method",
    "-s",
    type=click.Choice([m.value for m in SubwordMethod]),
    default=SubwordMethod.MEAN.value,
    help="Method for approximating embeddings with subword tokens",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of tokens to process in each batch",
    show_default=True,
)
@click.option(
    "--allow-lm-head-prefix-match/--no-allow-lm-head-prefix-match",
    is_flag=True,
    default=False,
    help="Allow prefix matches for LM head tokens",
    show_default=True,
)
@add_merge_options
def main(
    model: str,
    donor: str,
    out_path: str,
    k: int,
    knn: bool,
    cosine_similarity: bool,
    approximation_method: str,
    weight_scheme: str,
    subword_method: str,
    batch_size: Optional[int],
    allow_lm_head_prefix_match: bool,
    merge_options: MergeOptions,
):
    merge_options.apply_global_options()
    logging.warning("This script is experimental and may produce unexpected results.")
    options = TokenSurgeonOptions(
        model=ModelReference.model_validate(model),
        donor=ModelReference.model_validate(donor),
        out_path=out_path,
        k=k,
        knn=knn,
        cosine_similarity=cosine_similarity,
        method=ApproximationMethod(approximation_method),
        weight_scheme=WeightingScheme(weight_scheme),
        subword_method=SubwordMethod(subword_method),
        batch_size=batch_size,
    )

    cache = LoaderCache()
    cache.setup(options=merge_options)

    device = "cuda" if merge_options.cuda else "cpu"

    arch_info = get_arch_info(options.model, merge_options)
    embed_wi, lm_head_wi = get_embedding_info(arch_info)
    orig_vocab, orig_embed, orig_lm_head = get_stuff(
        options.model, merge_options, arch_info=arch_info, device=device
    )
    donor_vocab, donor_embed, donor_lm_head = get_stuff(
        options.donor, merge_options, arch_info=None, get_tied=True, device=device
    )

    if orig_embed is not None:
        if donor_embed is None:
            raise RuntimeError(
                f"Missing tensor {embed_wi.name} in model {options.donor}"
            )
        new_embed = build_embedding_matrix(
            embed_wi,
            orig_embed,
            donor_embed,
            orig_vocab=orig_vocab,
            donor_vocab=donor_vocab,
            allow_prefix=False,
            allow_byte=True,
            is_lm_head=False,
            options=options,
        )
    else:
        if not embed_wi.optional:
            raise RuntimeError(
                f"Missing tensor {embed_wi.name} in model {options.model}"
            )
        new_embed = None

    if orig_lm_head is not None:
        if donor_lm_head is None:
            raise RuntimeError(
                f"Missing tensor {lm_head_wi.name} in model {options.donor}"
            )
        new_lm_head = build_embedding_matrix(
            lm_head_wi,
            orig_lm_head,
            donor_lm_head,
            orig_vocab=orig_vocab,
            donor_vocab=donor_vocab,
            allow_prefix=allow_lm_head_prefix_match,
            allow_byte=True,
            is_lm_head=True,
            options=options,
        )
    else:
        if not lm_head_wi.optional:
            raise RuntimeError(
                f"Missing tensor {lm_head_wi.name} in model {options.model}"
            )
        new_lm_head = None

    new_vocab_size = None
    if new_embed is not None:
        new_vocab_size = new_embed.shape[0]
    elif new_lm_head is not None:
        new_vocab_size = new_lm_head.shape[0]
    LOG.info(f"Saving new model to {out_path}")
    out_arch_info = get_out_arch_info(
        options.model, options.donor, new_vocab_size, merge_options
    )
    writer = TensorWriter(
        out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )
    for weight_info in tqdm.tqdm(out_arch_info.all_weights(), desc="Saving weights"):
        if weight_info.name == embed_wi.name:
            tensor = new_embed
        elif lm_head_wi is not None and weight_info.name == lm_head_wi.name:
            tensor = new_lm_head
        else:
            tensor = cache.get(options.model).get_tensor(
                weight_info.name, aliases=weight_info.aliases
            )
        if tensor is None:
            if weight_info.optional:
                continue
            raise RuntimeError(
                f"Missing tensor {weight_info.name} in model {options.model}"
            )
        writer.save_tensor(weight_info.name, tensor, clone=merge_options.clone_tensors)
    writer.finalize()
    out_arch_info.config.save_pretrained(out_path)

    tokenizer_out = transformers.AutoTokenizer.from_pretrained(
        options.donor.model.path,
        revision=options.donor.model.revision,
        trust_remote_code=merge_options.trust_remote_code,
    )
    tokenizer_out.save_pretrained(out_path)
    LOG.info("Done!")
