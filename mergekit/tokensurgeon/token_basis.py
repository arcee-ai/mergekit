import logging
from typing import Dict, List, Tuple

import torch

from mergekit.tokenizer.normalization import NormalizedToken
from mergekit.tokensurgeon.omp import batch_omp

LOG = logging.getLogger(__name__)


def sparse_linear_basis(
    points: torch.Tensor,
    k: int,
    d: int,
    eps: float = 1e-8,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    """
    Form an approximate orthogonal basis from sparse linear combinations of input points.
    Args:
        points: (num_pts, embed_dim) tensor of input points
        k: number of points to select per basis vector
        d: dimensionality of the basis
        eps: numerical stability parameter
    Returns:
        indices: (d, k) tensor of selected indices
        coeffs: (d, k) tensor of coefficients for each selected point
    """
    assert points.dim() == 2
    num_pts, embed_dim = points.shape
    assert k <= num_pts, "k must be less than or equal to the number of points"
    assert d <= embed_dim, "d must be less than or equal to the embedding dimension"

    mean_embed = points.mean(dim=0)
    centered_embeddings = (points - mean_embed).to(torch.float32)
    covariance_matrix = (
        centered_embeddings.T @ centered_embeddings
    ) / num_pts  # (embed_dim, embed_dim)

    U, _S, _V = torch.linalg.svd(covariance_matrix)
    U_d = U[:, :d]  # (embed_dim, d)

    # use OMP to approximate the singular vectors
    indices, coeffs = batch_omp(
        U_d.t(),  # (d, embed_dim)
        centered_embeddings,  # (num_pts, embed_dim)
        k,
        eps=eps,
    )

    if LOG.isEnabledFor(logging.DEBUG):
        rc_basis = torch.bmm(
            coeffs.unsqueeze(1).to(torch.float),
            centered_embeddings[indices].to(torch.float),
        ).squeeze(1)
        for i in range(d):
            v_0 = U_d[:, i]
            v_1 = rc_basis[i]
            cos_sim = torch.nn.functional.cosine_similarity(v_0, v_1, dim=0)
            rms = torch.norm(v_0 - v_1)
            norm_rms = torch.norm(v_0 - (v_1 / v_1.norm().clamp_min(1e-6)))
            LOG.debug(
                f"Basis vector {i}: cos_sim = {cos_sim.item():.4f}, RMS = {rms.item():.4f}, norm_rms = {norm_rms.item():.4f}"
            )

    return indices, coeffs


def compute_token_basis(
    orig_embed: torch.Tensor,
    donor_embed: torch.Tensor,
    orig_vocab: Dict[NormalizedToken, int],
    donor_vocab: Dict[NormalizedToken, int],
    junk_tokens: List[int],
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute approximately orthogonal bases for both original and donor embeddings
    as sparse linear combinations of elements.

    Args:
        orig_embed: Original embedding matrix
        donor_embed: Donor embedding matrix
        orig_vocab: Vocabulary mapping for original model
        donor_vocab: Vocabulary mapping for donor model
        junk_tokens: List of junk token indices to exclude
        k: Number of points to select per basis vector
    Returns:
        donor_basis: Approximate orthogonal basis for donor model
        orig_basis: Approximate orthogonal basis for original model
    """
    common_vocab = set(orig_vocab.keys()) & set(donor_vocab.keys())
    junk_set = set(junk_tokens)
    common_vocab = [
        tok
        for tok in common_vocab
        if (tok not in donor_vocab or donor_vocab[tok] not in junk_set)
    ]
    effective_dim = min(orig_embed.shape[1], donor_embed.shape[1])
    orig_shared_embeds = orig_embed[torch.tensor([orig_vocab[t] for t in common_vocab])]
    donor_shared_embeds = donor_embed[
        torch.tensor([donor_vocab[t] for t in common_vocab])
    ]
    if donor_embed.shape[1] < orig_embed.shape[1]:
        basis_src_embeds = donor_shared_embeds
        LOG.debug("Using donor embeds to compute token basis")
    else:
        basis_src_embeds = orig_shared_embeds
        LOG.debug("Using original embeds to compute token basis")
    LOG.debug(f"Basis dimension: {effective_dim}")
    tb_indices, tb_weights = sparse_linear_basis(
        basis_src_embeds,
        k=k,
        d=effective_dim,
    )
    donor_basis = (
        torch.bmm(
            tb_weights.unsqueeze(1).to(torch.float),
            donor_shared_embeds[tb_indices].to(torch.float),
        )
        .squeeze(1)
        .to(donor_embed.dtype)
    )
    orig_basis = (
        torch.bmm(
            tb_weights.unsqueeze(1).to(torch.float),
            orig_shared_embeds[tb_indices].to(torch.float),
        )
        .squeeze(1)
        .to(orig_embed.dtype)
    )
    return (donor_basis, orig_basis)
