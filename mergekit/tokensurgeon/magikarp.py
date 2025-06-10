import logging
from typing import Dict, List, Optional

import torch

from mergekit.tokenizer.normalization import NormalizedToken, unnormalize_token

LOG = logging.getLogger(__name__)


def well_trained_tokens(
    vocab: Dict[NormalizedToken, int],
    embed: torch.Tensor,
    lm_head: Optional[torch.Tensor],
    known_unused: Optional[List[NormalizedToken]] = None,
    quantile: float = 0.01,
) -> List[NormalizedToken]:
    """Get a list of tokens that are well-trained in the model.

    Uses the approach from "Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models"
    (https://arxiv.org/abs/2405.05417).

    Args:
        vocab: The vocabulary of the model, mapping tokens to indices.
        embed: The input embedding matrix of the model.
        lm_head: The output embedding matrix of the model (optional).
        known_unused: A list of known unused tokens (optional).
        quantile: The quantile to use for filtering (default: 0.01).

    Returns:
        A list of tokens that can be assumed to be well-trained in the model.
    """
    unused_indices = set(range(embed.shape[0])) - set(vocab.values())
    if known_unused:
        unused_indices.update(vocab[tok] for tok in known_unused if tok in vocab)
    for tok in vocab:
        tok_text = unnormalize_token(tok)
        if "unused_token" in tok_text or "reserved_special_token" in tok_text:
            LOG.debug(f"Assuming {tok_text} is unused")
            unused_indices.add(vocab[tok])

    if unused_indices:
        mean_unused_in = embed[list(unused_indices)].mean(dim=0)
        mean_unused_out = (
            lm_head[list(unused_indices)].mean(dim=0) if lm_head is not None else None
        )
        LOG.info(f"Found {len(unused_indices)} unused tokens")
    else:
        mean_unused_in = None
        mean_unused_out = None

    bad_indices = set(unused_indices)

    if lm_head is not None:
        # check L2 norm of input embeddings
        l2_norms = embed.norm(dim=1).float()
        threshold = torch.quantile(l2_norms, quantile, dim=0)
        LOG.debug(
            f"Unused token L2 norm threshold: {threshold.item():.4f} ({int(quantile * 100)}th percentile)"
        )
        l2_bad_indices = torch.where(l2_norms < threshold)[0]
        if len(l2_bad_indices) > 0:
            bad_indices.update(l2_bad_indices.tolist())
            LOG.info(f"Discarding {len(l2_bad_indices)} low-l2 tokens")

    if mean_unused_in is not None:
        # check cosine similarity of input embeddings
        cos_sim = torch.nn.functional.cosine_similarity(
            embed, mean_unused_in.unsqueeze(0), dim=1
        ).float()
        threshold = torch.quantile(cos_sim, 1 - quantile, dim=0)
        LOG.debug(
            f"Unused token threshold in embed_tokens: {threshold.item():.4f} ({int((1 - quantile) * 100)}th percentile)"
        )
        if threshold < 0.5:
            threshold = 0.5
            LOG.debug("Clamping threshold to 0.5")
        cos_bad_indices = torch.where(cos_sim > threshold)[0]
        if len(cos_bad_indices) > 0:
            bad_indices.update(cos_bad_indices.tolist())
            LOG.info(
                f"Discarding {len(cos_bad_indices)} high-sim to unused mean tokens"
            )

    if lm_head is not None and mean_unused_out is not None:
        # check cosine similarity of output embeddings
        cos_sim = torch.nn.functional.cosine_similarity(
            lm_head, mean_unused_out.unsqueeze(0), dim=1
        ).float()
        threshold = torch.quantile(cos_sim, 1 - quantile, dim=0)
        LOG.debug(
            f"Unused token threshold in lm_head: {threshold.item():.4f} ({int((1 - quantile) * 100)}th percentile)"
        )
        if threshold < 0.5:
            threshold = 0.5
            LOG.debug("Clamping threshold to 0.5")
        cos_bad_indices = torch.where(cos_sim > threshold)[0]
        if len(cos_bad_indices) > 0:
            bad_indices.update(cos_bad_indices.tolist())
            LOG.info(
                f"Discarding {len(cos_bad_indices)} high-sim to unused mean tokens"
            )

    good_tokens = [tok for tok, idx in vocab.items() if idx not in bad_indices]
    LOG.info(
        f"Found {len(good_tokens)} well-trained tokens, {len(bad_indices)} bad tokens"
    )
    return good_tokens
