import enum
import logging
from typing import Dict, Generator, List, Tuple, Union

import transformers
from typing_extensions import TypeAlias

LOG = logging.getLogger(__name__)


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


def unnormalize_token(token: NormalizedToken) -> str:
    if isinstance(token, tuple):
        if token[0] == TokenMarker.WORD_START:
            return " " + token[1]
        return token[1]
    return token


def token_prefixes(
    token: NormalizedToken, allow_whitespace: bool = False
) -> Generator[NormalizedToken, None, None]:
    """Yield potential prefixes of a token."""
    marker = None
    if isinstance(token, tuple):
        marker, token = token

    if marker == TokenMarker.SPECIAL:
        # special tokens have no prefixes
        return
    for i in range(len(token) - 1, 0, -1):
        prefix = token[:i]
        if not allow_whitespace and not prefix.strip():
            break
        if marker is not None:
            yield (marker, prefix)
        else:
            yield prefix


def normalized_vocabulary(
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> Dict[NormalizedToken, int]:
    """
    Get a normalized vocabulary for a tokenizer.

    Attempts to handle word start prefixes and special tokens in a consistent way.
    """
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

    return {
        normalize_token(
            token,
            special_tokens_map=tokenizer.special_tokens_map,
            word_start_prefix=word_start_prefix,
        ): i
        for token, i in vocab.items()
    }
