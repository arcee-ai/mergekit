import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

from mergekit.moe.router import get_hidden_states


def _left_padded(sequences, pad_token_id=1):
    length = max(len(seq) for seq in sequences)
    input_ids = [[pad_token_id] * (length - len(seq)) + seq for seq in sequences]
    attention_mask = [[0] * (length - len(seq)) + [1] * len(seq) for seq in sequences]
    return BatchEncoding(
        {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }
    )


def test_hidden_avg_ignores_batch_padding():
    """A prompt's averaged hidden state must not depend on the padding that
    longer prompts in the same batch add, so the batch average is the mean of
    the prompts' own averages."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
    ).eval()
    short = [1, 5, 6, 7]
    long = [1] + list(range(10, 22))

    together = get_hidden_states(model, _left_padded([short, long]), average=True)
    separate = (
        get_hidden_states(model, _left_padded([short]), average=True)
        + get_hidden_states(model, _left_padded([long]), average=True)
    ) / 2

    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
