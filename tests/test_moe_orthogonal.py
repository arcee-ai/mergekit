import torch
import unittest
from unittest.mock import MagicMock
from mergekit.moe.router import get_gate_params


class TestMoEOrthogonal(unittest.TestCase):
    def test_gate_orthogonality(self):
        mock_cfg = MagicMock()
        mock_cfg.num_hidden_layers = 2
        mock_cfg.hidden_size = 512
        mock_cfg.torch_dtype = torch.float32  # Force float32 for the test precision

        mock_model_ref = MagicMock()
        mock_model_ref.config.return_value = mock_cfg

        mock_tokenizer = MagicMock()
        mock_experts = [MagicMock()] * 8

        # Call the function
        gates = get_gate_params(
            model_ref=mock_model_ref,
            tokenizer=mock_tokenizer,
            experts=mock_experts,
            mode="orthogonal",
            device="cpu",
        )

        # gates is a list of tensors
        self.assertEqual(len(gates), mock_cfg.num_hidden_layers)

        for layer_gate in gates:
            # Check orthogonality: Q @ Q.T = I
            g = layer_gate.float()
            product = torch.matmul(g, g.t())
            identity = torch.eye(len(mock_experts))

            self.assertTrue(torch.allclose(product, identity, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
