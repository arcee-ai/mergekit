import torch
import unittest
from unittest.mock import MagicMock
from mergekit.moe.router import get_gate_params


class TestOrthogonalInit(unittest.TestCase):
    def test_gate_orthogonality(self):
        # Mock Config
        mock_cfg = MagicMock()
        mock_cfg.num_hidden_layers = 2
        mock_cfg.hidden_size = 128
        mock_experts = [MagicMock()] * 4  # 4 experts

        gates = get_gate_params(
            model_cfg=mock_cfg, experts=mock_experts, mode="orthogonal"
        )

        for i in range(mock_cfg.num_hidden_layers):
            layer_gate = gates[i].float()
            # Q * Q^T should be Identity matrix
            prod = torch.matmul(layer_gate, layer_gate.t())
            identity = torch.eye(len(mock_experts))

            # Use high tolerance for float16 artifacts if you casted
            self.assertTrue(torch.allclose(prod, identity, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
