import torch
import unittest
from unittest.mock import MagicMock
from mergekit.moe.router import get_gate_params

class TestMoEOrthogonal(unittest.TestCase):
    def test_gate_orthogonality(self):
        # 1. Setup Mock Environment
        mock_cfg = MagicMock()
        mock_cfg.num_hidden_layers = 2
        mock_cfg.hidden_size = 256
        mock_experts = [MagicMock()] * 8  # 8 experts
        
        # 2. Call your new implementation
        gates = get_gate_params(
            model_cfg=mock_cfg, 
            experts=mock_experts, 
            mode="orthogonal",
            out_dtype=torch.float32 # Use float32 for high precision check
        )
        
        # 3. Verify mathematical properties
        for layer_idx in range(mock_cfg.num_hidden_layers):
            layer_gate = gates[layer_idx]
            # For orthogonal matrices, Q @ Q.T = Identity
            # Note: Since num_experts < hidden_size, it's a semi-orthogonal matrix
            product = torch.matmul(layer_gate, layer_gate.t())
            identity = torch.eye(len(mock_experts))
            
            # Check if product is close to Identity
            self.assertTrue(torch.allclose(product, identity, atol=1e-5))

if __name__ == "__main__":
    unittest.main()