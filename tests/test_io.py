import os
import tempfile

import torch

from mergekit.io import TensorWriter


class TestTensorWriter:
    def test_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, safe_serialization=True)
            writer.save_tensor("steve", torch.randn(4))
            writer.finalize()

            assert os.path.exists(os.path.join(d, "model.safetensors"))

    def test_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, safe_serialization=False)
            writer.save_tensor("timothan", torch.randn(4))
            writer.finalize()

            assert os.path.exists(os.path.join(d, "pytorch_model.bin"))

    def test_duplicate_tensor(self):
        with tempfile.TemporaryDirectory() as d:
            writer = TensorWriter(d, safe_serialization=True)
            jim = torch.randn(4)
            writer.save_tensor("jim", jim)
            writer.save_tensor("jimbo", jim)
            writer.finalize()

            assert os.path.exists(os.path.join(d, "model.safetensors"))
