import torch

from mergekit.io import LazyTensorLoader


class TestLazyUnpickle:
    def test_lazy_unpickle(self, tmp_path):
        data = {
            "a": torch.tensor([1, 2, 3]),
            "b": torch.tensor([4, 5, 6]),
        }
        path = tmp_path / "pytorch_model.bin"
        torch.save(data, path)
        loader = LazyTensorLoader.from_disk(tmp_path)
        for name in data:
            assert name in loader.index.tensor_paths
            tensor = loader.get_tensor(name)
            assert torch.equal(tensor, data[name])
