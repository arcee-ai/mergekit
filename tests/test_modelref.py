import pytest

from mergekit.common import ModelPath, ModelReference


class TestModelReference:
    def test_parse_simple(self):
        text = "hf_user/model"
        mr = ModelReference.parse(text)
        assert mr.model == ModelPath(path="hf_user/model", revision=None)
        assert mr.lora is None
        assert str(mr) == text

    def test_parse_lora(self):
        text = "hf_user/model+hf_user/lora"
        mr = ModelReference.parse(text)
        assert mr.model == ModelPath(path="hf_user/model", revision=None)
        assert mr.lora == ModelPath(path="hf_user/lora", revision=None)
        assert str(mr) == text

    def test_parse_revision(self):
        text = "hf_user/model@v0.0.1"
        mr = ModelReference.parse(text)
        assert mr.model == ModelPath(path="hf_user/model", revision="v0.0.1")
        assert mr.lora is None
        assert str(mr) == text

    def test_parse_lora_plus_revision(self):
        text = "hf_user/model@v0.0.1+hf_user/lora@main"
        mr = ModelReference.parse(text)
        assert mr.model == ModelPath(path="hf_user/model", revision="v0.0.1")
        assert mr.lora == ModelPath(path="hf_user/lora", revision="main")
        assert str(mr) == text

    def test_parse_bad(self):
        with pytest.raises(RuntimeError):
            ModelReference.parse("@@@@@")

        with pytest.raises(RuntimeError):
            ModelReference.parse("a+b+c")

        with pytest.raises(RuntimeError):
            ModelReference.parse("a+b+c@d+e@f@g")
