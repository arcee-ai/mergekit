from typing import Dict, Optional

import pytest
from common import run_and_check_merge
from transformers import (
    AutoConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from mergekit.config import (
    InputModelDefinition,
    MergeConfiguration,
    ParameterSetting,
)


def make_mini_whisper(path: str, vocab_size: int = 64):
    """Create a minimal Whisper model for testing."""
    cfg = WhisperConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        d_model=32,
        decoder_ffn_dim=64,
        encoder_ffn_dim=64,
        decoder_start_token_id=1,
        pad_token_id=1,
        bos_token_id=1,
        eos_token_id=2,
        suppress_tokens=[],
    )
    model = WhisperForConditionalGeneration(cfg)
    
    # Ensure the model has the proj_out.weight tensor
    if not hasattr(model, "proj_out") or not hasattr(model.proj_out, "weight"):
        raise ValueError("Model does not have proj_out.weight tensor")
    
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture(scope="session")
def whisper_a(tmp_path_factory):
    return make_mini_whisper(tmp_path_factory.mktemp("whisper_a"))


@pytest.fixture(scope="session")
def whisper_b(tmp_path_factory):
    return make_mini_whisper(tmp_path_factory.mktemp("whisper_b"))


@pytest.fixture(scope="session")
def whisper_c(tmp_path_factory):
    return make_mini_whisper(tmp_path_factory.mktemp("whisper_c"))


class TestWhisperMerges:
    def test_whisper_copy(self, whisper_a):
        """Test copying a Whisper model."""
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model=whisper_a)],
            dtype="bfloat16",
        )
        run_and_check_merge(config)

    def test_whisper_linear_merge(self, whisper_a, whisper_b):
        """Test linear merging of Whisper models."""
        config = self.two_model_config(whisper_a, whisper_b, merge_method="linear")
        run_and_check_merge(config)

    def test_whisper_encoder_decoder_weighted_merge(self, whisper_a, whisper_b):
        """Test encoder-decoder weighted merging of Whisper models."""
        config = MergeConfiguration(
            merge_method="encoder_decoder_weighted",
            models=[
                InputModelDefinition(
                    model=whisper_a,
                    parameters={
                        "encoder_weight": 0.8,
                        "decoder_weight": 0.3,
                    },
                ),
                InputModelDefinition(
                    model=whisper_b,
                    parameters={
                        "encoder_weight": 0.2,
                        "decoder_weight": 0.7,
                    },
                ),
            ],
            dtype="bfloat16",
        )
        
        def _check_config(p: str):
            """Verify the model config is correct."""
            config = AutoConfig.from_pretrained(p)
            assert config.model_type == "whisper"
            assert config.encoder_layers == 2
            assert config.decoder_layers == 2
        
        run_and_check_merge(config, validate=_check_config)

    def two_model_config(
        self,
        model_a,
        model_b,
        merge_method: str,
        base_model: Optional[str] = None,
        params: Optional[Dict[str, ParameterSetting]] = None,
    ):
        """Create a configuration for merging two models."""
        config = MergeConfiguration(
            merge_method=merge_method,
            base_model=base_model,
            models=[
                InputModelDefinition(
                    model=model_a,
                    parameters={"weight": 0.6},
                ),
                InputModelDefinition(
                    model=model_b,
                    parameters={"weight": 0.4},
                ),
            ],
            dtype="bfloat16",
            parameters=params,
        )

        return config 