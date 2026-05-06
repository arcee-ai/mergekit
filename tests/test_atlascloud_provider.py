from mergekit.atlascloud import AtlasCloudConfig, extract_message_text, normalize_base_url


def test_config_from_env_file(tmp_path):
    env_file = tmp_path / ".env.atlascloud.local"
    env_file.write_text(
        "\n".join(
            [
                "ATLASCLOUD_API_KEY=apikey-test",
                "ATLASCLOUD_BASE_URL=https://api.atlascloud.ai/v1",
                "ATLASCLOUD_MODEL=deepseek-ai/DeepSeek-V3.1",
            ]
        ),
        encoding="utf-8",
    )

    config = AtlasCloudConfig.from_env(env={}, env_file=str(env_file))

    assert config.api_key == "apikey-test"
    assert config.base_url == "https://api.atlascloud.ai/v1"
    assert config.model == "deepseek-ai/DeepSeek-V3.1"


def test_config_prefers_explicit_env_over_file(tmp_path):
    env_file = tmp_path / ".env.atlascloud.local"
    env_file.write_text("ATLASCLOUD_API_KEY=apikey-file", encoding="utf-8")

    config = AtlasCloudConfig.from_env(
        env={"ATLASCLOUD_API_KEY": "apikey-env"},
        env_file=str(env_file),
    )

    assert config.api_key == "apikey-env"


def test_normalize_base_url_variants():
    assert normalize_base_url("https://api.atlascloud.ai") == "https://api.atlascloud.ai/v1"
    assert (
        normalize_base_url("https://api.atlascloud.ai/v1")
        == "https://api.atlascloud.ai/v1"
    )
    assert (
        normalize_base_url("https://api.atlascloud.ai/api/v1")
        == "https://api.atlascloud.ai/v1"
    )


def test_extract_message_text_from_string_content():
    payload = {"choices": [{"message": {"content": "pong"}}]}

    assert extract_message_text(payload) == "pong"


def test_extract_message_text_from_structured_content():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": " world"},
                    ]
                }
            }
        ]
    }

    assert extract_message_text(payload) == "hello world"
