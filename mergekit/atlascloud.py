from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional
from urllib import error, request


DEFAULT_BASE_URL = "https://api.atlascloud.ai/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.1"
DEFAULT_ENV_FILE = ".env.atlascloud.local"


def _parse_env_file(env_file: Optional[str]) -> dict[str, str]:
    if not env_file:
        return {}

    path = Path(env_file).expanduser()
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


@dataclass(frozen=True)
class AtlasCloudConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        env_file: Optional[str] = None,
    ) -> "AtlasCloudConfig":
        merged_env: dict[str, str] = {}
        merged_env.update(_parse_env_file(env_file))
        merged_env.update(dict(os.environ if env is None else env))

        api_key = merged_env.get("ATLASCLOUD_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing ATLASCLOUD_API_KEY. Set it in the environment or in "
                f"{env_file or DEFAULT_ENV_FILE}."
            )

        base_url = merged_env.get("ATLASCLOUD_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        model = merged_env.get("ATLASCLOUD_MODEL", DEFAULT_MODEL)
        return cls(api_key=api_key, base_url=base_url, model=model)


def extract_message_text(payload: Mapping) -> str:
    choices = payload.get("choices")
    if not choices:
        raise ValueError("AtlasCloud response does not contain choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        if text_parts:
            return "".join(text_parts)

    raise ValueError("AtlasCloud response does not contain a text message.")


def chat_completion(
    config: AtlasCloudConfig,
    prompt: str,
    timeout: int = 60,
) -> dict:
    payload = json.dumps(
        {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    req = request.Request(
        url=f"{config.base_url}/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "mergekit-atlascloud-test/0.1",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"AtlasCloud request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"AtlasCloud request failed: {exc.reason}") from exc
