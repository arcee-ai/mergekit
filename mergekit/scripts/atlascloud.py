from __future__ import annotations

import json
from pathlib import Path

import click

from mergekit.atlascloud import (
    DEFAULT_ENV_FILE,
    AtlasCloudConfig,
    chat_completion,
    extract_message_text,
)


@click.command("mergekit-atlascloud-test")
@click.option(
    "--env-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_ENV_FILE,
    show_default=True,
    help="Path to the local AtlasCloud credentials file.",
)
@click.option(
    "--prompt",
    type=str,
    default="Reply with the single word: pong",
    show_default=True,
    help="Prompt used for the smoke test.",
)
@click.option(
    "--system-prompt",
    type=str,
    default="You are a concise API smoke test assistant.",
    show_default=True,
    help="Optional system prompt.",
)
@click.option("--model", type=str, help="Override the AtlasCloud model name.")
@click.option("--base-url", type=str, help="Override the AtlasCloud base URL.")
@click.option("--timeout", type=int, default=60, show_default=True)
@click.option(
    "--show-json/--no-show-json",
    is_flag=True,
    default=False,
    help="Print the raw JSON response after the message text.",
)
def main(
    env_file: Path,
    prompt: str,
    system_prompt: str,
    model: str | None,
    base_url: str | None,
    timeout: int,
    show_json: bool,
):
    config = AtlasCloudConfig.from_env(env_file=str(env_file))
    if model:
        config = AtlasCloudConfig(
            api_key=config.api_key,
            base_url=config.base_url,
            model=model,
        )
    if base_url:
        config = AtlasCloudConfig(
            api_key=config.api_key,
            base_url=base_url.rstrip("/"),
            model=config.model,
        )

    response = chat_completion(
        config=config,
        prompt=prompt,
        system_prompt=system_prompt,
        timeout=timeout,
    )
    click.echo(extract_message_text(response))
    if show_json:
        click.echo(json.dumps(response, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
