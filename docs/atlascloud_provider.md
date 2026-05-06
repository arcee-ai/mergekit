# AtlasCloud Provider

This repository does not natively depend on a hosted inference provider for its
core merge flow, but it can still keep a project-local AtlasCloud setup for API
validation and smoke testing.

AtlasCloud's LLM endpoint is OpenAI-compatible, so the provider settings for
this project are:

- Base URL: `https://api.atlascloud.ai/v1`
- Chat endpoint: `https://api.atlascloud.ai/v1/chat/completions`
- Default model: `deepseek-ai/DeepSeek-V3.1`
- Local credentials file: `.env.atlascloud.local`

## Local Setup

1. Copy the example file:

   ```bash
   cp .env.atlascloud.example .env.atlascloud.local
   ```

2. Fill in the API key:

   ```bash
   ATLASCLOUD_API_KEY=your-key
   ATLASCLOUD_BASE_URL=https://api.atlascloud.ai/v1
   ATLASCLOUD_MODEL=deepseek-ai/DeepSeek-V3.1
   ```

3. Install the project:

   ```bash
   pip install -e .[test]
   ```

## Smoke Test

Run the provider smoke test:

```bash
mergekit-atlascloud-test --env-file .env.atlascloud.local
```

Override the model or endpoint if needed:

```bash
mergekit-atlascloud-test \
  --env-file .env.atlascloud.local \
  --model deepseek-ai/DeepSeek-V3.1 \
  --base-url https://api.atlascloud.ai/v1 \
  --show-json
```

## Notes

- `.env.atlascloud.local` is git-ignored and intended for local credentials only.
- The smoke test uses the OpenAI-compatible AtlasCloud chat endpoint with a
  minimal prompt and does not change any core merge behavior.
