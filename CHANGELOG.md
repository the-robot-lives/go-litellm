# Changelog

Dated work log, newest first. One line per landed change, extended in place as it progresses (`→ stage`, `→ main`).

## 2026-10-13
- feat(providers): gemini + openrouter adapters — default bases (generativelanguage.googleapis.com/v1beta/openai, openrouter.ai/api/v1) honor per-entry api_base overrides; openai-compatible adapters now forward litellm_params.extra_headers (openrouter HTTP-Referer/X-Title, z.ai anthropic-product) (#3)
- feat(core): honor litellm_params.num_retries (default 2, clamp 0–8) with backoff on transport errors and 408/429/5xx; streaming requests apply stream_timeout/timeout caps (#3)
- fix(gateway): /v1/messages strips additional_drop_params (context_management/thinking/reasoning_effort) from upstream bodies for z.ai (sub) and Alibaba token-plan anthropic deployments; keeps the mapped z.ai/wafer reasoning_effort when the entry relies on it (#3)
- test: production litellm-config.yaml copied verbatim to testdata/ — 184 entries, duplicate model_name groups (whisper-1, zai/glm-5.3-flash), per-family routing/URL/header/timeout assertions (#3)
- test: fix pre-existing go vet failures in gateway tests (response errors checked before use) (#3)
