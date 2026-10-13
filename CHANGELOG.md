# Changelog

Dated work log, newest first. One line per landed change, extended in place as it progresses (`→ stage`, `→ main`).

## 2026-10-13
- fix(gateway): audio-route process crash — /v1/audio/* (multipart transcriptions et al) fell to front-proxy passthrough whose LiteLLM target resolves to the gateway itself, self-forwarding ~7k nested requests per call until the pod died (production abort, no panic trace); audio routes now answer a clean authenticated 501 and are never forwarded (#6)
- fix(gateway): refuse ANY front-proxy forward whose target resolves to this gateway (seed rules or custom /front/rules) with the 502 upstream-error envelope — systemic guard against self-forward loops (#6)
- feat(gateway): global panic-recovery middleware — any handler panic is a structured 500 JSON (OpenAI envelope), logged method/path/panic only (no bodies/keys), server keeps serving; regression suite proves multipart audio 501 + subsequent /v1/models 200 (#6)
- fix(gateway): SECURITY — /v1/messages master-key auth: non-claude-* models (the LiteLLM surface, MatchMessagesNotModel → AuthMasterKey) now 401 without/with a wrong key when LITELLM_MASTER_KEY is set, matching /v1/chat/completions; claude-* stays AuthPassthrough (client's own token forwarded) so run-claude's front-proxy OAuth flow is untouched (#4)
- test: production-config suite no longer depends on ambient provider keys — z.ai (sub) ValidateEnvironment pins a dummy ANTHROPIC_API_KEY; full suite green in a scrubbed env (no live keys required anywhere) (#4)
- ci: image workflow — PRs run test-and-build (vet + tests + image build, no push), pushes to main publish ghcr.io/the-robot-lives/go-litellm:{sha-<short>,latest} (#2)
- feat(providers): gemini + openrouter adapters — default bases (generativelanguage.googleapis.com/v1beta/openai, openrouter.ai/api/v1) honor per-entry api_base overrides; openai-compatible adapters now forward litellm_params.extra_headers (openrouter HTTP-Referer/X-Title, z.ai anthropic-product) (#3)
- feat(core): honor litellm_params.num_retries (default 2, clamp 0–8) with backoff on transport errors and 408/429/5xx; streaming requests apply stream_timeout/timeout caps (#3)
- fix(gateway): /v1/messages strips additional_drop_params (context_management/thinking/reasoning_effort) from upstream bodies for z.ai (sub) and Alibaba token-plan anthropic deployments; keeps the mapped z.ai/wafer reasoning_effort when the entry relies on it (#3)
- test: production litellm-config.yaml copied verbatim to testdata/ — 184 entries, duplicate model_name groups (whisper-1, zai/glm-5.3-flash), per-family routing/URL/header/timeout assertions (#3)
- test: fix pre-existing go vet failures in gateway tests (response errors checked before use) (#3)
