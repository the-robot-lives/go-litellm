package core

// Compatibility suite over the REAL production litellm config
// (testdata/litellm-config.yaml, copied verbatim from
// terraform/kubernetes/platform/ai/files/open-webui/litellm-config.yaml).
// The proxy behind inference.noizu.com must consume this file unmodified.

import (
	"os"
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/config"
	"github.com/noizu-labs/go-litellm/internal/providers"
	"github.com/noizu-labs/go-litellm/internal/router"
)

func prodConfig(t *testing.T) *config.Config {
	t.Helper()
	cfg, err := config.LoadFile("../../testdata/litellm-config.yaml")
	if err != nil {
		t.Fatalf("load production config: %v", err)
	}
	return cfg
}

func entriesWhere(cfg *config.Config, pred func(name, model, base string) bool) []map[string]any {
	var out []map[string]any
	for _, d := range cfg.ModelList {
		lp, _ := d["litellm_params"].(map[string]any)
		name, _ := d["model_name"].(string)
		model, _ := lp["model"].(string)
		base, _ := lp["api_base"].(string)
		if pred(name, model, base) {
			out = append(out, d)
		}
	}
	return out
}

func TestProductionConfigLoadsVerbatim(t *testing.T) {
	cfg := prodConfig(t)
	if len(cfg.ModelList) != 187 {
		t.Fatalf("entries=%d want 187", len(cfg.ModelList))
	}
	rt := router.New(cfg)
	names := rt.ModelNames()
	if len(names) != 185 { // whisper-1 and zai/glm-5.3-flash each appear twice
		t.Fatalf("distinct model names=%d want 185", len(names))
	}
	if got := rt.Group("whisper-1"); len(got) != 2 {
		t.Fatalf("whisper-1 deployments=%d want 2 (duplicate model_name must co-exist)", len(got))
	}
	if got := rt.Group("zai/glm-5.3-flash"); len(got) != 2 {
		t.Fatalf("zai/glm-5.3-flash deployments=%d want 2", len(got))
	}
}

func TestProductionConfigEveryEntryClassifies(t *testing.T) {
	cfg := prodConfig(t)
	for _, d := range cfg.ModelList {
		name, _ := d["model_name"].(string)
		lp, _ := d["litellm_params"].(map[string]any)
		model, _ := lp["model"].(string)
		res, e := providers.Resolve(model, lp)
		if e != nil {
			t.Fatalf("%s (%s): %v", name, model, e)
		}
		if res.Adapter == nil {
			t.Fatalf("%s: nil adapter", name)
		}
	}
}

func TestProductionGeminiEntries(t *testing.T) {
	cfg := prodConfig(t)
	entries := entriesWhere(cfg, func(_, model, _ string) bool {
		return strings.HasPrefix(model, "gemini/")
	})
	if len(entries) != 6 {
		t.Fatalf("gemini entries=%d want 6", len(entries))
	}
	t.Setenv("GEMINI_API_KEY", "test-dummy-gemini")
	for _, d := range entries {
		lp, _ := d["litellm_params"].(map[string]any)
		model, _ := lp["model"].(string)
		res, e := providers.Resolve(model, lp)
		if e != nil {
			t.Fatal(e)
		}
		if res.Provider != "gemini" {
			t.Fatalf("%s: provider=%s", model, res.Provider)
		}
		req := &providers.Request{Model: res.Model, Provider: res.Provider, LiteLLMParams: lp, CallType: "chat"}
		if got := res.Adapter.CompleteURL(req); got != "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" {
			t.Fatalf("%s url=%s", model, got)
		}
	}
}

func TestProductionOpenRouterEntries(t *testing.T) {
	cfg := prodConfig(t)
	entries := entriesWhere(cfg, func(_, model, _ string) bool {
		return strings.HasPrefix(model, "openrouter/")
	})
	if len(entries) != 37 {
		t.Fatalf("openrouter entries=%d want 37", len(entries))
	}
	for _, d := range entries {
		lp, _ := d["litellm_params"].(map[string]any)
		model, _ := lp["model"].(string)
		res, e := providers.Resolve(model, lp)
		if e != nil {
			t.Fatal(e)
		}
		if res.Provider != "openrouter" {
			t.Fatalf("%s: provider=%s", model, res.Provider)
		}
		req := &providers.Request{Model: res.Model, Provider: res.Provider, LiteLLMParams: lp, CallType: "chat"}
		if got := res.Adapter.CompleteURL(req); got != "https://openrouter.ai/api/v1/chat/completions" {
			t.Fatalf("%s url=%s", model, got)
		}
	}
}

func TestProductionZaiSubEntries(t *testing.T) {
	// ValidateEnvironment checks litellm_params.api_key (an os.environ/ ref that
	// hydrates empty without ambient keys) and falls back to ANTHROPIC_API_KEY.
	// The suite must pass keyless: pin a dummy so the fallback resolves without
	// ever requiring or using a live key.
	t.Setenv("ANTHROPIC_API_KEY", "test-dummy-not-real")
	cfg := prodConfig(t)
	entries := entriesWhere(cfg, func(_, _, base string) bool {
		return base == "https://api.z.ai/api/anthropic"
	})
	if len(entries) != 11 {
		t.Fatalf("zai (sub) entries=%d want 11", len(entries))
	}
	for _, d := range entries {
		lp, _ := d["litellm_params"].(map[string]any)
		model, _ := lp["model"].(string)
		res, e := providers.Resolve(model, lp)
		if e != nil {
			t.Fatal(e)
		}
		if res.Provider != "anthropic" {
			t.Fatalf("%s: provider=%s", model, res.Provider)
		}
		req := &providers.Request{Model: res.Model, Provider: res.Provider, LiteLLMParams: lp, CallType: "chat"}
		if got := res.Adapter.CompleteURL(req); got != "https://api.z.ai/api/anthropic/v1/messages" {
			t.Fatalf("%s url=%s", model, got)
		}
		if !providers.ZaiThinkingApplies(lp) {
			t.Fatalf("%s: zai thinking mapping should apply", model)
		}
		extra, _ := lp["extra_headers"].(map[string]any)
		if extra["anthropic-product"] != "claude_code" {
			t.Fatalf("%s extra_headers=%v", model, extra)
		}
		headers, e := res.Adapter.ValidateEnvironment(req, map[string]string{})
		if e != nil {
			t.Fatal(e)
		}
		if headers["anthropic-product"] != "claude_code" {
			t.Fatalf("%s: header lost: %+v", model, headers)
		}
	}
}

func TestProductionAlibabaTokenPlanEntries(t *testing.T) {
	cfg := prodConfig(t)
	entries := entriesWhere(cfg, func(_, _, base string) bool {
		return strings.Contains(base, "maas.aliyuncs.com")
	})
	if len(entries) != 31 {
		t.Fatalf("alibaba token-plan entries=%d want 31", len(entries))
	}
	for _, d := range entries {
		lp, _ := d["litellm_params"].(map[string]any)
		model, _ := lp["model"].(string)
		res, e := providers.Resolve(model, lp)
		if e != nil {
			t.Fatal(e)
		}
		if res.Provider != "anthropic" {
			t.Fatalf("%s: provider=%s", model, res.Provider)
		}
		req := &providers.Request{Model: res.Model, Provider: res.Provider, LiteLLMParams: lp, CallType: "chat"}
		want := strings.TrimSuffix(lp["api_base"].(string), "/") + "/v1/messages"
		if got := res.Adapter.CompleteURL(req); got != want {
			t.Fatalf("%s url=%s want %s", model, got, want)
		}
	}
}

func TestProductionCerebrasTimeoutAndRetries(t *testing.T) {
	cfg := prodConfig(t)
	entries := entriesWhere(cfg, func(_, _, base string) bool {
		return base == "https://api.cerebras.ai/v1"
	})
	if len(entries) != 8 {
		t.Fatalf("cerebras entries=%d want 8", len(entries))
	}
	for _, d := range entries {
		lp, _ := d["litellm_params"].(map[string]any)
		name, _ := d["model_name"].(string)
		if got := timeoutOf(lp, 0); got.String() != "2m0s" {
			t.Fatalf("%s timeout=%s", name, got)
		}
		if got := retriesOf(lp); got != 4 {
			t.Fatalf("%s num_retries=%d", name, got)
		}
		if got := streamTimeoutOf(lp); got.String() != "2m0s" {
			t.Fatalf("%s stream_timeout=%s", name, got)
		}
	}
}

func TestProductionOpenAICompatibleCustomBases(t *testing.T) {
	cfg := prodConfig(t)
	cases := map[string]int{
		"https://open.bigmodel.cn/api/paas/v4": 27,
		"https://api.cerebras.ai/v1":           8,
		"http://lmstudio-proxy:3713/v1":        1,
	}
	for base, want := range cases {
		entries := entriesWhere(cfg, func(_, _, b string) bool { return b == base })
		if len(entries) != want {
			t.Fatalf("%s entries=%d want %d", base, len(entries), want)
		}
		for _, d := range entries {
			lp, _ := d["litellm_params"].(map[string]any)
			model, _ := lp["model"].(string)
			res, e := providers.Resolve(model, lp)
			if e != nil {
				t.Fatalf("%s: %v", model, e)
			}
			if res.Provider != "openai" {
				t.Fatalf("%s: provider=%s", model, res.Provider)
			}
			req := &providers.Request{Model: res.Model, Provider: res.Provider, LiteLLMParams: lp, CallType: "chat"}
			if got := res.Adapter.CompleteURL(req); got != base+"/chat/completions" {
				t.Fatalf("%s url=%s", model, got)
			}
		}
	}
}

func TestProductionEnvHydrationResolvesAtLoad(t *testing.T) {
	// api_key entries are os.environ/ refs; after hydration they resolve to
	// the live env value (fake values here — never real secrets).
	t.Setenv("ZAI_API_KEY", "test-dummy-zai")
	t.Setenv("XAI_API_KEY", "test-dummy-xai")
	cfg := prodConfig(t)
	entries := entriesWhere(cfg, func(name, _, _ string) bool { return name == "zai/glm-5" })
	if len(entries) == 0 {
		t.Fatal("zai/glm-5 (bigmodel) entry missing")
	}
	lp, _ := entries[0]["litellm_params"].(map[string]any)
	req := &providers.Request{Model: "glm-5", Provider: "openai", LiteLLMParams: lp, CallType: "chat"}
	headers, e := providers.OpenAI.ValidateEnvironment(req, map[string]string{})
	if e != nil {
		t.Fatal(e)
	}
	if headers["authorization"] != "Bearer test-dummy-zai" {
		t.Fatalf("authorization=%q", headers["authorization"])
	}
}

func TestPrepareAgainstEveryProductionEntry(t *testing.T) {
	// Chat-shaped Prepare must succeed for every chat-shaped deployment (the
	// whisper/audio entries are STT and out of scope for chat completion).
	// Upstream keys stay unresolved (no live env here) — Prepare must not
	// require them; only ValidateEnvironment at call time does.
	cfg := prodConfig(t)
	rt := router.New(cfg)
	skipped := 0
	for _, d := range cfg.ModelList {
		name, _ := d["model_name"].(string)
		lp, _ := d["litellm_params"].(map[string]any)
		model, _ := lp["model"].(string)
		if strings.Contains(model, "whisper") {
			skipped++
			continue
		}
		body := map[string]any{
			"model":    name,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		}
		p, e := Prepare(rt, cfg, body)
		if e != nil {
			t.Fatalf("%s (%s): %v", name, model, e)
		}
		if p.Req == nil {
			t.Fatalf("%s: nil request", name)
		}
	}
	if skipped != 2 {
		t.Fatalf("skipped=%d want 2", skipped)
	}
}

func TestEnvRefsStayEnvRefsWhenVarUnset(t *testing.T) {
	// A missing env var must NOT hard-fail config load: entries just resolve
	// to empty keys until the runtime env provides them (k8s secret wiring).
	os.Unsetenv("GROQ_API_KEY")
	cfg, err := config.LoadFile("../../testdata/litellm-config.yaml")
	if err != nil {
		t.Fatalf("load must tolerate unset env: %v", err)
	}
	if len(cfg.ModelList) != 187 {
		t.Fatalf("entries=%d", len(cfg.ModelList))
	}
}
