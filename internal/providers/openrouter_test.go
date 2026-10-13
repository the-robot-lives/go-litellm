package providers

import (
	"testing"
)

func TestOpenRouterRegistryResolution(t *testing.T) {
	// Production entries are `openrouter/<vendor>/<model>` with no api_base.
	res, e := Resolve("openrouter/qwen/qwen3.8-max", map[string]any{
		"model":   "openrouter/qwen/qwen3.8-max",
		"api_key": "os.environ/OPENROUTER_API_KEY",
	})
	if e != nil {
		t.Fatal(e)
	}
	if res.Provider != "openrouter" {
		t.Fatalf("provider=%s", res.Provider)
	}
	// The vendor segment must survive prefix stripping — OpenRouter ids are
	// two-segment paths.
	if res.Model != "qwen/qwen3.8-max" {
		t.Fatalf("model=%s", res.Model)
	}
}

func TestOpenRouterDefaultBase(t *testing.T) {
	req := &Request{
		Model:         "qwen/qwen3.8-max",
		Provider:      "openrouter",
		LiteLLMParams: map[string]any{},
		CallType:      "chat",
	}
	if got := OpenRouter.CompleteURL(req); got != "https://openrouter.ai/api/v1/chat/completions" {
		t.Fatalf("url=%s", got)
	}
}

func TestOpenRouterCustomBaseAndHeaders(t *testing.T) {
	req := &Request{
		Model:    "qwen/qwen3.8-max",
		Provider: "openrouter",
		LiteLLMParams: map[string]any{
			"api_base": "https://openrouter-proxy.internal/api/v1",
			"api_key":  "test-dummy-openrouter-key",
			"extra_headers": map[string]any{
				"HTTP-Referer": "https://inference.example.test",
				"X-Title":      "go-litellm",
			},
		},
	}
	if got := OpenRouter.CompleteURL(req); got != "https://openrouter-proxy.internal/api/v1/chat/completions" {
		t.Fatalf("url=%s", got)
	}
	headers, e := OpenRouter.ValidateEnvironment(req, map[string]string{})
	if e != nil {
		t.Fatal(e)
	}
	if headers["authorization"] != "Bearer test-dummy-openrouter-key" {
		t.Fatalf("authorization=%q", headers["authorization"])
	}
	if headers["HTTP-Referer"] != "https://inference.example.test" || headers["X-Title"] != "go-litellm" {
		t.Fatalf("extra headers lost: %+v", headers)
	}
}

func TestEndpointProviderDetectsOpenRouterAndGeminiHosts(t *testing.T) {
	p, _ := classify("some/model", map[string]any{
		"api_base": "https://openrouter.ai/api/v1",
		"model":    "some/model",
	})
	if p != "openrouter" {
		t.Fatalf("openrouter host -> %s", p)
	}
	p, _ = classify("some/model", map[string]any{
		"api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
		"model":    "some/model",
	})
	if p != "gemini" {
		t.Fatalf("gemini host -> %s", p)
	}
}
