package providers

import (
	"testing"
)

// The production config ships gemini entries with no api_base — only
// os.environ/GEMINI_API_KEY — so the adapter default must be Google's
// OpenAI-compatible surface.

func TestGeminiRegistryResolution(t *testing.T) {
	res, e := Resolve("gemini/gemini-3.1-pro-preview", map[string]any{
		"model": "gemini/gemini-3.1-pro-preview",
	})
	if e != nil {
		t.Fatal(e)
	}
	if res.Provider != "gemini" {
		t.Fatalf("provider=%s", res.Provider)
	}
	if res.Model != "gemini-3.1-pro-preview" {
		t.Fatalf("model=%s", res.Model)
	}
	if res.Adapter.Name() != "gemini" {
		t.Fatalf("adapter=%s", res.Adapter.Name())
	}
}

func TestGeminiDefaultBase(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-dummy-gemini-key")
	req := &Request{
		Model:         "gemini-3.1-pro-preview",
		Provider:      "gemini",
		LiteLLMParams: map[string]any{},
		CallType:      "chat",
	}
	if got := Gemini.CompleteURL(req); got != "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" {
		t.Fatalf("url=%s", got)
	}
	req.CallType = "embedding"
	if got := Gemini.CompleteURL(req); got != "https://generativelanguage.googleapis.com/v1beta/openai/embeddings" {
		t.Fatalf("embedding url=%s", got)
	}
}

func TestGeminiCustomBaseWins(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-dummy-gemini-key")
	req := &Request{
		Model:    "gemini-2.0-flash",
		Provider: "gemini",
		LiteLLMParams: map[string]any{
			"api_base": "https://gemini-proxy.internal/v1beta/openai",
		},
		CallType: "chat",
	}
	if got := Gemini.CompleteURL(req); got != "https://gemini-proxy.internal/v1beta/openai/chat/completions" {
		t.Fatalf("url=%s", got)
	}
}

func TestGeminiKeyFromLitellmParams(t *testing.T) {
	// api_key as an os.environ/ ref must resolve ahead of the env default.
	req := &Request{
		Model:    "gemini-2.0-flash",
		Provider: "gemini",
		LiteLLMParams: map[string]any{
			"api_key": "os.environ/GEMINI_API_KEY",
		},
	}
	t.Setenv("GEMINI_API_KEY", "test-dummy-from-env")
	headers, e := Gemini.ValidateEnvironment(req, map[string]string{})
	if e != nil {
		t.Fatal(e)
	}
	if headers["authorization"] != "Bearer test-dummy-from-env" {
		t.Fatalf("authorization=%q", headers["authorization"])
	}
}

func TestGeminiMissingKeyIs401(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "")
	req := &Request{Model: "m", Provider: "gemini", LiteLLMParams: map[string]any{}}
	_, e := Gemini.ValidateEnvironment(req, map[string]string{})
	if e == nil || e.Status != 401 {
		t.Fatalf("expected 401, got %+v", e)
	}
}
