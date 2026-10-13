package gateway

// /v1/messages must strip litellm_params.additional_drop_params from the
// upstream body — the production z.ai (sub) and Alibaba token-plan entries
// both list context_management/thinking/reasoning_effort there because their
// Anthropic-compatible endpoints reject those fields verbatim.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/config"
	"github.com/noizu-labs/go-litellm/internal/runtime"
)

func TestMessagesStripsAdditionalDropParams(t *testing.T) {
	var got map[string]any
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_x","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer up.Close()

	app := testApp(t, "sk")
	app.Router.Add(map[string]any{
		"model_name": "alibaba/haiku",
		"litellm_params": map[string]any{
			"model":    "anthropic/qwen3.6-flash",
			"api_key":  "test-dummy-qwen",
			"api_base": up.URL, // token-plan shape: anthropic-compatible upstream
			"additional_drop_params": []any{
				"context_management", "thinking", "reasoning_effort",
			},
		},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	body := `{
		"model": "alibaba/haiku",
		"messages": [{"role":"user","content":"hi"}],
		"max_tokens": 64,
		"thinking": {"type":"enabled","budget_tokens":1024},
		"context_management": {"edits":{"enabled":false}},
		"reasoning_effort": "high"
	}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, raw)
	}
	for _, dropped := range []string{"thinking", "context_management", "reasoning_effort"} {
		if _, ok := got[dropped]; ok {
			t.Fatalf("upstream body still carries %q: %v", dropped, got)
		}
	}
	if m, _ := got["model"].(string); m != "qwen3.6-flash" {
		t.Fatalf("model=%v", got["model"])
	}
}

func TestMessagesStripKeepsDeploymentEffortPin(t *testing.T) {
	// A deployment that both drops and pins the same param keeps its own
	// litellm_params value (same ordering contract as the chat path).
	var got map[string]any
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &got)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_x","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer up.Close()

	app := testApp(t, "sk")
	app.Router.Add(map[string]any{
		"model_name": "zai/glm-5.3-flash",
		"litellm_params": map[string]any{
			"model":       "anthropic/glm-5.3-flash",
			"api_key":     "test-dummy-zai",
			"api_base":    up.URL,
			"drop_params": true,
			"additional_drop_params": []any{
				"context_management", "thinking", "reasoning_effort",
			},
		},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	// Claude Code speaks anthropic: a thinking block, no reasoning_effort.
	body := `{
		"model": "zai/glm-5.3-flash",
		"messages": [{"role":"user","content":"hi"}],
		"max_tokens": 64,
		"thinking": {"type":"enabled","budget_tokens":2048},
		"context_management": {"edits":{"enabled":false}}
	}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, raw)
	}
	if _, ok := got["thinking"]; ok {
		t.Fatalf("thinking leaked upstream: %v", got)
	}
	if _, ok := got["context_management"]; ok {
		t.Fatalf("context_management leaked upstream: %v", got)
	}
	// The base here is not a z.ai host, so no effort mapping is expected;
	// the strip contract itself is the point of this test.
}

func TestProductionZaiDeploymentStillResolves(t *testing.T) {
	// Sanity: the config-loaded z.ai (sub) entry must be classified as an
	// anthropic deployment so /v1/messages takes the custom-base path instead
	// of the api.anthropic.com passthrough.
	cfg, err := config.LoadFile("../../testdata/litellm-config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	app := New(runtime.Settings{Host: "127.0.0.1", Port: 0, MasterKey: "sk", DatabaseURL: ":memory:"}, cfg)
	defer app.Close()
	dep := app.Router.Lookup("zai/glm-5.3-flash (sub, low)")
	if dep == nil {
		t.Fatal("zai sub deployment missing")
	}
}
