package gateway

// Env-gated end-to-end test against the real DashScope native text-generation
// endpoint. NEVER runs in CI: requires GO_LITELLM_DASHSCOPE_E2E=1 and
// DASHSCOPE_API_KEY in the environment (optional GO_LITELLM_DASHSCOPE_MODEL
// overrides the model, default qwen-max).
//
//	GO_LITELLM_DASHSCOPE_E2E=1 DASHSCOPE_API_KEY=sk-… go test ./internal/gateway -run TestDashScopeE2E -v

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/config"
	"github.com/noizu-labs/go-litellm/internal/runtime"
)

func TestDashScopeE2E(t *testing.T) {
	if os.Getenv("GO_LITELLM_DASHSCOPE_E2E") != "1" || os.Getenv("DASHSCOPE_API_KEY") == "" {
		t.Skip("GO_LITELLM_DASHSCOPE_E2E=1 + DASHSCOPE_API_KEY required")
	}
	model := os.Getenv("GO_LITELLM_DASHSCOPE_MODEL")
	if model == "" {
		model = "qwen-max"
	}
	cfgYAML := "model_list:\n" +
		"  - model_name: \"dashscope-e2e\"\n" +
		"    litellm_params:\n" +
		"      model: dashscope/" + model + "\n" +
		"      api_key: \"os.environ/DASHSCOPE_API_KEY\"\n" +
		"      api_base: https://dashscope.aliyuncs.com\n" +
		"general_settings:\n" +
		"  master_key: os.environ/GO_LITELLM_TEST_MASTER_KEY\n"
	path := t.TempDir() + "/config.yaml"
	if err := os.WriteFile(path, []byte(cfgYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("GO_LITELLM_TEST_MASTER_KEY", "test-master-key-e2e")
	app := New(runtime.Settings{Host: "127.0.0.1", Port: 0, MasterKey: "test-master-key-e2e", DatabaseURL: ":memory:"}, cfg)
	t.Cleanup(app.Close)
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()

	post := func(body string) (int, string) {
		t.Helper()
		req, err := http.NewRequest("POST", srv.URL+"/v1/chat/completions", bytes.NewReader([]byte(body)))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer test-master-key-e2e")
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(raw)
	}

	// Non-stream.
	if code, raw := post(`{"model":"dashscope-e2e","messages":[{"role":"user","content":"Say pong."}],"max_completion_tokens":16}`); code != 200 {
		t.Fatalf("non-stream %d %s", code, raw)
	} else {
		var out map[string]any
		if err := json.Unmarshal([]byte(raw), &out); err != nil {
			t.Fatalf("body: %s", raw)
		}
		choices, _ := out["choices"].([]any)
		if len(choices) == 0 {
			t.Fatalf("no choices: %s", raw)
		}
		t.Logf("non-stream ok: usage=%v", out["usage"])
	}

	// Stream: synthesized OpenAI SSE from native incremental frames.
	code, raw := post(`{"model":"dashscope-e2e","messages":[{"role":"user","content":"Count 1 to 5."}],"stream":true}`)
	if code != 200 {
		t.Fatalf("stream %d %s", code, raw)
	}
	if !strings.Contains(raw, "data: [DONE]") {
		t.Fatalf("stream missing [DONE]: %s", raw)
	}
	deltas := strings.Count(raw, `"content":`)
	t.Logf("stream ok: %d content deltas", deltas)
	if deltas == 0 {
		t.Fatal("no content deltas observed")
	}

	// Thinking smoke: only for models that actually expose the thinking
	// surface (qwq-* / qwen3-*); incremental_output covers the thinking stream.
	if strings.Contains(model, "qwq") || strings.Contains(model, "qwen3") {
		code, raw = post(`{"model":"dashscope-e2e","messages":[{"role":"user","content":"What is 2+2?"}],"stream":true,"reasoning_effort":"low"}`)
		if code != 200 {
			t.Fatalf("thinking stream %d %s", code, raw)
		}
		t.Logf("thinking stream ok: %d content deltas, reasoning deltas present=%v",
			strings.Count(raw, `"content":`), strings.Contains(raw, `"reasoning"`))
	}

	// Bad key must fail cleanly through the flat-envelope ErrorClass. Config
	// hydration happens at LoadFile, so the negative path needs its own app
	// built after swapping the env value.
	t.Setenv("DASHSCOPE_API_KEY", "sk-wrong-on-purpose")
	badCfg, err := config.LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	badApp := New(runtime.Settings{Host: "127.0.0.1", Port: 0, MasterKey: "test-master-key-e2e", DatabaseURL: ":memory:"}, badCfg)
	t.Cleanup(badApp.Close)
	badSrv := httptest.NewServer(badApp.Handler())
	defer badSrv.Close()
	req, err := http.NewRequest("POST", badSrv.URL+"/v1/chat/completions",
		bytes.NewReader([]byte(`{"model":"dashscope-e2e","messages":[{"role":"user","content":"hi"}]}`)))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer test-master-key-e2e")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	raw2, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode == 200 {
		t.Fatalf("bad key must not succeed: %s", raw2)
	}
	t.Logf("negative test: bad key → %d %s", resp.StatusCode, raw2)
}
