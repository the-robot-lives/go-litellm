package gateway

// Auth contract for /v1/messages (security fix):
//   - non-claude-* models are the LiteLLM surface → require the master key
//     exactly like /v1/chat/completions (only when LITELLM_MASTER_KEY is set);
//   - claude-* stays AuthPassthrough (client's own token forwarded to
//     api.anthropic.com) — the front-proxy contract run-claude depends on;
//   - a gateway launched without a master key serves everything keyless, as before.

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func postMessages(t *testing.T, baseURL, bearer, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequest("POST", baseURL+"/v1/messages", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestMessagesDeploymentRequiresMasterKey(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_x","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}`))
	}))
	defer up.Close()

	app := testApp(t, "sk-test-messages") // production posture: master key set
	app.Router.Add(map[string]any{
		"model_name": "echo",
		"litellm_params": map[string]any{
			"model":    "anthropic/echo-1",
			"api_base": up.URL,
			"api_key":  "upstream-key",
		},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	body := `{"model":"echo","messages":[{"role":"user","content":"hi"}],"max_tokens":8}`

	resp := postMessages(t, ts.URL, "", body) // missing key
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Fatalf("missing key: status %d body %s", resp.StatusCode, b)
	}

	resp = postMessages(t, ts.URL, "sk-wrong", body) // wrong key
	b, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 401 {
		t.Fatalf("wrong key: status %d body %s", resp.StatusCode, b)
	}

	resp = postMessages(t, ts.URL, "sk-test-messages", body) // correct key → routed
	b, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("correct key: status %d body %s", resp.StatusCode, b)
	}
	if !strings.Contains(string(b), "ok") {
		t.Fatalf("correct key: upstream response not relayed: %s", b)
	}
}

// recordingDoer stands in for the Anthropic upstream: /v1/messages claude-* has
// api.anthropic.com hardcoded, so the test captures the outbound request at the
// HTTPDoer seam instead of hitting the network.
type recordingDoer struct {
	gotAuth string
	gotPath string
}

func (d *recordingDoer) Do(r *http.Request) (*http.Response, error) {
	d.gotAuth = r.Header.Get("Authorization")
	d.gotPath = r.URL.Path
	return &http.Response{
		StatusCode: 200,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(`{"id":"msg_x","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}`)),
		Request:    r,
	}, nil
}

func TestMessagesClaudeModelPassesThroughWithoutMasterKey(t *testing.T) {
	app := testApp(t, "sk-test-messages") // master key set — claude-* must not need it
	up := &recordingDoer{}
	app.HTTP = up
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	body := `{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}],"max_tokens":8}`
	resp := postMessages(t, ts.URL, "", body) // no client key at all
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("claude passthrough: status %d body %s", resp.StatusCode, b)
	}
	if up.gotPath != "/v1/messages" {
		t.Fatalf("forward path=%s", up.gotPath)
	}
	if up.gotAuth != "" {
		t.Fatalf("passthrough must forward the client's own (absent) token, got %q", up.gotAuth)
	}

	// A client-presented token is forwarded verbatim (AuthPassthrough).
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer sk-ant-oat01-client-owns-this")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if up.gotAuth != "Bearer sk-ant-oat01-client-owns-this" {
		t.Fatalf("client token not forwarded verbatim: %q", up.gotAuth)
	}
}

func TestMessagesKeylessGatewayServesDeploymentWithoutKey(t *testing.T) {
	// Negative regression: no LITELLM_MASTER_KEY (local front-proxy posture) —
	// deployment-path /v1/messages serves without any client key, as before.
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_x","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}`))
	}))
	defer up.Close()

	app := testApp(t, "")
	app.Router.Add(map[string]any{
		"model_name": "echo",
		"litellm_params": map[string]any{
			"model":    "anthropic/echo-1",
			"api_base": up.URL,
			"api_key":  "upstream-key",
		},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	body := `{"model":"echo","messages":[{"role":"user","content":"hi"}],"max_tokens":8}`
	resp := postMessages(t, ts.URL, "", body)
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("keyless gateway: status %d body %s", resp.StatusCode, b)
	}
}
