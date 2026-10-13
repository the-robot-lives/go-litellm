package gateway

// Crash-hardening regression: POST /v1/audio/transcriptions (multipart,
// model=whisper-1) used to enter front-proxy passthrough, whose LiteLLM
// target resolves to the gateway itself. Each forward re-entered the handler
// — 7090 nested requests per call in the local reproduction — exhausting
// ports/fds until the pod died with no panic trace (Recreate strategy = full
// outage for all traffic). Three layers now hold:
//   1. /v1/audio/* routes answer a clean authenticated 501, never forwarded;
//   2. any forward that resolves to this gateway is refused (502 envelope);
//   3. a global panic-recovery middleware turns handler panics into 500 JSON.

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/config"
	"github.com/noizu-labs/go-litellm/internal/runtime"
)

// selfApp builds an App whose front-proxy LiteLLM target (settings host:port)
// is the test server itself — the exact production shape that looped.
func selfApp(t *testing.T, ts *httptest.Server, master string) *App {
	t.Helper()
	app := New(runtime.Settings{
		Host:        "127.0.0.1",
		Port:        ts.Listener.Addr().(*net.TCPAddr).Port,
		MasterKey:   master,
		DatabaseURL: ":memory:",
	}, config.Empty())
	t.Cleanup(app.Close)
	return app
}

func multipartBody(t *testing.T) (*bytes.Buffer, string) {
	t.Helper()
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	fw, err := mw.CreateFormFile("file", "tiny.wav")
	if err != nil {
		t.Fatal(err)
	}
	// Minimal 44-byte WAV header — content is irrelevant; the route must not
	// forward regardless of payload.
	if _, err := fw.Write([]byte("RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00data\x00\x00\x00\x00")); err != nil {
		t.Fatal(err)
	}
	if err := mw.WriteField("model", "whisper-1"); err != nil {
		t.Fatal(err)
	}
	if err := mw.Close(); err != nil {
		t.Fatal(err)
	}
	return &buf, mw.FormDataContentType()
}

func mustJSONError(t *testing.T, resp *http.Response) map[string]any {
	t.Helper()
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Fatalf("content-type %q", ct)
	}
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("body not JSON: %v", err)
	}
	e, ok := body["error"].(map[string]any)
	if !ok {
		t.Fatalf("no error envelope: %v", body)
	}
	if s, _ := e["message"].(string); s == "" {
		t.Fatalf("empty error message: %v", body)
	}
	return body
}

func TestAudioTranscriptionsCleanErrorAndServerSurvives(t *testing.T) {
	ts := httptest.NewUnstartedServer(http.NewServeMux())
	app := selfApp(t, ts, "sk-test-audio")
	ts.Config.Handler = app.Handler()
	ts.Start()
	defer ts.Close()

	buf, ct := multipartBody(t)
	req, err := http.NewRequest("POST", ts.URL+"/v1/audio/transcriptions", buf)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", ct)
	req.Header.Set("Authorization", "Bearer sk-test-audio")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	if resp.StatusCode != http.StatusNotImplemented {
		t.Fatalf("want 501, got %d", resp.StatusCode)
	}
	body := mustJSONError(t, resp)
	if e := body["error"].(map[string]any); e["type"] != "invalid_request_error" {
		t.Fatalf("error type %v", e["type"])
	}

	// The process must still be serving: the very next request succeeds.
	req2, _ := http.NewRequest("GET", ts.URL+"/v1/models", nil)
	req2.Header.Set("Authorization", "Bearer sk-test-audio")
	resp2, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatalf("server died after audio request: %v", err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != 200 {
		t.Fatalf("models after audio: %d", resp2.StatusCode)
	}
}

func TestAudioRequiresKey(t *testing.T) {
	ts := httptest.NewUnstartedServer(http.NewServeMux())
	app := selfApp(t, ts, "sk-test-audio")
	ts.Config.Handler = app.Handler()
	ts.Start()
	defer ts.Close()

	buf, ct := multipartBody(t)
	req, _ := http.NewRequest("POST", ts.URL+"/v1/audio/transcriptions", buf)
	req.Header.Set("Content-Type", ct)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("want 401 without key, got %d", resp.StatusCode)
	}
}

func TestAudioSpeechCleanError(t *testing.T) {
	ts := httptest.NewUnstartedServer(http.NewServeMux())
	app := selfApp(t, ts, "sk-test-audio")
	ts.Config.Handler = app.Handler()
	ts.Start()
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/v1/audio/speech", strings.NewReader(`{"model":"tts-1","input":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-test-audio")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusNotImplemented {
		t.Fatalf("want 501, got %d", resp.StatusCode)
	}
	mustJSONError(t, resp)
}

func TestPassthroughSelfForwardRefused(t *testing.T) {
	ts := httptest.NewUnstartedServer(http.NewServeMux())
	app := selfApp(t, ts, "sk-test-audio")
	ts.Config.Handler = app.Handler()
	ts.Start()
	defer ts.Close()

	// /v1/images has no native handler: falls to passthrough, whose LiteLLM
	// target is this server — the guard must refuse it with 502, not loop.
	req, _ := http.NewRequest("POST", ts.URL+"/v1/images/generations", strings.NewReader(`{"model":"dall-e-3","prompt":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-test-audio")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("want 502, got %d", resp.StatusCode)
	}
	body := mustJSONError(t, resp)
	e := body["error"].(map[string]any)
	if msg, _ := e["message"].(string); !strings.Contains(msg, "resolves to this gateway") {
		t.Fatalf("message %q", msg)
	}

	// Still alive.
	req2, _ := http.NewRequest("GET", ts.URL+"/v1/models", nil)
	req2.Header.Set("Authorization", "Bearer sk-test-audio")
	resp2, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatalf("server died: %v", err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != 200 {
		t.Fatalf("models after self-forward refusal: %d", resp2.StatusCode)
	}
}

func TestRecoveryMiddleware(t *testing.T) {
	app := testApp(t, "sk-test-recover")
	boom := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("boom-recovery-test")
	})
	ts := httptest.NewServer(app.withRecovery(boom))
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/anything")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("want 500, got %d", resp.StatusCode)
	}
	mustJSONError(t, resp)

	// Server keeps serving after the panic.
	resp2, err := http.Get(ts.URL + "/anything")
	if err != nil {
		t.Fatalf("server died after panic: %v", err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusInternalServerError {
		t.Fatalf("second request: %d", resp2.StatusCode)
	}
}

func TestRecoveryMiddlewareMidStreamNoDoubleWrite(t *testing.T) {
	app := testApp(t, "")
	after := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		_, _ = io.WriteString(w, "partial")
		panic("mid-stream-boom")
	})
	ts := httptest.NewServer(app.withRecovery(after))
	defer ts.Close()
	resp, err := http.Get(ts.URL + "/stream")
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("committed status changed: %d", resp.StatusCode)
	}
}

func TestIsSelfTarget(t *testing.T) {
	app := testApp(t, "")
	app.Settings.Host = "0.0.0.0"
	app.Settings.Port = 4000
	cases := []struct {
		url  string
		want bool
	}{
		{"http://0.0.0.0:4000/v1/audio/x", true},
		{"http://127.0.0.1:4000/v1/audio/x", true},
		{"http://localhost:4000/v1/audio/x", true},
		{"http://127.0.0.1:4444/v1/audio/x", false},
		{"https://api.anthropic.com/v1/messages", false},
		{"http://inference.internal:4000/v1/x", false},
		{"http://127.0.0.1:0/v1/x", false}, // port 0 never self
	}
	for _, c := range cases {
		if got := app.isSelfTarget(c.url); got != c.want {
			t.Errorf("isSelfTarget(%q) = %v, want %v", c.url, got, c.want)
		}
	}
}
