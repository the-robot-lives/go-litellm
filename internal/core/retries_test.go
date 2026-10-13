package core

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/providers"
)

func TestNumRetriesHonoredOn5xx(t *testing.T) {
	var hits int32
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&hits, 1) < 3 {
			w.WriteHeader(500)
			_, _ = w.Write([]byte(`{"error":{"message":"boom"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer up.Close()

	p := &Prepared{
		Adapter: providers.OpenAI,
		Req: &providers.Request{
			Model:         "m",
			Provider:      "openai",
			LiteLLMParams: map[string]any{"api_key": "test-dummy", "api_base": up.URL, "num_retries": float64(4)},
			CallType:      "chat",
		},
	}
	resp, e := DoCompletion(http.DefaultClient, p)
	if e != nil {
		t.Fatalf("expected success after retries: %v", e)
	}
	if atomic.LoadInt32(&hits) != 3 {
		t.Fatalf("hits=%d", hits)
	}
	if resp.ID != "x" {
		t.Fatalf("id=%q", resp.ID)
	}
}

func TestNumRetriesZeroMeansNoRetry(t *testing.T) {
	var hits int32
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down"}}`))
	}))
	defer up.Close()

	p := &Prepared{
		Adapter: providers.OpenAI,
		Req: &providers.Request{
			Model:         "m",
			Provider:      "openai",
			LiteLLMParams: map[string]any{"api_key": "test-dummy", "api_base": up.URL, "num_retries": float64(0)},
			CallType:      "chat",
		},
	}
	_, e := DoCompletion(http.DefaultClient, p)
	if e == nil || e.Status != 429 {
		t.Fatalf("expected 429, got %+v", e)
	}
	if atomic.LoadInt32(&hits) != 1 {
		t.Fatalf("hits=%d — retry ran despite num_retries=0", hits)
	}
}

func TestDefaultRetriesIsTwo(t *testing.T) {
	var hits int32
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if atomic.AddInt32(&hits, 1) < 3 {
			w.WriteHeader(503)
			_, _ = w.Write([]byte(`{"error":{"message":"unavailable"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"y","object":"chat.completion","choices":[]}`))
	}))
	defer up.Close()

	p := &Prepared{
		Adapter: providers.OpenAI,
		Req: &providers.Request{
			Model:         "m",
			Provider:      "openai",
			LiteLLMParams: map[string]any{"api_key": "test-dummy", "api_base": up.URL},
			CallType:      "chat",
		},
	}
	if _, e := DoCompletion(http.DefaultClient, p); e != nil {
		t.Fatalf("expected success with default retries: %v", e)
	}
	if atomic.LoadInt32(&hits) != 3 {
		t.Fatalf("hits=%d", hits)
	}
}

func TestRetriesOfBounds(t *testing.T) {
	if got := retriesOf(nil); got != 2 {
		t.Fatalf("default=%d", got)
	}
	if got := retriesOf(map[string]any{"num_retries": float64(4)}); got != 4 {
		t.Fatalf("cerebras config value -> %d", got)
	}
	if got := retriesOf(map[string]any{"num_retries": float64(-3)}); got != 0 {
		t.Fatalf("negative -> %d", got)
	}
	if got := retriesOf(map[string]any{"num_retries": float64(500)}); got != 8 {
		t.Fatalf("uncapped -> %d", got)
	}
}

func TestStreamTimeoutOfPrefersStreamTimeout(t *testing.T) {
	if got := streamTimeoutOf(map[string]any{"timeout": float64(30), "stream_timeout": float64(120)}); got.String() != "2m0s" {
		t.Fatalf("stream_timeout -> %s", got)
	}
	if got := streamTimeoutOf(map[string]any{"timeout": float64(30)}); got.String() != "30s" {
		t.Fatalf("timeout fallback -> %s", got)
	}
	if got := streamTimeoutOf(map[string]any{}); got != 0 {
		t.Fatalf("unset -> %s", got)
	}
}

// guard: the 408/429/5xx retry set must not swallow 4xx client errors.
func TestRetryableStatus(t *testing.T) {
	for _, c := range []int{408, 429, 500, 502, 503} {
		if !retryableStatus(c) {
			t.Fatalf("%d should retry", c)
		}
	}
	for _, c := range []int{400, 401, 403, 404, 422} {
		if retryableStatus(c) {
			t.Fatalf("%d must not retry", c)
		}
	}
}
