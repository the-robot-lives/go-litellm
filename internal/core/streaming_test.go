package core

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/providers"
)

type doerFunc func(*http.Request) (*http.Response, error)

func (f doerFunc) Do(r *http.Request) (*http.Response, error) { return f(r) }

func groqPrepared() *Prepared {
	return &Prepared{
		Adapter: providers.Groq,
		Req: &providers.Request{
			Model:         "openai/gpt-oss-20b",
			Provider:      "groq",
			LiteLLMParams: map[string]any{"api_key": "gsk-test"},
			Params:        map[string]any{"model": "openai/gpt-oss-20b", "stream": true},
			Stream:        true,
		},
	}
}

func TestStreamAnthropicForwardsGroqToolCalls(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Bash","arguments":""}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":\"ls\"}"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	client := doerFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader(sse)),
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		}, nil
	})

	rec := httptest.NewRecorder()
	if e := StreamAnthropic(rec, client, groqPrepared(), "groq/haiku"); e != nil {
		t.Fatalf("StreamAnthropic: %v", e)
	}
	body := rec.Body.String()
	for _, needle := range []string{
		"event: message_start",
		`"type":"tool_use"`,
		`"name":"Bash"`,
		"input_json_delta",
		`{\"command\":\"ls\"}`,
		`"stop_reason":"tool_use"`,
		"event: message_stop",
	} {
		if !strings.Contains(body, needle) {
			t.Fatalf("missing %q in\n%s", needle, body)
		}
	}
	if strings.Count(body, `"type":"text"`) > 0 && !strings.Contains(body, "text_delta") {
		// A trailing empty text block is only allowed when the stream had no
		// other content. Tool-only streams must not invent one.
		if strings.Contains(body, `"type":"tool_use"`) {
			t.Fatalf("tool-only stream opened a text block:\n%s", body)
		}
	}
}

func TestStreamAnthropicDropsEmptyToolCallParseShape(t *testing.T) {
	// Regression: previously we opened a text block, ignored tool_calls, and
	// closed with stop_reason=tool_use — Claude Code then failed to parse.
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Read","arguments":"{\"file_path\":\"/tmp/x\"}"}}]},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	client := doerFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader(sse)),
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		}, nil
	})
	rec := httptest.NewRecorder()
	if e := StreamAnthropic(rec, client, groqPrepared(), "groq/haiku"); e != nil {
		t.Fatalf("StreamAnthropic: %v", e)
	}
	body := rec.Body.String()
	if strings.Contains(body, `"stop_reason":"tool_use"`) && !strings.Contains(body, `"type":"tool_use"`) {
		t.Fatalf("stop_reason tool_use without a tool_use block:\n%s", body)
	}
	if !strings.Contains(body, `"name":"Read"`) {
		t.Fatalf("missing Read tool:\n%s", body)
	}
}

func TestStreamAnthropicTextOnly(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"

	client := doerFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader(sse)),
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		}, nil
	})
	rec := httptest.NewRecorder()
	if e := StreamAnthropic(rec, client, groqPrepared(), "groq/haiku"); e != nil {
		t.Fatalf("StreamAnthropic: %v", e)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "text_delta") || !strings.Contains(body, "hello") {
		t.Fatalf("text: %s", body)
	}
	if strings.Contains(body, "tool_use") {
		t.Fatalf("text-only invented tools: %s", body)
	}
	if !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("stop: %s", body)
	}
}
