package responses

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/providers"
)

func TestResponsesValidateStateless(t *testing.T) {
	if e := ValidateStateless(map[string]any{"model": "m", "input": "hi"}); e != nil {
		t.Fatalf("clean body rejected: %v", e)
	}

	cases := []struct {
		name  string
		body  map[string]any
		field string
	}{
		{"previous_response_id", map[string]any{"model": "m", "previous_response_id": "resp_old"}, "previous_response_id"},
		{"store", map[string]any{"model": "m", "store": true}, "store"},
		{"item_reference", map[string]any{"model": "m", "input": []any{
			map[string]any{"type": "item_reference", "id": "item_1"},
		}}, "item_reference"},
	}
	for _, tc := range cases {
		e := ValidateStateless(tc.body)
		if e == nil {
			t.Fatalf("%s: expected 400, got nil", tc.name)
		}
		if e.Status != 400 {
			t.Fatalf("%s: status %d", tc.name, e.Status)
		}
		if e.Type != "invalid_request_error" {
			t.Fatalf("%s: type %q", tc.name, e.Type)
		}
		if !strings.Contains(e.Message, tc.field) {
			t.Fatalf("%s: message %q does not name the field", tc.name, e.Message)
		}
	}
}

// TestResponsesRequestToOpenAI pins the full translation with inline literals:
// instructions, mixed item array (message blocks incl. image, convenience
// string content, function_call, function_call_output, dropped reasoning),
// flat tools, param renames, and the silently dropped fields.
func TestResponsesRequestToOpenAI(t *testing.T) {
	in := `{
      "model": "router-model",
      "instructions": "be brief",
      "input": [
        {"type":"message","role":"developer","content":[{"type":"input_text","text":"dev note"}]},
        {"type":"message","role":"user","content":[
          {"type":"input_text","text":"look"},
          {"type":"input_image","image_url":"https://x.test/y.png"}
        ]},
        {"role":"assistant","content":"earlier answer"},
        {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"},
        {"type":"function_call_output","call_id":"call_1","output":[{"type":"output_text","text":"72F"},{"type":"output_text","text":"sunny"}]},
        {"type":"reasoning","summary":[{"type":"summary_text","text":"hidden"}]}
      ],
      "tools":[
        {"type":"function","name":"get_weather","description":"weather","parameters":{"type":"object"},"strict":true},
        {"type":"web_search"}
      ],
      "tool_choice":{"type":"function","name":"get_weather"},
      "max_output_tokens": 256,
      "reasoning":{"effort":"high"},
      "text":{"format":{"type":"json_object"}},
      "metadata":{"user":"u-1"},
      "temperature": 0.5,
      "top_p": 0.9,
      "parallel_tool_calls": false,
      "stream": true,
      "truncation":"auto",
      "include":["reasoning.encrypted_content"],
      "background": false,
      "service_tier":"auto"
    }`
	var body map[string]any
	if err := json.Unmarshal([]byte(in), &body); err != nil {
		t.Fatal(err)
	}

	want := `{
      "model": "router-model",
      "messages": [
        {"role":"system","content":"be brief"},
        {"role":"developer","content":[{"type":"text","text":"dev note"}]},
        {"role":"user","content":[
          {"type":"text","text":"look"},
          {"type":"image_url","image_url":{"url":"https://x.test/y.png"}}
        ]},
        {"role":"assistant","content":"earlier answer"},
        {"role":"assistant","content":null,"tool_calls":[
          {"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}
        ]},
        {"role":"tool","tool_call_id":"call_1","content":"72F\nsunny"}
      ],
      "tools":[{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object"},"strict":true}}],
      "tool_choice":{"type":"function","name":"get_weather"},
      "max_completion_tokens": 256,
      "reasoning_effort": "high",
      "response_format":{"type":"json_object"},
      "user":"u-1",
      "temperature": 0.5,
      "top_p": 0.9,
      "parallel_tool_calls": false,
      "stream": true
    }`
	var expected map[string]any
	if err := json.Unmarshal([]byte(want), &expected); err != nil {
		t.Fatal(err)
	}

	got := RequestToOpenAI(body)
	if !reflect.DeepEqual(got, expected) {
		gb, _ := json.Marshal(got)
		eb, _ := json.Marshal(expected)
		t.Fatalf("mismatch\ngot:  %s\nwant: %s", gb, eb)
	}
	for _, dropped := range []string{"max_output_tokens", "truncation", "include", "background", "service_tier", "instructions", "reasoning", "text", "metadata"} {
		if _, ok := got[dropped]; ok {
			t.Fatalf("field %q must be dropped/renamed", dropped)
		}
	}

	// Convenience string input becomes a single user message.
	got2 := RequestToOpenAI(map[string]any{"model": "m", "input": "hi"})
	if !reflect.DeepEqual(got2["messages"], []any{map[string]any{"role": "user", "content": "hi"}}) {
		t.Fatalf("string input: %+v", got2["messages"])
	}
}

func TestResponsesFromModelResponse(t *testing.T) {
	r := &providers.ModelResponse{
		ID: "chatcmpl-1",
		Choices: []any{map[string]any{
			"finish_reason": "tool_calls",
			"message": map[string]any{
				"role":    "assistant",
				"content": "checking",
				"tool_calls": []any{map[string]any{
					"id": "call_9", "type": "function",
					"function": map[string]any{"name": "get_weather", "arguments": `{"city":"SF"}`},
				}},
			},
		}},
		Usage: map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
	}
	body := FromModelResponse(r, "router-model")

	if id, _ := body["id"].(string); !strings.HasPrefix(id, "resp_") {
		t.Fatalf("id %v", body["id"])
	}
	if body["object"] != "response" {
		t.Fatalf("object %v", body["object"])
	}
	if body["status"] != "completed" || body["model"] != "router-model" {
		t.Fatalf("status/model %v/%v", body["status"], body["model"])
	}
	if _, ok := body["created_at"]; !ok {
		t.Fatal("missing created_at")
	}
	if body["error"] != nil {
		t.Fatalf("error %v", body["error"])
	}
	if body["incomplete_details"] != nil {
		t.Fatalf("incomplete_details %v", body["incomplete_details"])
	}
	if meta, ok := body["metadata"].(map[string]any); !ok || len(meta) != 0 {
		t.Fatalf("metadata %v", body["metadata"])
	}
	if _, flat := body["output_text"]; flat {
		t.Fatal("flat output_text must never be emitted")
	}
	if _, echo := body["store"]; echo {
		t.Fatal("store must never be echoed")
	}

	output := body["output"].([]any)
	msg := output[0].(map[string]any)
	if msg["type"] != "message" || msg["role"] != "assistant" || msg["status"] != "completed" {
		t.Fatalf("message item %v", msg)
	}
	if !strings.HasPrefix(msg["id"].(string), "msg_") {
		t.Fatalf("message id %v", msg["id"])
	}
	content := msg["content"].([]any)
	part := content[0].(map[string]any)
	if part["type"] != "output_text" || part["text"] != "checking" {
		t.Fatalf("output_text part %v", part)
	}
	if ann, ok := part["annotations"].([]any); !ok || len(ann) != 0 {
		t.Fatalf("annotations %v", part["annotations"])
	}

	fc := output[1].(map[string]any)
	if fc["type"] != "function_call" || fc["call_id"] != "call_9" || fc["name"] != "get_weather" {
		t.Fatalf("function_call item %v", fc)
	}
	if fc["arguments"] != `{"city":"SF"}` {
		t.Fatalf("arguments %v", fc["arguments"])
	}
	if !strings.HasPrefix(fc["id"].(string), "fc_") {
		t.Fatalf("fc id %v", fc["id"])
	}

	usage := body["usage"].(map[string]any)
	if got, want := usage["input_tokens"], 10; got != want {
		t.Fatalf("usage input_tokens %v (%T)", got, usage["input_tokens"])
	}
	if got := usage["output_tokens"]; got != 5 {
		t.Fatalf("usage output_tokens %v", got)
	}
	if got := usage["total_tokens"]; got != 15 {
		t.Fatalf("usage total_tokens %v", got)
	}

	// finish length/content_filter → incomplete + reason.
	incomplete := FromModelResponse(&providers.ModelResponse{
		Choices: []any{map[string]any{"finish_reason": "length", "message": map[string]any{}}},
	}, "m")
	if incomplete["status"] != "incomplete" {
		t.Fatalf("length status %v", incomplete["status"])
	}
	details := incomplete["incomplete_details"].(map[string]any)
	if details["reason"] != "max_output_tokens" {
		t.Fatalf("length reason %v", details)
	}
	filtered := FromModelResponse(&providers.ModelResponse{
		Choices: []any{map[string]any{"finish_reason": "content_filter", "message": map[string]any{}}},
	}, "m")
	if filtered["incomplete_details"].(map[string]any)["reason"] != "content_filter" {
		t.Fatalf("content_filter reason %v", filtered["incomplete_details"])
	}
}
