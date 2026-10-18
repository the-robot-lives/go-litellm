package responses

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/providers"
)

type sseEvent struct {
	name string
	data map[string]any
}

// parseFrames splits "event: X\ndata: {json}\n\n" frames and asserts the wire shape.
func parseFrames(t *testing.T, raw string) []sseEvent {
	t.Helper()
	var events []sseEvent
	for _, block := range strings.Split(raw, "\n\n") {
		if block == "" {
			continue
		}
		if !strings.HasPrefix(block, "event: ") {
			t.Fatalf("frame missing event prefix: %q", block)
		}
		lines := strings.Split(block, "\n")
		if len(lines) != 2 || !strings.HasPrefix(lines[1], "data: ") {
			t.Fatalf("frame shape: %q", block)
		}
		var data map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(lines[1], "data: ")), &data); err != nil {
			t.Fatalf("frame data not JSON: %q (%v)", lines[1], err)
		}
		events = append(events, sseEvent{name: strings.TrimPrefix(lines[0], "event: "), data: data})
	}
	return events
}

func runStream(t *testing.T, s *Stream, chunks []providers.StreamChunk) []sseEvent {
	t.Helper()
	var raw strings.Builder
	for _, f := range s.Preamble() {
		raw.WriteString(f)
	}
	for _, c := range chunks {
		for _, f := range s.Push(c) {
			raw.WriteString(f)
		}
	}
	for _, f := range s.Close() {
		raw.WriteString(f)
	}
	return parseFrames(t, raw.String())
}

// TestResponsesStreamAssembly pins the exact event sequence, the monotonically
// increasing sequence_number, and the assembled terminal response for a
// scripted feed: two text fragments then one tool call (L3 list shape).
func TestResponsesStreamAssembly(t *testing.T) {
	chunks := []providers.StreamChunk{
		{Text: "Hello"},
		{Text: " world"},
		{ToolUse: []any{map[string]any{
			"index": 0, "id": "call_1", "type": "function",
			"function": map[string]any{"name": "get_weather", "arguments": ""},
		}}},
		{ToolUse: []any{map[string]any{
			"index": 0, "function": map[string]any{"arguments": `{"city":"SF"}`},
		}}},
		{FinishReason: "tool_calls", IsFinished: true,
			Usage: map[string]any{"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}},
	}
	events := runStream(t, NewStream("router-model"), chunks)

	want := []string{
		"response.created",
		"response.output_item.added",  // message item opens
		"response.content_part.added", // output_text part opens
		"response.output_text.delta",  // "Hello"
		"response.output_text.delta",  // " world"
		"response.output_item.added",  // function_call item opens (named fragment)
		"response.function_call_arguments.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done", // message
		"response.function_call_arguments.done",
		"response.output_item.done", // function_call
		"response.completed",
	}
	if len(events) != len(want) {
		t.Fatalf("event count %d, want %d: %v", len(events), len(want), names(events))
	}
	for i, ev := range events {
		if ev.name != want[i] {
			t.Fatalf("event[%d] = %q, want %q (all: %v)", i, ev.name, want[i], names(events))
		}
		if seq, _ := ev.data["sequence_number"].(float64); int(seq) != i {
			t.Fatalf("event[%d] sequence_number %v, want %d", i, ev.data["sequence_number"], i)
		}
	}

	// Text deltas carry the fragments.
	if d, _ := events[3].data["delta"].(string); d != "Hello" {
		t.Fatalf("delta[0] %v", events[3].data["delta"])
	}
	if d, _ := events[4].data["delta"].(string); d != " world" {
		t.Fatalf("delta[1] %v", events[4].data["delta"])
	}

	// The function_call item opens with its name payload + call_id.
	fcOpen := events[5].data["item"].(map[string]any)
	if fcOpen["type"] != "function_call" || fcOpen["call_id"] != "call_1" || fcOpen["name"] != "get_weather" {
		t.Fatalf("fc open item %v", fcOpen)
	}
	if !strings.HasPrefix(fcOpen["id"].(string), "fc_") {
		t.Fatalf("fc item id %v", fcOpen["id"])
	}
	if d, _ := events[6].data["delta"].(string); d != `{"city":"SF"}` {
		t.Fatalf("args delta %v", events[6].data["delta"])
	}

	// The terminal envelope carries the FULL assembled response with real usage.
	completed := events[len(events)-1].data["response"].(map[string]any)
	if completed["id"] != events[0].data["response"].(map[string]any)["id"] {
		t.Fatal("response id must be stable across created/completed")
	}
	if completed["status"] != "completed" || completed["model"] != "router-model" {
		t.Fatalf("status/model %v/%v", completed["status"], completed["model"])
	}
	if _, flat := completed["output_text"]; flat {
		t.Fatal("flat output_text must never be emitted")
	}
	output := completed["output"].([]any)
	msg := output[0].(map[string]any)
	if msg["type"] != "message" || msg["status"] != "completed" {
		t.Fatalf("assembled message %v", msg)
	}
	if text := msg["content"].([]any)[0].(map[string]any)["text"]; text != "Hello world" {
		t.Fatalf("assembled text %v", text)
	}
	fc := output[1].(map[string]any)
	if fc["type"] != "function_call" || fc["call_id"] != "call_1" || fc["name"] != "get_weather" || fc["arguments"] != `{"city":"SF"}` {
		t.Fatalf("assembled function_call %v", fc)
	}
	usage := completed["usage"].(map[string]any)
	if usage["input_tokens"] != 3.0 || usage["output_tokens"] != 7.0 || usage["total_tokens"] != 10.0 {
		t.Fatalf("completed usage %v", usage)
	}
	if completed["error"] != nil {
		t.Fatalf("completed error %v", completed["error"])
	}
}

// TestResponsesStreamEmptyFallback mirrors anthropic.Stream: a stream with no
// output still yields one (empty) message item so clients always see an item.
func TestResponsesStreamEmptyFallback(t *testing.T) {
	events := runStream(t, NewStream("m"), nil)
	want := []string{
		"response.created",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}
	if len(events) != len(want) {
		t.Fatalf("events %v", names(events))
	}
	for i, ev := range events {
		if ev.name != want[i] {
			t.Fatalf("event[%d] = %q, want %q", i, ev.name, want[i])
		}
	}
	output := events[len(events)-1].data["response"].(map[string]any)["output"].([]any)
	text := output[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]
	if text != "" {
		t.Fatalf("fallback text %v", text)
	}
}

// TestResponsesStreamIncompleteStatus: finish length → incomplete envelope.
func TestResponsesStreamIncompleteStatus(t *testing.T) {
	events := runStream(t, NewStream("m"), []providers.StreamChunk{
		{Text: "truncated", FinishReason: "length", IsFinished: true},
	})
	completed := events[len(events)-1].data["response"].(map[string]any)
	if completed["status"] != "incomplete" {
		t.Fatalf("status %v", completed["status"])
	}
	details := completed["incomplete_details"].(map[string]any)
	if details["reason"] != "max_output_tokens" {
		t.Fatalf("reason %v", details)
	}
	if completed["usage"].(map[string]any)["total_tokens"] != 0.0 {
		t.Fatalf("usage absent upstream must be zeros: %v", completed["usage"])
	}
}

func names(events []sseEvent) []string {
	out := make([]string, len(events))
	for i, e := range events {
		out[i] = e.name
	}
	return out
}
