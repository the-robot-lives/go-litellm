package anthropic

import (
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/providers"
)

func TestStreamPreambleIsMessageStartOnly(t *testing.T) {
	frames := NewStream("groq/opus").Preamble()
	joined := strings.Join(frames, "")
	if !strings.Contains(joined, "message_start") {
		t.Fatalf("missing message_start: %s", joined)
	}
	if strings.Contains(joined, "content_block_start") {
		t.Fatalf("preamble must not open a text block: %s", joined)
	}
}

func TestStreamReasoningThenText(t *testing.T) {
	s := NewStream("groq/haiku")
	_ = s.Preamble()

	joined := strings.Join(s.Push(providers.StreamChunk{Reasoning: "let me think"}), "")
	if !strings.Contains(joined, "thinking") || !strings.Contains(joined, "thinking_delta") || !strings.Contains(joined, "let me think") {
		t.Fatalf("thinking block: %s", joined)
	}

	joined = strings.Join(s.Push(providers.StreamChunk{Text: "pong", FinishReason: "stop", IsFinished: true}), "")
	if !strings.Contains(joined, "content_block_stop") || !strings.Contains(joined, "text_delta") || !strings.Contains(joined, "pong") {
		t.Fatalf("text after thinking: %s", joined)
	}

	joined = strings.Join(s.Close(), "")
	if !strings.Contains(joined, "message_delta") || !strings.Contains(joined, "end_turn") || !strings.Contains(joined, "message_stop") {
		t.Fatalf("close: %s", joined)
	}
}

func TestStreamToolCallsBecomeToolUse(t *testing.T) {
	s := NewStream("groq/opus")
	_ = s.Preamble()

	joined := strings.Join(s.Push(providers.StreamChunk{
		ToolUse: []any{
			map[string]any{
				"id":   "call_1",
				"type": "function",
				"function": map[string]any{
					"name":      "Bash",
					"arguments": `{"command":"ls"}`,
				},
			},
		},
		FinishReason: "tool_calls",
	}), "")
	for _, needle := range []string{"tool_use", "Bash", "input_json_delta", "partial_json", `{\"command\":\"ls\"}`} {
		if !strings.Contains(joined, needle) {
			t.Fatalf("missing %q in %s", needle, joined)
		}
	}

	joined = strings.Join(s.Close(), "")
	if !strings.Contains(joined, `"stop_reason":"tool_use"`) {
		t.Fatalf("stop_reason: %s", joined)
	}
}

func TestStreamToolCallFragments(t *testing.T) {
	s := NewStream("groq/haiku")
	_ = s.Preamble()

	start := strings.Join(s.Push(providers.StreamChunk{
		ToolUse: []any{
			map[string]any{
				"index": 0,
				"id":    "call_1",
				"type":  "function",
				"function": map[string]any{
					"name":      "Bash",
					"arguments": "",
				},
			},
		},
	}), "")
	if !strings.Contains(start, `"name":"Bash"`) {
		t.Fatalf("start: %s", start)
	}
	if strings.Contains(start, "input_json_delta") {
		t.Fatalf("empty args should not emit delta: %s", start)
	}

	delta := strings.Join(s.Push(providers.StreamChunk{
		ToolUse: []any{
			map[string]any{
				"index": 0,
				"function": map[string]any{
					"arguments": `{"command":`,
				},
			},
		},
	}), "")
	if !strings.Contains(delta, "input_json_delta") || !strings.Contains(delta, `{\"command\":`) {
		t.Fatalf("delta: %s", delta)
	}

	delta2 := strings.Join(s.Push(providers.StreamChunk{
		ToolUse: []any{
			map[string]any{
				"index": 0,
				"function": map[string]any{
					"arguments": `"ls"}`,
				},
			},
		},
		FinishReason: "tool_calls",
	}), "")
	if !strings.Contains(delta2, `\"ls\"}`) {
		t.Fatalf("delta2: %s", delta2)
	}
}

func TestStreamParallelToolsClosePrevious(t *testing.T) {
	s := NewStream("groq/sonnet")
	_ = s.Preamble()
	joined := strings.Join(s.Push(providers.StreamChunk{
		ToolUse: []any{
			map[string]any{
				"index": 0, "id": "call_a", "type": "function",
				"function": map[string]any{"name": "Read", "arguments": `{"path":"a"}`},
			},
			map[string]any{
				"index": 1, "id": "call_b", "type": "function",
				"function": map[string]any{"name": "Read", "arguments": `{"path":"b"}`},
			},
		},
		FinishReason: "tool_calls",
	}), "")
	if strings.Count(joined, `"type":"tool_use"`) != 2 {
		t.Fatalf("want 2 tool_use starts: %s", joined)
	}
	if strings.Count(joined, "event: content_block_stop") != 1 {
		t.Fatalf("first tool must close before the second starts: %s", joined)
	}
	if !strings.Contains(joined, `"id":"call_a"`) || !strings.Contains(joined, `"id":"call_b"`) {
		t.Fatalf("ids: %s", joined)
	}
}

func TestStreamEmptyStillEmitsTextBlock(t *testing.T) {
	s := NewStream("m")
	_ = s.Preamble()
	joined := strings.Join(s.Close(), "")
	if !strings.Contains(joined, "content_block_start") || !strings.Contains(joined, "content_block_stop") || !strings.Contains(joined, "message_stop") {
		t.Fatalf("empty close: %s", joined)
	}
	if strings.Contains(joined, "tool_use") {
		t.Fatalf("empty stream must not invent tools: %s", joined)
	}
}

func TestStreamCoercesStopToToolUseWhenToolsEmitted(t *testing.T) {
	s := NewStream("groq/haiku")
	_ = s.Preamble()
	_ = s.Push(providers.StreamChunk{
		ToolUse: []any{
			map[string]any{
				"id": "call_1", "type": "function",
				"function": map[string]any{"name": "Bash", "arguments": `{}`},
			},
		},
		FinishReason: "stop",
	})
	joined := strings.Join(s.Close(), "")
	if !strings.Contains(joined, `"stop_reason":"tool_use"`) {
		t.Fatalf("coerced stop_reason: %s", joined)
	}
}

func TestStreamObjectArguments(t *testing.T) {
	s := NewStream("groq/haiku")
	_ = s.Preamble()
	joined := strings.Join(s.Push(providers.StreamChunk{
		ToolUse: []any{
			map[string]any{
				"id": "call_1", "type": "function",
				"function": map[string]any{
					"name":      "Bash",
					"arguments": map[string]any{"command": "ls"},
				},
			},
		},
		FinishReason: "tool_calls",
	}), "")
	if !strings.Contains(joined, `{\"command\":\"ls\"}`) {
		t.Fatalf("marshaled args: %s", joined)
	}
}
