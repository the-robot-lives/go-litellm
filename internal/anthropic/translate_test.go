package anthropic

import (
	"testing"

	"github.com/noizu-labs/go-litellm/internal/providers"
)

func TestRequestToOpenAISystemAndTools(t *testing.T) {
	body := map[string]any{
		"model":      "gpt-4o",
		"max_tokens": 32,
		"system":     "be nice",
		"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		"tools": []any{
			map[string]any{"name": "lookup", "description": "d", "input_schema": map[string]any{"type": "object"}},
		},
	}
	out := RequestToOpenAI(body)
	msgs := out["messages"].([]any)
	sys := msgs[0].(map[string]any)
	if sys["role"] != "system" || sys["content"] != "be nice" {
		t.Fatalf("%+v", msgs)
	}
	if out["max_tokens"] != 32 {
		t.Fatalf("%v", out["max_tokens"])
	}
	tools := out["tools"].([]any)
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "lookup" {
		t.Fatalf("%+v", fn)
	}
}

func TestResponseFromModelResponse(t *testing.T) {
	resp := providers.NewModelResponse(map[string]any{
		"id": "abc",
		"choices": []any{
			map[string]any{
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": "hello"},
			},
		},
		"usage": map[string]any{"prompt_tokens": 3, "completion_tokens": 2},
	})
	out := ResponseFromModelResponse(resp, "my-model")
	if out["stop_reason"] != "end_turn" || out["model"] != "my-model" {
		t.Fatalf("%+v", out)
	}
	u := out["usage"].(map[string]any)
	if u["input_tokens"] != 3 || u["output_tokens"] != 2 {
		t.Fatalf("%+v", u)
	}
}

func TestRequestToOpenAIFlattensSystemContentBlocks(t *testing.T) {
	out := RequestToOpenAI(map[string]any{
		"model": "cerebras/gpt-oss-120b",
		"system": []any{
			map[string]any{
				"type":          "text",
				"text":          "be nice",
				"cache_control": map[string]any{"type": "ephemeral"},
			},
		},
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": "ok"},
			map[string]any{
				"role": "system",
				"content": []any{
					map[string]any{
						"type":          "text",
						"text":          "reminder",
						"cache_control": map[string]any{"type": "ephemeral"},
					},
				},
			},
		},
	})
	msgs := out["messages"].([]any)
	if len(msgs) != 4 {
		t.Fatalf("len=%d %+v", len(msgs), msgs)
	}
	top := msgs[0].(map[string]any)
	if top["role"] != "system" || top["content"] != "be nice" {
		t.Fatalf("top system: %+v", top)
	}
	if _, ok := top["cache_control"]; ok {
		t.Fatalf("cache_control leaked on top system: %+v", top)
	}
	mid := msgs[3].(map[string]any)
	if mid["role"] != "system" || mid["content"] != "reminder" {
		t.Fatalf("in-array system: %+v", mid)
	}
}

func TestThinkingMapsToReasoningEffort(t *testing.T) {
	out := RequestToOpenAI(map[string]any{
		"model":      "groq/opus",
		"max_tokens": 8,
		"messages":   []any{map[string]any{"role": "user", "content": "hi"}},
		"thinking":   map[string]any{"type": "enabled", "budget_tokens": 32},
	})
	if out["reasoning_effort"] != "high" {
		t.Fatalf("%+v", out)
	}
}

func TestStopReason(t *testing.T) {
	if StopReason("stop") != "end_turn" || StopReason("tool_calls") != "tool_use" {
		t.Fatal(StopReason("stop"), StopReason("tool_calls"))
	}
}

func TestResponseFromModelResponseToolCalls(t *testing.T) {
	resp := providers.NewModelResponse(map[string]any{
		"id": "abc",
		"choices": []any{
			map[string]any{
				"finish_reason": "tool_calls",
				"message": map[string]any{
					"role":    "assistant",
					"content": nil,
					"tool_calls": []any{
						map[string]any{
							"id":   "call_1",
							"type": "function",
							"function": map[string]any{
								"name":      "Bash",
								"arguments": `{"command":"ls"}`,
							},
						},
					},
				},
			},
		},
	})
	out := ResponseFromModelResponse(resp, "groq/haiku")
	if out["stop_reason"] != "tool_use" {
		t.Fatalf("stop_reason: %+v", out)
	}
	blocks, _ := out["content"].([]any)
	if len(blocks) != 1 {
		t.Fatalf("blocks: %+v", blocks)
	}
	b := blocks[0].(map[string]any)
	if b["type"] != "tool_use" || b["name"] != "Bash" {
		t.Fatalf("%+v", b)
	}
	input, _ := b["input"].(map[string]any)
	if input["command"] != "ls" {
		t.Fatalf("input: %+v", input)
	}
}
