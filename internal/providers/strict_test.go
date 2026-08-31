package providers

import "testing"

func TestShapeStrictFlattensSystemCacheControl(t *testing.T) {
	body := map[string]any{
		"cache_control": map[string]any{"type": "ephemeral"},
		"messages": []any{
			map[string]any{"role": "system", "content": "top"},
			map[string]any{"role": "user", "content": "hi"},
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
	}
	out := ShapeStrict(body)
	if _, ok := out["cache_control"]; ok {
		t.Fatal("top-level cache_control should be dropped")
	}
	msgs := out["messages"].([]any)
	sys := msgs[2].(map[string]any)
	if sys["content"] != "reminder" {
		t.Fatalf("system content: %#v", sys["content"])
	}
}

func TestShapeStrictStripsUserBlocksButKeepsList(t *testing.T) {
	body := map[string]any{
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":          "text",
						"text":          "hi",
						"cache_control": map[string]any{"type": "ephemeral"},
					},
					map[string]any{"type": "thinking", "thinking": "secret"},
				},
			},
		},
	}
	out := ShapeStrict(body)
	msgs := out["messages"].([]any)
	content := msgs[0].(map[string]any)["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected thinking dropped, got %#v", content)
	}
	block := content[0].(map[string]any)
	if _, ok := block["cache_control"]; ok {
		t.Fatalf("cache_control leaked: %#v", block)
	}
	if block["text"] != "hi" {
		t.Fatalf("%#v", block)
	}
}

func TestShapeStrictCleansTools(t *testing.T) {
	body := map[string]any{
		"tools": []any{
			map[string]any{
				"type":          "function",
				"cache_control": map[string]any{"type": "ephemeral"},
				"function": map[string]any{
					"name":          "lookup",
					"cache_control": map[string]any{"type": "ephemeral"},
				},
			},
		},
	}
	out := ShapeStrict(body)
	tool := out["tools"].([]any)[0].(map[string]any)
	if _, ok := tool["cache_control"]; ok {
		t.Fatalf("%#v", tool)
	}
	fn := tool["function"].(map[string]any)
	if _, ok := fn["cache_control"]; ok {
		t.Fatalf("%#v", fn)
	}
	if fn["name"] != "lookup" {
		t.Fatalf("%#v", fn)
	}
}

func TestStrictProvider(t *testing.T) {
	if !StrictProvider("cerebras") || !StrictProvider("groq") {
		t.Fatal("cerebras and groq should be strict")
	}
	if StrictProvider("openai") || StrictProvider("anthropic") {
		t.Fatal("openai/anthropic should not be strict")
	}
}
