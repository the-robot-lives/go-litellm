package providers

import (
	"strings"

	"github.com/noizu-labs/go-litellm/internal/jsonx"
)

// StrictProvider reports providers whose chat APIs reject Anthropic extras
// (cache_control, provider_specific_fields) and require system content as a string.
func StrictProvider(name string) bool {
	switch name {
	case "groq", "cerebras":
		return true
	default:
		return false
	}
}

var strictStripFields = map[string]bool{
	"cache_control":            true,
	"provider_specific_fields": true,
	"output_config":            true,
}

var strictMessageStrip = map[string]bool{
	"cache_control":            true,
	"provider_specific_fields": true,
	"output_config":            true,
	"thinking_blocks":          true,
	"reasoning_content":        true,
}

// ShapeStrict sanitizes a chat body for Groq/Cerebras.
//
// Claude Code sends Anthropic content blocks with cache_control on system
// reminders. Cerebras validates SystemMessage.content as string | text/image
// parts and 400s with:
//
//	messages.N.system.content.str: Input should be a valid string
//	messages.N.system.content.list[...].0.text.cache_control: unsupported
func ShapeStrict(body map[string]any) map[string]any {
	if body == nil {
		body = map[string]any{}
	}
	if msgs, ok := body["messages"].([]any); ok {
		body["messages"] = cleanStrictMessages(msgs)
	}
	if tools, ok := body["tools"].([]any); ok {
		body["tools"] = cleanStrictTools(tools)
	}
	for k := range strictStripFields {
		delete(body, k)
	}
	return body
}

func cleanStrictMessages(msgs []any) []any {
	out := make([]any, 0, len(msgs))
	for _, raw := range msgs {
		m := jsonx.AsMap(raw)
		if m == nil {
			out = append(out, raw)
			continue
		}
		msg := make(map[string]any, len(m))
		for k, v := range m {
			if !strictMessageStrip[k] {
				msg[k] = v
			}
		}
		switch c := msg["content"].(type) {
		case []any:
			cleaned := cleanStrictBlocks(c)
			if jsonx.Str(msg, "role") == "system" {
				msg["content"] = flattenStrictText(cleaned)
			} else {
				msg["content"] = cleaned
			}
		case map[string]any:
			msg["content"] = stripStrictMap(c)
		}
		out = append(out, msg)
	}
	return out
}

func cleanStrictBlocks(blocks []any) []any {
	out := make([]any, 0, len(blocks))
	for _, b := range blocks {
		m := jsonx.AsMap(b)
		if m == nil {
			out = append(out, b)
			continue
		}
		if jsonx.Str(m, "type") == "thinking" {
			continue
		}
		out = append(out, stripStrictMap(m))
	}
	return out
}

func stripStrictMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if !strictStripFields[k] {
			out[k] = v
		}
	}
	return out
}

func flattenStrictText(blocks []any) string {
	var parts []string
	for _, b := range blocks {
		if s, ok := b.(string); ok {
			parts = append(parts, s)
			continue
		}
		m := jsonx.AsMap(b)
		if m == nil {
			continue
		}
		if t := jsonx.Str(m, "type"); t == "text" || t == "" {
			parts = append(parts, jsonx.Str(m, "text"))
		}
	}
	return strings.Join(parts, "\n")
}

func cleanStrictTools(tools []any) []any {
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		m := jsonx.AsMap(t)
		if m == nil {
			out = append(out, t)
			continue
		}
		tool := jsonx.Clone(m)
		delete(tool, "cache_control")
		delete(tool, "provider_specific_fields")
		if fn := jsonx.AsMap(tool["function"]); fn != nil {
			fn = jsonx.Clone(fn)
			delete(fn, "cache_control")
			delete(fn, "provider_specific_fields")
			tool["function"] = fn
		}
		out = append(out, tool)
	}
	return out
}
