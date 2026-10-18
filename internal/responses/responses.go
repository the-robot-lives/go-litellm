// Package responses implements the inbound OpenAI Responses surface
// (POST /v1/responses) as a stateless translation layer over chat
// completions: request bodies translate to OpenAI chat bodies, upstream
// ModelResponses translate back to a Responses object or a Responses SSE
// stream.
//
// Stateless contract (v1): previous_response_id and store are explicit 400s,
// item_reference input items are rejected, nothing is ever persisted, and
// reasoning items/deltas are silently dropped (no reasoning output in v1).
package responses

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/noizu-labs/go-litellm/internal/errx"
	"github.com/noizu-labs/go-litellm/internal/jsonx"
	"github.com/noizu-labs/go-litellm/internal/providers"
)

// ValidateStateless rejects Responses request fields that imply server-side
// state. Each rejection is a 400 invalid_request_error naming the field.
func ValidateStateless(body map[string]any) *errx.Error {
	if _, ok := body["previous_response_id"]; ok {
		return errx.New(400,
			"previous_response_id is not supported: this gateway is stateless and stores no responses",
			errx.WithType("invalid_request_error"), errx.WithParam("previous_response_id"))
	}
	if _, ok := body["store"]; ok {
		// Any explicit value (even false) is refused: the gateway never persists.
		return errx.New(400,
			"store is not supported: this gateway is stateless and stores no responses",
			errx.WithType("invalid_request_error"), errx.WithParam("store"))
	}
	if items, ok := body["input"].([]any); ok {
		for _, item := range items {
			if m := jsonx.AsMap(item); m != nil && jsonx.Str(m, "type") == "item_reference" {
				return errx.New(400,
					"item_reference input items are not supported: this gateway is stateless and stores no responses",
					errx.WithType("invalid_request_error"), errx.WithParam("input"))
			}
		}
	}
	return nil
}

// RequestToOpenAI translates a Responses request body into an OpenAI chat
// completions body. instructions becomes a prepended system message; input
// (string or item list) becomes messages; the flat tool shape nests; params
// rename (max_output_tokens → max_completion_tokens, reasoning.effort →
// reasoning_effort, text.format → response_format, metadata.user → user).
// truncation / include / background / service_tier are dropped silently.
func RequestToOpenAI(body map[string]any) map[string]any {
	var msgs []any
	if instr := jsonx.Str(body, "instructions"); instr != "" {
		msgs = append(msgs, map[string]any{"role": "system", "content": instr})
	}
	switch input := body["input"].(type) {
	case string:
		msgs = append(msgs, map[string]any{"role": "user", "content": input})
	case []any:
		for _, item := range input {
			msgs = append(msgs, inputItem(jsonx.AsMap(item))...)
		}
	}
	out := map[string]any{
		"model":    body["model"],
		"messages": msgs,
	}
	putTools(out, body["tools"])
	copyIf(out, body, "tool_choice")
	copyIf(out, body, "temperature")
	copyIf(out, body, "top_p")
	copyIf(out, body, "parallel_tool_calls")
	copyIf(out, body, "stream")
	if v, ok := body["max_output_tokens"]; ok && v != nil {
		out["max_completion_tokens"] = v
	}
	if reasoning := jsonx.AsMap(body["reasoning"]); reasoning != nil {
		if effort := jsonx.Str(reasoning, "effort"); effort != "" {
			out["reasoning_effort"] = effort
		}
	}
	if text := jsonx.AsMap(body["text"]); text != nil {
		if format := text["format"]; format != nil {
			out["response_format"] = format
		}
	}
	if metadata := jsonx.AsMap(body["metadata"]); metadata != nil {
		if user := jsonx.Str(metadata, "user"); user != "" {
			out["user"] = user
		}
	}
	return out
}

// inputItem translates one Responses input item into zero or more chat
// messages. Message items (and the bare {role, content} convenience form)
// pass their role through verbatim — developer stays developer.
func inputItem(m map[string]any) []any {
	if m == nil {
		return nil
	}
	switch jsonx.Str(m, "type") {
	case "function_call":
		return []any{map[string]any{
			"role":    "assistant",
			"content": nil,
			"tool_calls": []any{map[string]any{
				"id":   m["call_id"],
				"type": "function",
				"function": map[string]any{
					"name":      m["name"],
					"arguments": argumentsString(m["arguments"]),
				},
			}},
		}}
	case "function_call_output":
		return []any{map[string]any{
			"role":         "tool",
			"tool_call_id": m["call_id"],
			"content":      outputContent(m["output"]),
		}}
	case "reasoning":
		// Reasoning items are silently dropped (stateless v1 contract).
		return nil
	}
	role := jsonx.Str(m, "role")
	if role == "" {
		return nil
	}
	return []any{map[string]any{"role": role, "content": messageContent(m["content"])}}
}

// messageContent passes strings through and translates content blocks:
// input_text/output_text → {type:"text", text}, input_image → {type:
// "image_url", image_url:{url}}; every other block type (thinking, summary,
// …) is dropped.
func messageContent(v any) any {
	switch t := v.(type) {
	case nil:
		return nil
	case string:
		return t
	case []any:
		var parts []any
		for _, b := range t {
			m := jsonx.AsMap(b)
			if m == nil {
				if s, ok := b.(string); ok {
					parts = append(parts, map[string]any{"type": "text", "text": s})
				}
				continue
			}
			switch jsonx.Str(m, "type") {
			case "input_text", "output_text":
				parts = append(parts, map[string]any{"type": "text", "text": m["text"]})
			case "input_image":
				parts = append(parts, map[string]any{
					"type":      "image_url",
					"image_url": map[string]any{"url": m["image_url"]},
				})
			}
		}
		return parts
	default:
		return t
	}
}

// outputContent normalizes a function_call_output payload into chat tool
// content: strings pass through; block lists flatten to their text joined by
// newlines; anything else JSON-encodes.
func outputContent(v any) any {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case []any:
		var parts []string
		for _, b := range t {
			if s, ok := b.(string); ok {
				parts = append(parts, s)
				continue
			}
			if m := jsonx.AsMap(b); m != nil {
				switch jsonx.Str(m, "type") {
				case "output_text", "input_text", "text":
					parts = append(parts, jsonx.Str(m, "text"))
					continue
				}
			}
			raw, _ := json.Marshal(b)
			parts = append(parts, string(raw))
		}
		return strings.Join(parts, "\n")
	default:
		raw, _ := json.Marshal(t)
		return string(raw)
	}
}

// argumentsString normalizes tool arguments to the JSON-string form chat
// completions carries.
func argumentsString(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	default:
		b, err := json.Marshal(t)
		if err != nil {
			return ""
		}
		return string(b)
	}
}

// putTools nests the Responses flat tool shape {type:"function", name,
// description, parameters, strict} into the chat-completions shape; only
// type "function" tools are kept.
func putTools(body map[string]any, tools any) {
	list, ok := tools.([]any)
	if !ok {
		return
	}
	var out []any
	for _, t := range list {
		m := jsonx.AsMap(t)
		if m == nil || jsonx.Str(m, "type") != "function" {
			continue
		}
		fn := map[string]any{"name": m["name"]}
		for _, k := range []string{"description", "parameters", "strict"} {
			if v, ok := m[k]; ok && v != nil {
				fn[k] = v
			}
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	if out != nil {
		body["tools"] = out
	}
}

func copyIf(acc, body map[string]any, key string) {
	if v, ok := body[key]; ok && v != nil {
		acc[key] = v
	}
}

// FromModelResponse translates an upstream chat completion into a Responses
// response object. Text becomes one message item (even when empty, mirroring
// the stream builder's empty-stream fallback) and every tool_call a
// function_call item; finish length/content_filter map to status incomplete
// + incomplete_details. Usage renames prompt/completion → input/output
// tokens. There is never a flat output_text field and never a store echo.
func FromModelResponse(r *providers.ModelResponse, requestedModel string) map[string]any {
	var choice, usage map[string]any
	if r != nil {
		if len(r.Choices) > 0 {
			choice = jsonx.AsMap(r.Choices[0])
		}
		usage = jsonx.AsMap(r.Usage)
	}
	if choice == nil {
		choice = map[string]any{}
	}
	status, incomplete := "completed", any(nil)
	switch jsonx.Str(choice, "finish_reason") {
	case "length":
		status, incomplete = "incomplete", map[string]any{"reason": "max_output_tokens"}
	case "content_filter":
		status, incomplete = "incomplete", map[string]any{"reason": "content_filter"}
	}
	message := jsonx.AsMap(choice["message"])
	if message == nil {
		message = map[string]any{}
	}
	output := []any{messageItem(
		"msg_"+providers.RandID(12), "completed",
		[]any{textPart(jsonx.Str(message, "content"))},
	)}
	for _, tc := range asList(message["tool_calls"]) {
		m := jsonx.AsMap(tc)
		if m == nil {
			continue
		}
		fun := jsonx.AsMap(m["function"])
		callID := jsonx.Str(m, "id")
		if callID == "" {
			callID = "call_" + providers.RandID(12)
		}
		output = append(output, toolItem(
			"fc_"+providers.RandID(12), callID,
			jsonx.Str(fun, "name"), argumentsString(fun["arguments"]), "completed",
		))
	}
	return map[string]any{
		"id":                 "resp_" + providers.RandID(16),
		"object":             "response",
		"created_at":         time.Now().Unix(),
		"status":             status,
		"model":              requestedModel,
		"output":             output,
		"usage":              usageOf(usage),
		"error":              nil,
		"incomplete_details": incomplete,
		"metadata":           map[string]any{},
	}
}

// Shared item builders (used by both the JSON and SSE translators).

func messageItem(id, status string, content []any) map[string]any {
	return map[string]any{
		"type": "message", "id": id, "role": "assistant",
		"status": status, "content": content,
	}
}

func textPart(text string) map[string]any {
	return map[string]any{"type": "output_text", "text": text, "annotations": []any{}}
}

func toolItem(id, callID, name, args, status string) map[string]any {
	return map[string]any{
		"type": "function_call", "id": id, "call_id": callID,
		"name": name, "arguments": args, "status": status,
	}
}

// usageOf renames OpenAI usage {prompt_tokens, completion_tokens,
// total_tokens} to Responses usage {input_tokens, output_tokens,
// total_tokens}; a missing total falls back to in+out, missing usage to zeros.
func usageOf(u map[string]any) map[string]any {
	in, _ := jsonx.Int(u, "prompt_tokens")
	out, _ := jsonx.Int(u, "completion_tokens")
	total, ok := jsonx.Int(u, "total_tokens")
	if !ok {
		total = in + out
	}
	return map[string]any{"input_tokens": in, "output_tokens": out, "total_tokens": total}
}

func asList(v any) []any {
	switch t := v.(type) {
	case nil:
		return nil
	case []any:
		return t
	default:
		if m := jsonx.AsMap(t); m != nil {
			return []any{m}
		}
		return nil
	}
}

func stringify(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	b, _ := json.Marshal(v)
	return strings.Trim(string(b), `"`)
}
