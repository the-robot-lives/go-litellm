package providers

import (
	"strings"

	"github.com/noizu-labs/go-litellm/internal/errx"
	"github.com/noizu-labs/go-litellm/internal/jsonx"
)

// DashScope native (non-OpenAI-compatible) text-generation adapter.
//
// Upstream shape: POST {base}/api/v1/services/aigc/text-generation/generation
// with {model, input:{messages}, parameters:{…}}; streaming requires the
// X-DashScope-SSE: enable header and result_format=message frames (the first
// frame replays the full prompt history — see ChunkParser's role guard).
//
// Thinking logic lives HERE, not in core.Prepare: enable_thinking /
// thinking_budget are derived from reasoning_effort / the Anthropic-style
// thinking object per request. The zai/wafer hooks stay untouched.
const (
	dashscopeDefaultBase = "https://dashscope.aliyuncs.com"
	dashscopeChatPath    = "/api/v1/services/aigc/text-generation/generation"
)

// DashScopeAdapter is the native DashScope adapter.
type DashScopeAdapter struct{}

func (DashScopeAdapter) Name() string { return "dashscope" }

func (DashScopeAdapter) SupportedParams(_ string) []string {
	var out []string
	for _, k := range DefaultSupported {
		switch k {
		case "logit_bias", "logprobs", "top_logprobs":
			continue
		}
		out = append(out, k)
	}
	return append(out, "top_k")
}

// MapOpenAIParams allowlists through the supported OpenAI surface (same body
// as Compat.MapOpenAIParams); transform decisions happen in TransformRequest.
func (a DashScopeAdapter) MapOpenAIParams(nonDefault, optional map[string]any, model string, drop bool) map[string]any {
	supported := map[string]bool{}
	for _, k := range a.SupportedParams(model) {
		supported[k] = true
	}
	out := jsonx.Clone(optional)
	for k, v := range nonDefault {
		switch {
		case supported[k]:
			out[k] = v
		case drop:
			// drop
		default:
			out[k] = v
		}
	}
	return out
}

func (DashScopeAdapter) ValidateEnvironment(req *Request, headers map[string]string) (map[string]string, *errx.Error) {
	if headers == nil {
		headers = map[string]string{}
	}
	headers["content-type"] = "application/json"
	key := resolveAPIKey(req.LiteLLMParams, "DASHSCOPE_API_KEY")
	if key == "" {
		return nil, errx.New(401, "no DashScope API key (checked litellm_params.api_key and DASHSCOPE_API_KEY)",
			errx.WithType("authentication_error"), errx.WithProvider("dashscope"))
	}
	headers["authorization"] = "Bearer " + key
	if req.Stream {
		// Native SSE only streams with this explicit opt-in header.
		headers["x-dashscope-sse"] = "enable"
	}
	if extra := jsonx.Nested(req.LiteLLMParams, "extra_headers"); extra != nil {
		for k, v := range extra {
			headers[k] = stringify(v)
		}
	}
	return headers, nil
}

func (DashScopeAdapter) CompleteURL(req *Request) string {
	base := strings.TrimRight(jsonx.Str(req.LiteLLMParams, "api_base"), "/")
	if base == "" {
		base = dashscopeDefaultBase
	}
	// Suffix-tolerant: an api_base already ending in the generation path wins.
	if strings.HasSuffix(base, dashscopeChatPath) {
		return base
	}
	return base + dashscopeChatPath
}

// TransformRequest maps the internal OpenAI params onto the native
// {model, input, parameters} body. Notable decisions:
//   - max_completion_tokens (else max_tokens) → parameters.max_completion_tokens;
//     the native deprecated max_tokens is never emitted.
//   - reasoning_effort {low,medium,high,max} or thinking.type ≠ disabled →
//     enable_thinking true; {none,minimal} or thinking disabled → false.
//     Explicit effort wins over the thinking object (zai.go convention).
//   - thinking.budget_tokens → thinking_budget.
//   - tool_choice "required" → "auto" (unsupported upstream); tools pass
//     through in OpenAI shape (accepted natively).
//   - result_format is forced to "message"; streaming sets incremental_output.
func (DashScopeAdapter) TransformRequest(req *Request) map[string]any {
	p := req.Params
	parameters := map[string]any{"result_format": "message"}
	copyParam(parameters, p, "temperature", "")
	copyParam(parameters, p, "top_p", "")
	copyParam(parameters, p, "top_k", "")
	copyParam(parameters, p, "seed", "")
	copyParam(parameters, p, "stop", "")
	if v, ok := p["max_completion_tokens"]; ok && v != nil {
		parameters["max_completion_tokens"] = v
	} else if v, ok := p["max_tokens"]; ok && v != nil {
		parameters["max_completion_tokens"] = v
	}
	if effort, ok := p["reasoning_effort"].(string); ok {
		switch strings.ToLower(strings.TrimSpace(effort)) {
		case "low", "medium", "high", "max":
			parameters["enable_thinking"] = true
		case "none", "minimal":
			parameters["enable_thinking"] = false
		}
	} else if thinking := jsonx.AsMap(p["thinking"]); thinking != nil {
		switch jsonx.Str(thinking, "type") {
		case "disabled":
			parameters["enable_thinking"] = false
		case "":
			// thinking object without a type carries no on/off signal.
		default:
			parameters["enable_thinking"] = true
		}
	}
	if thinking := jsonx.AsMap(p["thinking"]); thinking != nil {
		if b, ok := jsonx.Int(thinking, "budget_tokens"); ok {
			parameters["thinking_budget"] = b
		}
	}
	if v, ok := p["tools"]; ok && v != nil {
		parameters["tools"] = v
	}
	if v, ok := p["tool_choice"]; ok && v != nil {
		if s, isStr := v.(string); isStr && s == "required" {
			parameters["tool_choice"] = "auto"
		} else {
			parameters["tool_choice"] = v
		}
	}
	if req.Stream {
		parameters["incremental_output"] = true
	}
	return map[string]any{
		"model":      req.Model,
		"input":      map[string]any{"messages": req.Messages},
		"parameters": parameters,
	}
}

func (DashScopeAdapter) TransformResponse(raw map[string]any, req *Request) *ModelResponse {
	output := jsonx.AsMap(raw["output"])
	choice := dashscopeFirstChoice(output)
	message := map[string]any{"role": "assistant"}
	var content any = ""
	if msg := jsonx.AsMap(choice["message"]); msg != nil {
		if msg["content"] != nil {
			content = msg["content"]
		}
		if r := jsonx.Str(msg, "reasoning_content"); r != "" {
			message["reasoning_content"] = r
		}
		if tc := msg["tool_calls"]; tc != nil {
			message["tool_calls"] = tc
		}
	}
	message["content"] = content
	model := jsonx.Str(raw, "model")
	if model == "" {
		model = req.Model
	}
	return NewModelResponse(map[string]any{
		"id":    raw["request_id"],
		"model": model,
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       message,
				"finish_reason": mapDashscopeFinish(jsonx.Str(choice, "finish_reason")),
			},
		},
		"usage": dashscopeUsage(jsonx.AsMap(raw["usage"])),
	})
}

// ErrorClass parses the FLAT DashScope envelope {request_id, code, message}
// (no "error" wrapper).
func (DashScopeAdapter) ErrorClass(status int, body any, _ map[string]string) *errx.Error {
	msg := "dashscope provider error"
	code := ""
	if m := jsonx.AsMap(body); m != nil {
		code = jsonx.Str(m, "code")
		if s := jsonx.Str(m, "message"); s != "" {
			msg = s
		}
	} else if s, ok := body.(string); ok && s != "" {
		msg = s
	}
	if code != "" {
		msg = code + ": " + msg
	}
	return errx.New(status, msg, errx.WithProvider("dashscope"))
}

// ChunkParser reads output.choices[0] from a result_format=message SSE frame.
//
// History-frame guard: the first frame replays the full prompt history and
// every replayed message carries a role; incremental deltas carry only
// content / reasoning_content / tool_calls. Role-bearing frames are skipped
// ({Raw: event}). Usage is surfaced on every frame (cumulative — emitters
// overwrite, the final frame wins). A flat error frame mid-stream (has code,
// no output) closes the stream via Done.
func (DashScopeAdapter) ChunkParser(event any) StreamChunk {
	if event == nil {
		return StreamChunk{Done: true}
	}
	m := jsonx.AsMap(event)
	if m == nil {
		return StreamChunk{Raw: event}
	}
	if _, hasOutput := m["output"]; !hasOutput {
		if jsonx.Str(m, "code") != "" {
			return StreamChunk{Done: true, Raw: event}
		}
		return StreamChunk{Raw: event}
	}
	output := jsonx.AsMap(m["output"])
	if output == nil {
		return StreamChunk{Raw: event}
	}
	choice := dashscopeFirstChoice(output)
	if choice == nil {
		return StreamChunk{Raw: event}
	}
	msg := jsonx.AsMap(choice["message"])
	if msg == nil {
		msg = map[string]any{}
	}
	if jsonx.Str(msg, "role") != "" {
		return StreamChunk{Raw: event} // history replay frame — skip
	}
	finish := firstNonNil(choice["finish_reason"], output["finish_reason"])
	idx, _ := jsonx.Int(choice, "index")
	return StreamChunk{
		Text:         jsonx.Str(msg, "content"),
		Reasoning:    jsonx.Str(msg, "reasoning_content"),
		ToolUse:      msg["tool_calls"], // OpenAI delta.tool_calls list shape — verbatim
		IsFinished:   finish != nil,
		FinishReason: mapDashscopeFinish(dashscopeFinishString(finish)),
		Usage:        m["usage"],
		Index:        idx,
		Raw:          event,
	}
}

func dashscopeFirstChoice(output map[string]any) map[string]any {
	if output == nil {
		return nil
	}
	choices, _ := output["choices"].([]any)
	if len(choices) == 0 {
		return nil
	}
	return jsonx.AsMap(choices[0])
}

func dashscopeFinishString(finish any) string {
	if s, ok := finish.(string); ok {
		return s
	}
	return ""
}

// mapDashscopeFinish normalizes native finish reasons onto the OpenAI set.
func mapDashscopeFinish(r string) any {
	switch r {
	case "stop":
		return "stop"
	case "length":
		return "length"
	case "tool_calls", "tool_use":
		return "tool_calls"
	case "":
		return nil
	default:
		return r
	}
}

// dashscopeUsage renames the native usage fields onto OpenAI's.
func dashscopeUsage(u map[string]any) any {
	if u == nil {
		return nil
	}
	in, _ := jsonx.Int(u, "input_tokens")
	out, _ := jsonx.Int(u, "output_tokens")
	total, ok := jsonx.Int(u, "total_tokens")
	if !ok {
		total = in + out
	}
	return map[string]any{
		"prompt_tokens":     in,
		"completion_tokens": out,
		"total_tokens":      total,
	}
}

// DashScope is the registered DashScope adapter.
var DashScope Adapter = DashScopeAdapter{}
