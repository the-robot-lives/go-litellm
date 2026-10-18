package providers

import (
	"encoding/json"
	"testing"
)

func dashscopeReq(model string, params map[string]any, stream bool) *Request {
	return &Request{
		Model:         model,
		Provider:      "dashscope",
		Messages:      []any{map[string]any{"role": "user", "content": "hi"}},
		Params:        params,
		LiteLLMParams: map[string]any{},
		Stream:        stream,
		CallType:      "chat",
	}
}

func TestDashScopeRegistryResolution(t *testing.T) {
	res, e := Resolve("dashscope/qwen-max", map[string]any{"model": "dashscope/qwen-max"})
	if e != nil {
		t.Fatal(e)
	}
	if res.Provider != "dashscope" {
		t.Fatalf("provider=%s", res.Provider)
	}
	if res.Model != "qwen-max" {
		t.Fatalf("model=%s", res.Model)
	}
	if res.Adapter.Name() != "dashscope" {
		t.Fatalf("adapter=%s", res.Adapter.Name())
	}
}

func TestDashScopeEndpointClassification(t *testing.T) {
	p, _ := classify("some/model", map[string]any{
		"model":    "some/model",
		"api_base": "https://dashscope.aliyuncs.com",
	})
	if p != "dashscope" {
		t.Fatalf("cn host -> %s", p)
	}
	p, _ = classify("some/model", map[string]any{
		"model":    "some/model",
		"api_base": "https://dashscope-intl.aliyuncs.com",
	})
	if p != "dashscope" {
		t.Fatalf("intl host -> %s", p)
	}
	// NEGATIVE (R14): a bare qwen model on a compatible-mode api_base must NOT
	// classify as dashscope — no patternProvider case was added.
	p, _ = classify("qwen3.8-max", map[string]any{
		"model":    "qwen3.8-max",
		"api_base": "https://llm-host.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
	})
	if p == "dashscope" {
		t.Fatal("bare qwen on compatible-mode base misrouted to dashscope")
	}
	if p != "openai_compatible" {
		t.Fatalf("bare qwen compatible-mode -> %s, want openai_compatible", p)
	}
}

func TestDashScopeURL(t *testing.T) {
	req := &Request{Model: "qwen-max", Provider: "dashscope", LiteLLMParams: map[string]any{}, CallType: "chat"}
	if got := DashScope.CompleteURL(req); got != "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation" {
		t.Fatalf("default url=%s", got)
	}
	req.LiteLLMParams["api_base"] = "https://dashscope-intl.aliyuncs.com"
	if got := DashScope.CompleteURL(req); got != "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation" {
		t.Fatalf("custom url=%s", got)
	}
	req.LiteLLMParams["api_base"] = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation/"
	if got := DashScope.CompleteURL(req); got != "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation" {
		t.Fatalf("suffix-tolerant url=%s", got)
	}
}

func TestDashScopeValidateEnvironment(t *testing.T) {
	t.Setenv("DASHSCOPE_API_KEY", "")
	req := dashscopeReq("qwen-max", nil, false)
	req.LiteLLMParams["api_key"] = "test-dummy-dashscope"
	req.LiteLLMParams["extra_headers"] = map[string]any{"X-Trace": "t1"}
	headers, e := DashScope.ValidateEnvironment(req, map[string]string{})
	if e != nil {
		t.Fatal(e)
	}
	if headers["authorization"] != "Bearer test-dummy-dashscope" {
		t.Fatalf("authorization=%q", headers["authorization"])
	}
	if headers["content-type"] != "application/json" {
		t.Fatalf("content-type=%q", headers["content-type"])
	}
	if _, has := headers["x-dashscope-sse"]; has {
		t.Fatal("x-dashscope-sse set on non-stream request")
	}
	if headers["X-Trace"] != "t1" {
		t.Fatalf("extra_headers lost: %+v", headers)
	}

	// Stream request: x-dashscope-sse appears iff req.Stream.
	streamReq := dashscopeReq("qwen-max", nil, true)
	streamReq.LiteLLMParams["api_key"] = "test-dummy-dashscope"
	headers, e = DashScope.ValidateEnvironment(streamReq, map[string]string{})
	if e != nil {
		t.Fatal(e)
	}
	if headers["x-dashscope-sse"] != "enable" {
		t.Fatalf("stream header=%q", headers["x-dashscope-sse"])
	}
}

func TestDashScopeMissingKeyIs401(t *testing.T) {
	t.Setenv("DASHSCOPE_API_KEY", "")
	_, e := DashScope.ValidateEnvironment(dashscopeReq("qwen-max", nil, false), map[string]string{})
	if e == nil || e.Status != 401 {
		t.Fatalf("expected 401, got %+v", e)
	}
}

func TestDashScopeTransformRequestNonStream(t *testing.T) {
	req := dashscopeReq("qwen-max", map[string]any{
		"temperature":      0.7,
		"top_p":            0.9,
		"top_k":            50,
		"seed":             42,
		"stop":             []any{"END"},
		"max_tokens":       100,
		"presence_penalty": 0.1, // not in the parameters allowlist — dropped
	}, false)
	body := DashScope.TransformRequest(req)
	if body["model"] != "qwen-max" {
		t.Fatalf("model=%v", body["model"])
	}
	input, _ := body["input"].(map[string]any)
	if input == nil {
		t.Fatalf("input missing: %+v", body)
	}
	msgs, _ := input["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages=%v", input["messages"])
	}
	p, _ := body["parameters"].(map[string]any)
	if p == nil {
		t.Fatalf("parameters missing: %+v", body)
	}
	if p["temperature"] != 0.7 || p["top_p"] != 0.9 || p["top_k"] != 50 || p["seed"] != 42 {
		t.Fatalf("verbatim params: %+v", p)
	}
	stop, _ := p["stop"].([]any)
	if len(stop) != 1 || stop[0] != "END" {
		t.Fatalf("stop=%v", p["stop"])
	}
	// R1: deprecated native max_tokens must never be emitted.
	if p["max_completion_tokens"] != 100 {
		t.Fatalf("max_completion_tokens=%v", p["max_completion_tokens"])
	}
	if _, has := p["max_tokens"]; has {
		t.Fatal("native deprecated max_tokens emitted")
	}
	if p["result_format"] != "message" {
		t.Fatalf("result_format=%v", p["result_format"])
	}
	if _, has := p["incremental_output"]; has {
		t.Fatal("incremental_output on non-stream request")
	}
	if _, has := p["presence_penalty"]; has {
		t.Fatal("presence_penalty not dropped")
	}
	if _, has := p["enable_thinking"]; has {
		t.Fatal("enable_thinking without a thinking signal")
	}
}

func TestDashScopeTransformRequestStream(t *testing.T) {
	cases := []struct {
		name          string
		params        map[string]any
		wantThinking  bool // enable_thinking present
		wantEnabled   bool // its value when present
		wantBudget    int  // 0 = no thinking_budget expected
		wantToolChoce any  // nil = none expected
	}{
		{
			name: "effort_high",
			params: map[string]any{"reasoning_effort": "high",
				"thinking": map[string]any{"type": "enabled", "budget_tokens": 4096}},
			wantThinking: true, wantEnabled: true, wantBudget: 4096,
		},
		{name: "effort_none", params: map[string]any{"reasoning_effort": "none"}, wantThinking: true, wantEnabled: false},
		{name: "effort_minimal", params: map[string]any{"reasoning_effort": "minimal"}, wantThinking: true, wantEnabled: false},
		{name: "effort_medium", params: map[string]any{"reasoning_effort": "medium"}, wantThinking: true, wantEnabled: true},
		{name: "effort_max", params: map[string]any{"reasoning_effort": "max"}, wantThinking: true, wantEnabled: true},
		{name: "thinking_disabled", params: map[string]any{"thinking": map[string]any{"type": "disabled"}}, wantThinking: true, wantEnabled: false},
		{name: "effort_unsupported_dropped", params: map[string]any{"reasoning_effort": "xhigh"}, wantThinking: false},
	}
	for _, tc := range cases {
		tc.params["max_completion_tokens"] = 256
		tc.params["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{"name": "f"}}}
		if tc.wantToolChoce == nil {
			tc.params["tool_choice"] = "required"
		}
		req := dashscopeReq("qwen3.8-max", tc.params, true)
		body := DashScope.TransformRequest(req)
		p, _ := body["parameters"].(map[string]any)
		if p == nil {
			t.Fatalf("%s: parameters missing", tc.name)
		}
		if p["incremental_output"] != true {
			t.Fatalf("%s: incremental_output=%v", tc.name, p["incremental_output"])
		}
		if p["result_format"] != "message" {
			t.Fatalf("%s: result_format=%v", tc.name, p["result_format"])
		}
		if p["max_completion_tokens"] != 256 {
			t.Fatalf("%s: max_completion_tokens=%v", tc.name, p["max_completion_tokens"])
		}
		if _, has := p["max_tokens"]; has {
			t.Fatalf("%s: deprecated max_tokens emitted", tc.name)
		}
		if got, has := p["enable_thinking"]; has != tc.wantThinking || (has && got != tc.wantEnabled) {
			t.Fatalf("%s: enable_thinking=%v (want present=%v value=%v)", tc.name, p["enable_thinking"], tc.wantThinking, tc.wantEnabled)
		}
		if tc.wantBudget != 0 && p["thinking_budget"] != tc.wantBudget {
			t.Fatalf("%s: thinking_budget=%v want %d", tc.name, p["thinking_budget"], tc.wantBudget)
		}
		if tc.wantBudget == 0 {
			if _, has := p["thinking_budget"]; has {
				t.Fatalf("%s: unexpected thinking_budget=%v", tc.name, p["thinking_budget"])
			}
		}
		// tool_choice "required" → "auto"; tools pass through verbatim.
		if _, has := tc.params["tool_choice"]; has && p["tool_choice"] != "auto" {
			t.Fatalf("%s: tool_choice=%v want auto", tc.name, p["tool_choice"])
		}
		if _, has := p["tools"].([]any); !has {
			t.Fatalf("%s: tools lost: %+v", tc.name, p)
		}
	}
}

func TestDashScopeTransformResponse(t *testing.T) {
	raw := mustJSONMap(t, `{
		"request_id": "req-123",
		"model": "qwen-max",
		"output": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hello", "reasoning_content": "hmm"}}], "finish_reason": "stop"},
		"usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
	}`)
	resp := DashScope.TransformResponse(raw, &Request{Model: "qwen-max", Provider: "dashscope"})
	if resp.ID != "req-123" {
		t.Fatalf("id=%s", resp.ID)
	}
	if resp.Model != "qwen-max" {
		t.Fatalf("model=%s", resp.Model)
	}
	if resp.Object != "chat.completion" {
		t.Fatalf("object=%s", resp.Object)
	}
	choice, _ := resp.Choices[0].(map[string]any)
	if choice == nil {
		t.Fatalf("choice=%v", resp.Choices)
	}
	if choice["finish_reason"] != "stop" {
		t.Fatalf("finish_reason=%v", choice["finish_reason"])
	}
	msg, _ := choice["message"].(map[string]any)
	if msg["content"] != "hello" {
		t.Fatalf("content=%v", msg["content"])
	}
	if msg["reasoning_content"] != "hmm" {
		t.Fatalf("reasoning_content=%v", msg["reasoning_content"])
	}
	usage, _ := resp.Usage.(map[string]any)
	if usage["prompt_tokens"] != 5 || usage["completion_tokens"] != 7 || usage["total_tokens"] != 12 {
		t.Fatalf("usage=%v", resp.Usage)
	}

	// Tool calls pass through verbatim; finish tool_calls|tool_use → tool_calls;
	// missing model falls back to req.Model.
	raw = mustJSONMap(t, `{
		"request_id": "req-9",
		"output": {"choices": [{"finish_reason": "tool_use", "message": {"content": null, "tool_calls": [
			{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"sf\"}"}}
		]}}]},
		"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
	}`)
	resp = DashScope.TransformResponse(raw, &Request{Model: "qwen-max", Provider: "dashscope"})
	choice = resp.Choices[0].(map[string]any)
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason=%v", choice["finish_reason"])
	}
	msg = choice["message"].(map[string]any)
	tcs, _ := msg["tool_calls"].([]any)
	if len(tcs) != 1 {
		t.Fatalf("tool_calls=%v", msg["tool_calls"])
	}
	if tc := tcs[0].(map[string]any); tc["id"] != "call_1" {
		t.Fatalf("tool_call=%v", tc)
	}
	if resp.Model != "qwen-max" {
		t.Fatalf("model fallback=%s", resp.Model)
	}
}

func TestDashScopeErrorClass(t *testing.T) {
	// Flat envelope: {request_id, code, message} — NO error wrapper.
	body := mustJSONMap(t, `{"request_id": "r1", "code": "InvalidApiKey", "message": "invalid key"}`)
	e := DashScope.ErrorClass(401, body, nil)
	if e.Status != 401 {
		t.Fatalf("status=%d", e.Status)
	}
	if e.Message != "InvalidApiKey: invalid key" {
		t.Fatalf("message=%q", e.Message)
	}
	if e.Provider != "dashscope" {
		t.Fatalf("provider=%q", e.Provider)
	}
	// Code-only body and non-map bodies still produce a usable message.
	e = DashScope.ErrorClass(429, mustJSONMap(t, `{"code":"Throttling"}`), nil)
	if e.Message != "Throttling: dashscope provider error" {
		t.Fatalf("message=%q", e.Message)
	}
	e = DashScope.ErrorClass(500, "boom", nil)
	if e.Message != "boom" {
		t.Fatalf("message=%q", e.Message)
	}
}

func dashscopeFrame(t *testing.T, s string) any {
	t.Helper()
	return mustJSONMap(t, s)
}

func TestDashScopeChunkParserHistorySkip(t *testing.T) {
	// Frame 1: result_format=message first frame replays the FULL prompt
	// history — every replayed message carries a role → skipped.
	history := DashScope.ChunkParser(dashscopeFrame(t, `{
		"output": {"choices": [{"message": {"role": "user", "content": "hi"}}]},
		"request_id": "r", "usage": {"input_tokens": 3, "output_tokens": 0, "total_tokens": 3}
	}`))
	if history.Text != "" || history.Reasoning != "" || history.ToolUse != nil || history.IsFinished || history.Done {
		t.Fatalf("history frame leaked: %+v", history)
	}

	// Frame 2: content delta.
	delta := DashScope.ChunkParser(dashscopeFrame(t, `{
		"output": {"choices": [{"message": {"content": "Hel"}}]},
		"usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
	}`))
	if delta.Text != "Hel" {
		t.Fatalf("text=%q", delta.Text)
	}
	if delta.Usage == nil {
		t.Fatal("usage missing on delta frame (cumulative — emitters overwrite)")
	}

	// Frame 3: reasoning delta.
	reason := DashScope.ChunkParser(dashscopeFrame(t, `{
		"output": {"choices": [{"message": {"reasoning_content": "thinking"}}]}
	}`))
	if reason.Reasoning != "thinking" {
		t.Fatalf("reasoning=%q", reason.Reasoning)
	}

	// Frame 4: tool_calls delta fragments — OpenAI list shape passed verbatim.
	tools := DashScope.ChunkParser(dashscopeFrame(t, `{
		"output": {"choices": [{"message": {"tool_calls": [
			{"index": 0, "id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{\"a\""}}
		]}}]}
	}`))
	tcs, _ := tools.ToolUse.([]any)
	if len(tcs) != 1 {
		t.Fatalf("tool_calls=%v", tools.ToolUse)
	}
	if tc := tcs[0].(map[string]any); tc["id"] != "call_1" || tc["index"] != float64(0) {
		t.Fatalf("fragment=%v", tc)
	}

	// Frame 5: final — finish_reason (choice-level) + cumulative usage.
	final := DashScope.ChunkParser(dashscopeFrame(t, `{
		"output": {"choices": [{"finish_reason": "stop", "message": {"content": ""}}], "finish_reason": "stop"},
		"usage": {"input_tokens": 3, "output_tokens": 3, "total_tokens": 6}
	}`))
	if !final.IsFinished || final.FinishReason != "stop" {
		t.Fatalf("final=%+v", final)
	}
	usage, _ := final.Usage.(map[string]any)
	if usage == nil || usage["total_tokens"] != float64(6) {
		t.Fatalf("usage=%v", final.Usage)
	}

	// Output-level finish_reason fallback.
	alt := DashScope.ChunkParser(dashscopeFrame(t, `{
		"output": {"choices": [{"message": {"content": "x"}}], "finish_reason": "length"}
	}`))
	if !alt.IsFinished || alt.FinishReason != "length" {
		t.Fatalf("alt=%+v", alt)
	}

	// Flat error frame mid-stream (has code, no output) → Done.
	errFrame := DashScope.ChunkParser(dashscopeFrame(t, `{"code": "Throttling", "message": "slow down", "request_id": "r"}`))
	if !errFrame.Done {
		t.Fatalf("error frame not Done: %+v", errFrame)
	}
}

// mustJSONMap decodes an inline JSON literal the way the gateway decodes
// upstream bodies (encoding/json into any → map[string]any).
func mustJSONMap(t *testing.T, s string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatalf("bad literal: %v", err)
	}
	return m
}
