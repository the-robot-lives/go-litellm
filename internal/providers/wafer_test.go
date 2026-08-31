package providers

import (
	"reflect"
	"testing"
)

func TestIsWaferBase(t *testing.T) {
	cases := []struct {
		base string
		want bool
	}{
		{"https://pass.wafer.ai/v1", true},
		{"https://pass.wafer.ai", true},
		{"https://api.anthropic.com/v1", false},
		{"https://api.z.ai/api/anthropic", false},
		{"", false},
	}
	for _, c := range cases {
		if got := IsWaferBase(c.base); got != c.want {
			t.Errorf("IsWaferBase(%q) = %v, want %v", c.base, got, c.want)
		}
	}
}

func TestWaferThinkingApplies(t *testing.T) {
	cases := []struct {
		name string
		lp   map[string]any
		want bool
	}{
		{"wafer host default on", map[string]any{"api_base": "https://pass.wafer.ai/v1"}, true},
		{"opt out", map[string]any{"api_base": "https://pass.wafer.ai/v1", "wafer_thinking_mapping": false}, false},
		{"explicit opt in", map[string]any{"api_base": "https://pass.wafer.ai/v1", "wafer_thinking_mapping": true}, true},
		{"non-wafer host", map[string]any{"api_base": "https://api.anthropic.com/v1"}, false},
		{"no base", map[string]any{}, false},
		{"nil params", nil, false},
	}
	for _, c := range cases {
		if got := WaferThinkingApplies(c.lp); got != c.want {
			t.Errorf("%s: WaferThinkingApplies = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestApplyWaferThinking(t *testing.T) {
	thinking := func(typ string, budget any) map[string]any {
		m := map[string]any{"type": typ}
		if budget != nil {
			m["budget_tokens"] = budget
		}
		return m
	}

	cases := []struct {
		name   string
		model  string
		body   map[string]any
		lp     map[string]any
		want   map[string]any
		wantLP map[string]any
	}{
		{
			// GLM-5.3-class only accepts low|high|max: known values normalize.
			name:  "5.3 effort normalize",
			model: "glm-5.3",
			body:  map[string]any{"reasoning_effort": "medium"},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			name:  "5.3 unknown effort clamps high",
			model: "glm-5.3-flash",
			body:  map[string]any{"reasoning_effort": "ultra"},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			// GLM-5.2 keeps the full effort set untouched.
			name:  "5.2 effort untouched",
			model: "glm-5.2",
			body:  map[string]any{"reasoning_effort": "xhigh"},
			want:  map[string]any{"reasoning_effort": "xhigh"},
		},
		{
			// Kimi: pass explicit reasoning_effort through as-is.
			name:  "kimi effort passthrough",
			model: "kimi-k2-thinking",
			body:  map[string]any{"reasoning_effort": "Medium"},
			want:  map[string]any{"reasoning_effort": "Medium"},
		},
		{
			// Non-GLM, non-kimi models are left untouched.
			name:  "unknown model effort untouched",
			model: "some-other-model",
			body:  map[string]any{"reasoning_effort": "ultra"},
			want:  map[string]any{"reasoning_effort": "ultra"},
		},
		{
			// Explicit effort wins over the anthropic thinking object.
			name:  "effort wins over thinking",
			model: "glm-5.3",
			body:  map[string]any{"reasoning_effort": "low", "thinking": thinking("enabled", 32000)},
			want:  map[string]any{"reasoning_effort": "low"},
		},
		{
			// Anthropic thinking.budget_tokens buckets onto the effort scale.
			name:  "budget 1024 maps low",
			model: "glm-5.3",
			body:  map[string]any{"thinking": thinking("enabled", 1024)},
			want:  map[string]any{"reasoning_effort": "low"},
		},
		{
			name:  "budget 2048 boundary low",
			model: "glm-5.3",
			body:  map[string]any{"thinking": thinking("enabled", 2048)},
			want:  map[string]any{"reasoning_effort": "low"},
		},
		{
			name:  "budget 2049 maps high",
			model: "glm-5.3",
			body:  map[string]any{"thinking": thinking("enabled", 2049)},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			name:  "budget 8192 boundary high",
			model: "glm-5.3",
			body:  map[string]any{"thinking": thinking("enabled", 8192)},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			name:  "budget 8193 maps max",
			model: "glm-5.3",
			body:  map[string]any{"thinking": thinking("enabled", 8193)},
			want:  map[string]any{"reasoning_effort": "max"},
		},
		{
			name:  "enabled without budget maps high",
			model: "glm-5.3",
			body:  map[string]any{"thinking": thinking("enabled", nil)},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			name:  "disabled coerced low",
			model: "glm-5.3-flash",
			body:  map[string]any{"thinking": thinking("disabled", nil)},
			want:  map[string]any{"reasoning_effort": "low"},
		},
		{
			// Kimi: thinking translated to a best-effort effort; upstream decides.
			name:  "kimi enabled thinking maps high",
			model: "kimi-k2-thinking",
			body:  map[string]any{"thinking": thinking("enabled", nil)},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			name:  "kimi disabled thinking maps low",
			model: "Kimi-K2",
			body:  map[string]any{"thinking": thinking("disabled", nil)},
			want:  map[string]any{"reasoning_effort": "low"},
		},
		{
			// Deployment default seeds a silent body.
			name:   "default effort used when body silent",
			model:  "glm-5.3",
			body:   map[string]any{"model": "glm-5.3", "messages": []any{}},
			lp:     map[string]any{"api_base": "https://pass.wafer.ai/v1", "default_reasoning_effort": "high"},
			want:   map[string]any{"model": "glm-5.3", "messages": []any{}, "reasoning_effort": "high"},
			wantLP: map[string]any{"api_base": "https://pass.wafer.ai/v1"},
		},
		{
			// Explicit body effort beats the deployment default.
			name:   "default not applied when body has effort",
			model:  "glm-5.3",
			body:   map[string]any{"reasoning_effort": "low"},
			lp:     map[string]any{"default_reasoning_effort": "max"},
			want:   map[string]any{"reasoning_effort": "low"},
			wantLP: map[string]any{},
		},
		{
			// Thinking object beats the deployment default; key still consumed.
			name:   "default not applied when body has thinking",
			model:  "glm-5.3",
			body:   map[string]any{"thinking": thinking("enabled", 1024)},
			lp:     map[string]any{"default_reasoning_effort": "max"},
			want:   map[string]any{"reasoning_effort": "low"},
			wantLP: map[string]any{},
		},
		{
			// The default key is consumed even when nothing needs it.
			name:   "default key removed from lp with silent body",
			model:  "glm-5.2",
			body:   map[string]any{"messages": []any{}},
			lp:     map[string]any{"default_reasoning_effort": ""},
			want:   map[string]any{"messages": []any{}},
			wantLP: map[string]any{},
		},
		{
			name:  "unrelated body untouched",
			model: "glm-5.3",
			body:  map[string]any{"model": "glm-5.3", "messages": []any{}},
			want:  map[string]any{"model": "glm-5.3", "messages": []any{}},
		},
	}

	for _, c := range cases {
		got := ApplyWaferThinking(cloneMap(c.body), c.model, c.lp)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
		if c.wantLP != nil && !reflect.DeepEqual(c.lp, c.wantLP) {
			t.Errorf("%s: lp = %v, want %v", c.name, c.lp, c.wantLP)
		}
	}
}

func TestAnthropicWaferEffortForwarded(t *testing.T) {
	req := &Request{
		Model:    "glm-5.3",
		Provider: "anthropic",
		Params:   map[string]any{"reasoning_effort": "max"},
		LiteLLMParams: map[string]any{
			"api_base": "https://pass.wafer.ai/v1",
			"model":    "anthropic/glm-5.3",
		},
	}
	body := Anthropic.TransformRequest(req)
	if body["reasoning_effort"] != "max" {
		t.Errorf("wafer route: reasoning_effort = %v, want max", body["reasoning_effort"])
	}

	req.LiteLLMParams["api_base"] = "https://api.anthropic.com/v1"
	body = Anthropic.TransformRequest(req)
	if _, ok := body["reasoning_effort"]; ok {
		t.Errorf("non-wafer route: reasoning_effort should not be forwarded, got %v", body["reasoning_effort"])
	}
}
