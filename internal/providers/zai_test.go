package providers

import (
	"reflect"
	"testing"
)

func TestIsZaiBase(t *testing.T) {
	cases := []struct {
		base string
		want bool
	}{
		{"https://api.z.ai/api/anthropic", true},
		{"https://open.bigmodel.cn/api/paas/v4", true},
		{"https://api.anthropic.com/v1", false},
		{"", false},
	}
	for _, c := range cases {
		if got := IsZaiBase(c.base); got != c.want {
			t.Errorf("IsZaiBase(%q) = %v, want %v", c.base, got, c.want)
		}
	}
}

func TestZaiThinkingApplies(t *testing.T) {
	cases := []struct {
		name string
		lp   map[string]any
		want bool
	}{
		{"zai host default on", map[string]any{"api_base": "https://api.z.ai/api/anthropic"}, true},
		{"bigmodel host default on", map[string]any{"api_base": "https://open.bigmodel.cn/api/paas/v4"}, true},
		{"opt out", map[string]any{"api_base": "https://api.z.ai/api/anthropic", "zai_thinking_mapping": false}, false},
		{"explicit opt in", map[string]any{"api_base": "https://api.z.ai/api/anthropic", "zai_thinking_mapping": true}, true},
		{"non-zai host", map[string]any{"api_base": "https://api.anthropic.com/v1"}, false},
		{"no base", map[string]any{}, false},
		{"nil params", nil, false},
	}
	for _, c := range cases {
		if got := ZaiThinkingApplies(c.lp); got != c.want {
			t.Errorf("%s: ZaiThinkingApplies = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestApplyZaiThinking(t *testing.T) {
	thinking := func(typ string, budget any) map[string]any {
		m := map[string]any{"type": typ}
		if budget != nil {
			m["budget_tokens"] = budget
		}
		return m
	}

	cases := []struct {
		name  string
		model string
		body  map[string]any
		want  map[string]any
	}{
		{
			// GLM-5.3 only accepts low|high|max: known values normalize.
			name:  "5.3 effort normalize",
			model: "glm-5.3",
			body:  map[string]any{"reasoning_effort": "medium"},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			name:  "5.3 effort none maps low",
			model: "glm-5.3-flash",
			body:  map[string]any{"reasoning_effort": "none"},
			want:  map[string]any{"reasoning_effort": "low"},
		},
		{
			name:  "5.3 effort xhigh maps max",
			model: "glm-5.3",
			body:  map[string]any{"reasoning_effort": "xhigh"},
			want:  map[string]any{"reasoning_effort": "max"},
		},
		{
			name:  "5.3 unknown effort clamps high",
			model: "glm-5.3",
			body:  map[string]any{"reasoning_effort": "ultra"},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			// GLM-5.2 keeps the full effort set untouched.
			name:  "5.2 keeps none",
			model: "glm-5.2",
			body:  map[string]any{"reasoning_effort": "none"},
			want:  map[string]any{"reasoning_effort": "none"},
		},
		{
			name:  "5.2 keeps medium",
			model: "glm-5.2",
			body:  map[string]any{"reasoning_effort": "medium"},
			want:  map[string]any{"reasoning_effort": "medium"},
		},
		{
			name:  "5.2 unknown clamps high",
			model: "glm-5.2",
			body:  map[string]any{"reasoning_effort": "ultra"},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			// Pre-5.2 models do not support reasoning_effort at all.
			name:  "pre-5.2 effort removed",
			model: "glm-5",
			body:  map[string]any{"reasoning_effort": "high"},
			want:  map[string]any{},
		},
		{
			name:  "non-glm effort removed",
			model: "some-other-model",
			body:  map[string]any{"reasoning_effort": "high"},
			want:  map[string]any{},
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
			// Disabled thinking cannot be honored on 5.3; degrade to low.
			name:  "disabled coerced low on 5.3",
			model: "glm-5.3-flash",
			body:  map[string]any{"thinking": thinking("disabled", nil)},
			want:  map[string]any{"reasoning_effort": "low"},
		},
		{
			// ≤5.2 era: z.ai honors thinking.type=disabled; leave untouched.
			name:  "disabled left on pre-5.3",
			model: "glm-5",
			body:  map[string]any{"thinking": thinking("disabled", nil), "max_tokens": 99},
			want:  map[string]any{"thinking": map[string]any{"type": "disabled"}, "max_tokens": 99},
		},
		{
			// Pre-5.2 has no effort surface; enabled thinking stays as-is.
			name:  "enabled thinking left on pre-5.2",
			model: "glm-5.1",
			body:  map[string]any{"thinking": thinking("enabled", 8192)},
			want:  map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": 8192}},
		},
		{
			// 5.2 converts enabled thinking to effort.
			name:  "enabled thinking converts on 5.2",
			model: "glm-5.2",
			body:  map[string]any{"thinking": thinking("enabled", 4096)},
			want:  map[string]any{"reasoning_effort": "high"},
		},
		{
			name:  "unrelated body untouched",
			model: "glm-5.3",
			body:  map[string]any{"model": "glm-5.3", "messages": []any{}},
			want:  map[string]any{"model": "glm-5.3", "messages": []any{}},
		},
	}

	for _, c := range cases {
		got := ApplyZaiThinking(cloneMap(c.body), c.model)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}

func cloneMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if vm, ok := v.(map[string]any); ok {
			out[k] = cloneMap(vm)
			continue
		}
		out[k] = v
	}
	return out
}

func TestAnthropicZaiEffortForwarded(t *testing.T) {
	req := &Request{
		Model:    "glm-5.3",
		Provider: "anthropic",
		Params:   map[string]any{"reasoning_effort": "max"},
		LiteLLMParams: map[string]any{
			"api_base": "https://api.z.ai/api/anthropic",
			"model":    "anthropic/glm-5.3",
		},
	}
	body := Anthropic.TransformRequest(req)
	if body["reasoning_effort"] != "max" {
		t.Errorf("zai route: reasoning_effort = %v, want max", body["reasoning_effort"])
	}

	req.LiteLLMParams["api_base"] = "https://api.anthropic.com/v1"
	body = Anthropic.TransformRequest(req)
	if _, ok := body["reasoning_effort"]; ok {
		t.Errorf("non-zai route: reasoning_effort should not be forwarded, got %v", body["reasoning_effort"])
	}
}
