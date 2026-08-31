package providers

import (
	"strings"

	"github.com/noizu-labs/go-litellm/internal/jsonx"
)

// z.ai GLM reasoning contract (differs from Anthropic's thinking.budget_tokens):
//   - top-level reasoning_effort is the depth control surface
//   - GLM-5.3 / GLM-5.3-flash accept only low | high | max and cannot disable thinking
//   - GLM-5.2 accepts none | minimal | low | medium | high | xhigh | max
//   - pre-5.2 models (glm-5, glm-5.1, glm-4.x) do not support reasoning_effort at
//     all and only honor thinking.type

// IsZaiBase reports whether an api_base points at a z.ai / bigmodel endpoint.
func IsZaiBase(apiBase string) bool {
	return strings.Contains(apiBase, "z.ai") || strings.Contains(apiBase, "bigmodel.cn")
}

// ZaiThinkingApplies reports whether z.ai thinking mapping should run for a
// request routed through these litellm_params. Defaults ON for z.ai hosts;
// opt out per model with litellm_params.zai_thinking_mapping: false.
func ZaiThinkingApplies(lp map[string]any) bool {
	if lp == nil {
		return false
	}
	if b, ok := lp["zai_thinking_mapping"].(bool); ok && !b {
		return false
	}
	return IsZaiBase(jsonx.Str(lp, "api_base"))
}

// ApplyZaiThinking normalizes a request body's thinking controls to z.ai's
// reasoning_effort surface. body may be Anthropic-shaped (thinking object, as
// sent by Claude Code) or OpenAI-shaped (reasoning_effort, as sent by Codex and
// other OpenAI clients). Explicit effort wins over the thinking object. Mutates
// and returns body.
func ApplyZaiThinking(body map[string]any, model string) map[string]any {
	if body == nil {
		return body
	}
	if effort, hasEffort := body["reasoning_effort"].(string); hasEffort {
		// Explicit effort wins; the raw Anthropic thinking object is dropped.
		delete(body, "thinking")
		switch {
		case isZai53(model):
			body["reasoning_effort"] = zai53Effort(effort)
		case isZai52(model):
			body["reasoning_effort"] = zai52Effort(effort)
		default:
			// Unsupported pre-5.2 (glm-5, glm-5.1, glm-4.x, non-glm).
			delete(body, "reasoning_effort")
		}
		return body
	}
	thinking := jsonx.AsMap(body["thinking"])
	if thinking == nil {
		return body
	}
	switch jsonx.Str(thinking, "type") {
	case "enabled", "adaptive":
		// Pre-5.2 models have no reasoning_effort surface; z.ai honors
		// thinking.type there, so leave the object in place untouched.
		if isZai52(model) || isZai53(model) {
			delete(body, "thinking")
			body["reasoning_effort"] = zaiBudgetEffort(thinking)
		}
	case "disabled":
		if isZai53(model) {
			// GLM-5.3 cannot disable thinking; degrade to the shallowest effort.
			delete(body, "thinking")
			body["reasoning_effort"] = "low"
		}
		// ≤5.2 era: z.ai honors thinking.type=disabled; leave untouched.
	}
	return body
}

// zaiBudgetEffort buckets an Anthropic thinking budget onto z.ai's effort scale.
// An enabled thinking object without a budget maps to "high".
func zaiBudgetEffort(thinking map[string]any) string {
	budget, ok := jsonx.Int(thinking, "budget_tokens")
	if !ok {
		return "high"
	}
	switch {
	case budget <= 2048:
		return "low"
	case budget <= 8192:
		return "high"
	default:
		return "max"
	}
}

// zai53Effort collapses any effort onto GLM-5.3's low | high | max set.
func zai53Effort(e string) string {
	switch strings.ToLower(strings.TrimSpace(e)) {
	case "none", "minimal", "low":
		return "low"
	case "medium", "high":
		return "high"
	case "xhigh", "max":
		return "max"
	default:
		return "high"
	}
}

// zai52Effort keeps GLM-5.2's full effort set, clamping unknowns to "high".
func zai52Effort(e string) string {
	switch strings.ToLower(strings.TrimSpace(e)) {
	case "none", "minimal", "low", "medium", "high", "xhigh", "max":
		return strings.ToLower(strings.TrimSpace(e))
	default:
		return "high"
	}
}

func isZai53(model string) bool { return strings.Contains(model, "5.3") }
func isZai52(model string) bool { return strings.Contains(model, "5.2") }
