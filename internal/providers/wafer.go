package providers

import (
	"strings"

	"github.com/noizu-labs/go-litellm/internal/jsonx"
)

// wafer.ai reasoning contract (pass.wafer.ai endpoints front GLM/Kimi fleets):
//   - top-level reasoning_effort is the depth control surface
//   - GLM-5.3-class models accept only low | high | max and cannot disable thinking
//   - Kimi models accept reasoning_effort pass-through; Kimi has no thinking
//     budget contract of its own, so anthropic thinking is translated to a
//     best-effort effort (budget ladder when present, else high/low) and the
//     upstream endpoint makes the final call
//   - deployments may seed a default depth via litellm_params.default_reasoning_effort

// IsWaferBase reports whether an api_base points at a wafer.ai endpoint.
func IsWaferBase(apiBase string) bool {
	return strings.Contains(apiBase, "wafer.ai")
}

// WaferThinkingApplies reports whether wafer thinking mapping should run for a
// request routed through these litellm_params. Defaults ON for wafer.ai hosts;
// opt out per model with litellm_params.wafer_thinking_mapping: false.
func WaferThinkingApplies(lp map[string]any) bool {
	if lp == nil {
		return false
	}
	if b, ok := lp["wafer_thinking_mapping"].(bool); ok && !b {
		return false
	}
	return IsWaferBase(jsonx.Str(lp, "api_base"))
}

// ApplyWaferThinking normalizes a request body's thinking controls to wafer's
// reasoning_effort surface. body may be Anthropic-shaped (thinking object, as
// sent by Claude Code) or OpenAI-shaped (reasoning_effort). Precedence: explicit
// effort > thinking object > litellm_params.default_reasoning_effort. Consumes
// (deletes) default_reasoning_effort from lp in every path so it can never leak
// upstream. Mutates and returns body.
func ApplyWaferThinking(body map[string]any, model string, lp map[string]any) map[string]any {
	// Read the deployment default, then delete it immediately: core.go merges
	// only whitelisted deploymentTunables from litellm_params and adapters read
	// named lp fields only, so a leak is already impossible today — but the
	// deletion here makes the guarantee local to this feature.
	var effortDefault string
	if lp != nil {
		if s, ok := lp["default_reasoning_effort"].(string); ok {
			effortDefault = s
		}
		delete(lp, "default_reasoning_effort")
	}
	if body == nil {
		return body
	}
	if effort, hasEffort := body["reasoning_effort"].(string); hasEffort {
		// Explicit effort wins; the raw Anthropic thinking object is dropped.
		delete(body, "thinking")
		switch {
		case isKimi(model):
			// Kimi: pass reasoning_effort through untouched; upstream decides.
		case isZai53(model):
			// GLM-5.3-class endpoints only accept low | high | max.
			body["reasoning_effort"] = zai53Effort(effort)
		default:
			// GLM-5.2 and unknown models keep the full effort set untouched.
		}
		return body
	}
	if thinking := jsonx.AsMap(body["thinking"]); thinking != nil {
		switch jsonx.Str(thinking, "type") {
		case "enabled", "adaptive":
			// Kimi has no thinking-budget contract; translate anyway — the
			// budget ladder when a budget is present, else "high" — and let
			// upstream decide (every ladder value is a legal effort there).
			delete(body, "thinking")
			body["reasoning_effort"] = zaiBudgetEffort(thinking)
		case "disabled":
			// Wafer models cannot disable thinking; degrade to the shallowest effort.
			delete(body, "thinking")
			body["reasoning_effort"] = "low"
		}
		return body
	}
	// Body carries no thinking signal: seed the deployment default, if any.
	if effortDefault != "" {
		body["reasoning_effort"] = effortDefault
	}
	return body
}

// isKimi reports whether a model routes to a Kimi endpoint (case-insensitive).
func isKimi(model string) bool { return strings.Contains(strings.ToLower(model), "kimi") }
