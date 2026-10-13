package gateway

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/noizu-labs/go-litellm/internal/anthropic"
	"github.com/noizu-labs/go-litellm/internal/core"
	"github.com/noizu-labs/go-litellm/internal/errx"
	"github.com/noizu-labs/go-litellm/internal/frontproxy"
	"github.com/noizu-labs/go-litellm/internal/jsonx"
	"github.com/noizu-labs/go-litellm/internal/providers"
)

func (a *App) messages(w http.ResponseWriter, r *http.Request) {
	body := a.jsonBody(r)
	model := frontproxy.ExtractModel(body)
	a.tag(r, model, "", "")

	// Security: the LiteLLM surface of /v1/messages — any non-claude-* model —
	// routes through deployments/native inference (frontproxy rule
	// MatchMessagesNotModel → TargetLiteLLM/AuthMasterKey) and must require the
	// master key exactly like /v1/chat/completions. a.authenticate no-ops when
	// LITELLM_MASTER_KEY is unset, matching the existing conditional.
	//
	// claude-* stays AuthPassthrough — the client's own token is forwarded to
	// api.anthropic.com verbatim. That is the front-proxy contract run-claude
	// depends on: Claude Code arrives with its own OAuth credentials, never the
	// master key, so requiring it here would 401 every local session.
	if !strings.HasPrefix(model, "claude-") {
		if e := a.authenticate(r); e != nil {
			writeErr(w, e)
			return
		}
	}

	if strings.HasPrefix(model, "claude-") {
		a.tag(r, "", frontproxy.AnthropicAPI, "")
		a.forward(w, r, frontproxy.AnthropicAPI+r.URL.RequestURI(), a.forwardHeaders(r, frontproxy.AuthPassthrough), a.rawBody(r))
		return
	}

	if dep := a.Router.Lookup(model); dep != nil {
		a.messagesViaDeployment(w, r, body, dep)
		return
	}

	writeJSON(w, http.StatusNotFound, errx.New(404, "model not found: "+model, errx.WithType("not_found_error")).AnthropicBody())
}

func (a *App) messagesViaDeployment(w http.ResponseWriter, r *http.Request, body, dep map[string]any) {
	lp := jsonx.Nested(dep, "litellm_params")
	if lp == nil {
		lp = map[string]any{}
	}
	upstream := jsonx.Str(lp, "model")
	res, e := providers.Resolve(upstream, lp)
	if e != nil || res.Provider != "anthropic" {
		a.openaiFamilyTranslate(w, r, body)
		return
	}
	req := &providers.Request{
		Model:         res.Model,
		Provider:      "anthropic",
		LiteLLMParams: lp,
		CallType:      "chat",
	}
	headers, e := res.Adapter.ValidateEnvironment(req, map[string]string{})
	if e != nil {
		writeJSON(w, e.Status, e.AnthropicBody())
		return
	}
	url := res.Adapter.CompleteURL(req)
	out := jsonx.Clone(body)
	out["model"] = res.Model
	zai := providers.ZaiThinkingApplies(lp)
	wafer := providers.WaferThinkingApplies(lp)
	if zai {
		out = providers.ApplyZaiThinking(out, res.Model)
	}
	if wafer {
		// Clone lp: ApplyWaferThinking consumes default_reasoning_effort and
		// must not mutate the router's stored deployment map.
		out = providers.ApplyWaferThinking(out, res.Model, jsonx.Clone(lp))
	}
	pre := jsonx.Clone(out)
	stripAdditionalDropParams(out, lp)
	if (zai || wafer) && lp["reasoning_effort"] == nil {
		// Same contract as core.Prepare: z.ai/wafer keep the (already mapped)
		// effort even though additional_drop_params lists it — the entry is
		// there to strip raw client values, not the normalized one.
		if v, ok := pre["reasoning_effort"]; ok && v != nil {
			out["reasoning_effort"] = v
		}
	}
	raw, _ := json.Marshal(out)
	a.tag(r, "", url, "")
	a.forwardTo(w, r, url, headers, raw)
}

// stripAdditionalDropParams removes litellm_params.additional_drop_params keys
// from an upstream body (config form: ["context_management", "thinking", …]).
func stripAdditionalDropParams(body, lp map[string]any) {
	if body == nil || lp == nil {
		return
	}
	switch drop := lp["additional_drop_params"].(type) {
	case []any:
		for _, d := range drop {
			if s, ok := d.(string); ok {
				delete(body, s)
			}
		}
	case []string:
		for _, s := range drop {
			delete(body, s)
		}
	}
}

func (a *App) openaiFamilyTranslate(w http.ResponseWriter, r *http.Request, body map[string]any) {
	openaiBody := anthropic.RequestToOpenAI(body)
	if jsonx.Bool(body, "stream") {
		p, e := core.Prepare(a.Router, a.Config, openaiBody)
		if e != nil {
			writeJSON(w, e.Status, e.AnthropicBody())
			return
		}
		a.tag(r, "", "native:"+p.Req.Provider, "")
		if e := core.StreamAnthropic(w, a.HTTP, p, jsonx.Str(body, "model")); e != nil {
			writeJSON(w, e.Status, e.AnthropicBody())
		}
		return
	}
	resp, e := core.Completion(a.HTTP, a.Router, a.Config, openaiBody)
	if e != nil {
		writeJSON(w, e.Status, e.AnthropicBody())
		return
	}
	writeJSON(w, 200, anthropic.ResponseFromModelResponse(resp, jsonx.Str(body, "model")))
}
