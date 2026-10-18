package gateway

import (
	"net/http"
	"strings"

	"github.com/noizu-labs/go-litellm/internal/core"
	"github.com/noizu-labs/go-litellm/internal/errx"
	"github.com/noizu-labs/go-litellm/internal/jsonx"
	responsesPkg "github.com/noizu-labs/go-litellm/internal/responses"
)

// responsesIDRoute matches the id-addressed Responses routes
// (/v1/responses/{id} and /responses/{id}, one path segment only).
func responsesIDRoute(p string) bool {
	for _, base := range []string{"/v1/responses/", "/responses/"} {
		if !strings.HasPrefix(p, base) {
			continue
		}
		if rest := p[len(base):]; rest != "" && !strings.Contains(rest, "/") {
			return true
		}
	}
	return false
}

// responsesIDHandler answers the id-addressed Responses surface. The gateway
// is stateless — nothing was ever stored — so GET is a documented 404 and
// every mutating method (DELETE et al) a 405 (audio.go precedent: these are
// never forwarded).
func (a *App) responsesIDHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		writeErr(w, errx.New(http.StatusNotFound,
			"no stored response with this id: this gateway is stateless",
			errx.WithType("not_found_error"), errx.WithParam("response_id")))
		return
	}
	w.Header().Set("Allow", "GET")
	writeErr(w, errx.New(http.StatusMethodNotAllowed,
		"method not allowed: this gateway is stateless and stores no responses",
		errx.WithType("invalid_request_error")))
}

// responses implements POST /v1/responses — the stateless OpenAI Responses
// surface. Bodies translate to chat completions and upstream answers back
// (JSON, or Responses SSE when stream is set). Master-key auth is enforced
// by the dispatcher (a.authed), like /v1/chat/completions.
func (a *App) responses(w http.ResponseWriter, r *http.Request) {
	body := a.jsonBody(r)
	requested := jsonx.Str(body, "model")
	a.tag(r, requested, "", "")
	if e := responsesPkg.ValidateStateless(body); e != nil {
		a.tag(r, "", "", e.Message)
		writeErr(w, e)
		return
	}
	chat := responsesPkg.RequestToOpenAI(body)
	p, e := core.Prepare(a.Router, a.Config, chat)
	if e != nil {
		a.tag(r, "", "", e.Message)
		writeErr(w, e)
		return
	}
	a.tag(r, "", "native:"+p.Req.Provider, "")
	if jsonx.Bool(body, "stream") {
		if e := core.StreamResponses(w, a.HTTP, p, requested); e != nil {
			a.tag(r, "", "", e.Message)
			writeErr(w, e)
		}
		return
	}
	resp, e := core.DoCompletion(a.HTTP, p)
	if e != nil {
		a.tag(r, "", "", e.Message)
		writeErr(w, e)
		return
	}
	writeJSON(w, 200, responsesPkg.FromModelResponse(resp, requested))
}
