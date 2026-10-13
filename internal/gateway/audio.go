package gateway

import (
	"net/http"
	"strings"

	"github.com/noizu-labs/go-litellm/internal/errx"
)

// audioRoute matches the LiteLLM audio surface (/v1/audio/*). The gateway
// implements no audio providers; left to front-proxy passthrough these routes
// resolved to the LiteLLM target — this gateway itself — and looped until the
// pod died (7090 nested requests per call in the reproduction). They now get
// a clean, authenticated JSON error and are never forwarded.
func audioRoute(p string) bool {
	return p == "/v1/audio" || strings.HasPrefix(p, "/v1/audio/")
}

// audioUnsupported answers the audio surface without crashing or forwarding.
// 501 (not implemented) is honest — no configuration makes these routes work
// here — and the error shape matches the gateway's OpenAI envelope.
func (a *App) audioUnsupported(w http.ResponseWriter, r *http.Request) {
	writeErr(w, errx.New(http.StatusNotImplemented,
		"audio endpoints are not implemented by this gateway",
		errx.WithType("invalid_request_error")))
}
