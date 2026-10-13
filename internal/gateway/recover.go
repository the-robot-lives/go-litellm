package gateway

import (
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/noizu-labs/go-litellm/internal/errx"
)

// withRecovery guarantees no handler panic can take a request down uncleanly:
// any panic is converted into a structured 500 JSON error (gateway error
// envelope) and the server keeps serving. Handlers must stay panic-free, but
// an authenticated client must never be able to abort the pod.
//
// Logs carry method, path, and the panic value only — never request bodies,
// headers, or API keys.
func (a *App) withRecovery(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				a.logRecoveredPanic(r, rec)
				switch ww := w.(type) {
				case *countingWriter:
					if ww.wroteHeader {
						return // mid-stream: response already committed
					}
					ww.status = http.StatusInternalServerError
					writeErr(ww, errx.New(http.StatusInternalServerError, "internal server error"))
				default:
					writeErr(w, errx.New(http.StatusInternalServerError, "internal server error"))
				}
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// logRecoveredPanic keeps the trace bounded and body/key-free: the panic value
// only (runtime nil-deref messages carry no request data), no stack dump.
func (a *App) logRecoveredPanic(r *http.Request, rec any) {
	msg := "panic value"
	switch v := rec.(type) {
	case string:
		msg = v
	case error:
		msg = v.Error()
	}
	if len(msg) > 512 {
		msg = msg[:512] + "..."
	}
	log.Printf("[gateway] panic recovered: method=%s path=%s panic=%v", r.Method, r.URL.Path, msg)
}

// isSelfTarget reports whether raw resolves to this gateway's own listener —
// the shape behind the audio-route crash: front-proxy passthrough matched
// /v1/audio/* to the LiteLLM target, whose base URL is this gateway itself,
// so each forward re-entered the handler (7090 nested requests per call
// before port exhaustion; pod death in production).
func (a *App) isSelfTarget(raw string) bool {
	if a.Settings.Port == 0 {
		return false // unset port can never be self-referential
	}
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	port := u.Port()
	if port == "" {
		if u.Scheme == "https" {
			port = "443"
		} else {
			port = "80"
		}
	}
	if port != strconv.Itoa(a.Settings.Port) {
		return false
	}
	switch strings.ToLower(u.Hostname()) {
	case "127.0.0.1", "localhost", "::1", "0.0.0.0", "":
		return true
	}
	return strings.EqualFold(u.Hostname(), a.Settings.Host)
}
