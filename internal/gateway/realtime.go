package gateway

// Realtime voice bridge: WebUI/OpenAI-Realtime-shaped WebSocket clients ⇄
// DashScope session protocol (qwen omni realtime models on the Alibaba MaaS
// host). Browser never connects here — the WebUI backend dials
// /v1/realtime?model=… with a Bearer key and relays JSON frames.
//
// Protocol recap (live-verified against the MaaS host, 2026-10-18):
//   - client sends session.update / input_audio_buffer.append|commit|clear /
//     response.create / response.cancel / conversation.item.* ;
//   - upstream sends session.created UNSOLICITED right after the handshake
//     (we synthesize our own session.created for the client and swallow the
//     upstream one), then session.updated after each session.update;
//   - response deltas carry response_id/item_id/output_index/content_index
//     (we inject them as a safety net when absent);
//   - voice validation is lazy: an unsupported voice errors at response.create
//     and the upstream closes the socket (1007) right after the error event;
//   - an upstream {type:"error"} frame is relayed verbatim (WebUI aborts).

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"

	"github.com/noizu-labs/go-litellm/internal/jsonx"
)

const (
	realtimePathV1    = "/v1/realtime"
	realtimePathAlias = "/realtime"

	// Audio frames are bounded: WebUI caps input_audio_buffer.append at
	// 48_000 bytes of base64; 1 MiB covers every sane frame.
	realtimeReadLimitClient   = 1 << 20
	realtimeReadLimitUpstream = 16 << 20

	realtimeDialTimeout   = 10 * time.Second
	realtimeFinishTimeout = 5 * time.Second

	realtimeDefaultModelFallback = "alibaba/qwen3.8-omni-flash-realtime"
)

// realtimePath reports whether the request targets the realtime surface.
func realtimePath(p string) bool {
	return p == realtimePathV1 || p == realtimePathAlias
}

// isRealtimeUpgrade identifies WS upgrade requests so Handler() can skip the
// body pre-read + countingWriter wrapper (Hijacker must survive).
func isRealtimeUpgrade(r *http.Request) bool {
	if r.Method != http.MethodGet || !realtimePath(r.URL.Path) {
		return false
	}
	if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
		return false
	}
	for _, v := range strings.Split(r.Header.Get("Connection"), ",") {
		if strings.EqualFold(strings.TrimSpace(v), "upgrade") {
			return true
		}
	}
	return false
}

func realtimeDefaultModel() string {
	if m := strings.TrimSpace(os.Getenv("REALTIME_DEFAULT_MODEL")); m != "" {
		return m
	}
	return realtimeDefaultModelFallback
}

// realtimeAllowed gates which upstream models may be bridged.
func realtimeAllowed(upstreamID, modelName string) bool {
	if strings.HasSuffix(upstreamID, "-realtime") || strings.HasSuffix(modelName, "-realtime") {
		return true
	}
	for _, e := range strings.Split(os.Getenv("REALTIME_ALLOWLIST"), ",") {
		if e = strings.TrimSpace(e); e != "" && (e == upstreamID || e == modelName) {
			return true
		}
	}
	return false
}

// realtimeUpstreamModel derives the DashScope model id: the deployment's
// litellm_params.model minus its adapter prefix ("openai/…"), falling back to
// the requested model_name minus the "alibaba/" prefix and legacy "-maas"
// suffix when the entry carries no explicit model.
func realtimeUpstreamModel(litellmModel, modelName string) string {
	id := strings.TrimSpace(litellmModel)
	if id == "" {
		id = modelName
		if i := strings.LastIndex(id, "/"); i >= 0 {
			id = id[i+1:]
		}
		id = strings.TrimSuffix(id, "-maas")
		return id
	}
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	return id
}

// realtimeWSURL derives wss://{host}/api-ws/v1/realtime?model=<id> from the
// entry's resolved api_base (host only — the compatible-mode path never
// applies to the realtime surface).
func realtimeWSURL(apiBase, upstreamModel string) (string, error) {
	raw := strings.TrimSpace(apiBase)
	if raw == "" {
		return "", errors.New("deployment has no api_base")
	}
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return "", errors.New("invalid api_base")
	}
	scheme := "wss"
	if u.Scheme == "http" {
		scheme = "ws"
	}
	q := url.Values{}
	q.Set("model", upstreamModel)
	return (&url.URL{Scheme: scheme, Host: u.Host, Path: "/api-ws/v1/realtime", RawQuery: q.Encode()}).String(), nil
}

func randID(n int) string {
	buf := make([]byte, n)
	_, _ = rand.Read(buf)
	return hex.EncodeToString(buf)
}

// realtimeReject answers with a JSON error before any upgrade happens
// (unknown model / no deployment). Upgrade-capable clients see a plain 4xx.
func realtimeReject(w http.ResponseWriter, msg string) {
	writeJSON(w, http.StatusBadRequest, map[string]any{
		"error": map[string]any{"message": msg, "type": "invalid_request_error"},
	})
}

// realtime is the authed WS handler (GET /v1/realtime, GET /realtime).
func (a *App) realtime(w http.ResponseWriter, r *http.Request) {
	model := strings.TrimSpace(r.URL.Query().Get("model"))
	if model == "" {
		model = realtimeDefaultModel()
	}
	dep := a.Router.Lookup(model)
	if dep == nil {
		realtimeReject(w, "model not found: "+model)
		return
	}
	lp := jsonx.Nested(dep, "litellm_params")
	if lp == nil {
		realtimeReject(w, "deployment is malformed (no litellm_params): "+model)
		return
	}
	upstreamModel := realtimeUpstreamModel(jsonx.Str(lp, "model"), model)
	if !realtimeAllowed(upstreamModel, model) {
		realtimeReject(w, "model not realtime-capable: "+model)
		return
	}
	wsURL, err := realtimeWSURL(jsonx.Str(lp, "api_base"), upstreamModel)
	if err != nil {
		realtimeReject(w, "realtime upstream misconfigured: "+err.Error())
		return
	}
	apiKey := a.Router.Keys.Resolve(lp)

	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		// Server-to-server dial from WebUI; browsers never connect here, so
		// origin enforcement is meaningless — accept any.
		InsecureSkipVerify: true,
	})
	if err != nil {
		return
	}

	b := &realtimeBridge{
		app:           a,
		client:        conn,
		model:         model,
		upstreamModel: upstreamModel,
		wsURL:         wsURL,
		apiKey:        apiKey,
		voiceOverride: strings.TrimSpace(r.URL.Query().Get("voice")),
		started:       time.Now(),
	}

	// Contract: the server speaks first — session.created before anything else.
	hello := map[string]any{
		"type": "session.created",
		"session": map[string]any{
			"id":          "sess-b" + randID(12),
			"object":      "realtime.session",
			"model":       model,
			"voice":       "Tina",
			"modalities":  []any{"text", "audio"},
			"description": "synthesized by go-litellm realtime bridge",
		},
	}
	if b.writeClient(jsonx.EncodeJSON(hello)) != nil {
		b.logLine("client left before session.created")
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	up := &upstreamHolder{}
	clientDone := make(chan struct{})
	go func() {
		defer close(clientDone)
		a.realtimeClientPump(ctx, cancel, b, up)
	}()
	<-clientDone

	// Client is gone (or errored): end the upstream session politely.
	if u := up.get(); u != nil {
		finCtx, finCancel := context.WithTimeout(context.Background(), realtimeFinishTimeout)
		_ = u.Write(finCtx, websocket.MessageText, []byte(`{"type":"session.finish"}`))
		finCancel()
		_ = u.Close(websocket.StatusNormalClosure, "client disconnected")
	}
	_ = b.client.Close(websocket.StatusNormalClosure, "")
	b.logLine("")
}

// upstreamHolder carries the lazily dialed upstream conn across pumps.
type upstreamHolder struct {
	mu sync.Mutex
	u  *websocket.Conn
}

func (h *upstreamHolder) get() *websocket.Conn {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.u
}

func (h *upstreamHolder) set(u *websocket.Conn) {
	h.mu.Lock()
	h.u = u
	h.mu.Unlock()
}

type realtimeBridge struct {
	app           *App
	client        *websocket.Conn
	model         string
	upstreamModel string
	wsURL         string
	apiKey        string
	voiceOverride string
	started       time.Time

	// counters
	framesIn, framesOut atomic.Int64
	bytesIn, bytesOut   atomic.Int64
	// response state — owned by the upstream→client goroutine only
	respID, itemID string
	outputIndex    int
	contentIndex   int
}

func (b *realtimeBridge) writeClient(data []byte) error {
	b.framesOut.Add(1)
	b.bytesOut.Add(int64(len(data)))
	cctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return b.client.Write(cctx, websocket.MessageText, data)
}

func (b *realtimeBridge) logLine(closeInfo string) {
	log.Printf("[realtime] model=%s upstream=%s dur=%s cli_frames_in=%d cli_bytes_in=%d cli_frames_out=%d cli_bytes_out=%d %s",
		b.model, b.upstreamModel, time.Since(b.started).Round(time.Millisecond),
		b.framesIn.Load(), b.bytesIn.Load(), b.framesOut.Load(), b.bytesOut.Load(),
		closeInfo)
}

// realtimeClientPump relays client frames to the upstream, dialing lazily on
// the first forwardable frame.
func (a *App) realtimeClientPump(ctx context.Context, cancel context.CancelFunc, b *realtimeBridge, up *upstreamHolder) {
	b.client.SetReadLimit(realtimeReadLimitClient)
	for {
		typ, data, err := b.client.Read(ctx)
		if err != nil {
			return
		}
		b.framesIn.Add(1)
		b.bytesIn.Add(int64(len(data)))
		if typ != websocket.MessageText {
			continue
		}
		var ev map[string]any
		if json.Unmarshal(data, &ev) != nil {
			continue
		}
		t := jsonx.Str(ev, "type")
		if t == "" {
			continue
		}
		if t == "bridge.ping" {
			_ = b.writeClient([]byte(`{"type":"bridge.pong"}`))
			continue
		}
		u := up.get()
		if u == nil {
			u, err = a.realtimeDial(b)
			if err != nil {
				log.Printf("[realtime] upstream dial failed model=%s: %v", b.model, err)
				errFrame := map[string]any{
					"type": "error",
					"error": map[string]any{
						"code":    "upstream_unavailable",
						"message": "realtime upstream dial failed: " + err.Error(),
					},
				}
				_ = b.writeClient(jsonx.EncodeJSON(errFrame))
				_ = b.client.Close(websocket.StatusInternalError, "upstream dial failed")
				cancel()
				return
			}
			up.set(u)
			go a.realtimeUpstreamPump(ctx, cancel, b, u)
		}
		out := translateClientEvent(ev, b.voiceOverride)
		wctx := ctx
		if err := u.Write(wctx, websocket.MessageText, out); err != nil {
			cancel()
			return
		}
	}
}

// realtimeDial opens the DashScope realtime socket for this bridge session.
func (a *App) realtimeDial(b *realtimeBridge) (*websocket.Conn, error) {
	dctx, cancel := context.WithTimeout(context.Background(), realtimeDialTimeout)
	defer cancel()
	u, _, err := websocket.Dial(dctx, b.wsURL, &websocket.DialOptions{
		HTTPHeader: http.Header{"Authorization": []string{"Bearer " + b.apiKey}},
	})
	if err != nil {
		return nil, err
	}
	u.SetReadLimit(realtimeReadLimitUpstream)
	return u, nil
}

// realtimeUpstreamPump relays upstream frames to the client. It owns the
// bridge's response-id state.
func (a *App) realtimeUpstreamPump(ctx context.Context, cancel context.CancelFunc, b *realtimeBridge, u *websocket.Conn) {
	for {
		typ, data, err := u.Read(ctx)
		if err != nil {
			reason := ""
			if ce, ok := err.(websocket.CloseError); ok {
				reason = ce.Reason
			}
			code := websocket.CloseStatus(err)
			log.Printf("[realtime] upstream closed model=%s code=%d reason=%q", b.model, code, reason)
			// Relay the close to the client (1000 + reason; the error EVENT —
			// if any — was already forwarded verbatim).
			_ = b.client.Close(websocket.StatusNormalClosure, reason)
			cancel()
			return
		}
		if typ != websocket.MessageText {
			continue
		}
		out := b.translateUpstreamEvent(data)
		if out == nil {
			continue
		}
		if b.writeClient(out) != nil {
			cancel()
			return
		}
	}
}
