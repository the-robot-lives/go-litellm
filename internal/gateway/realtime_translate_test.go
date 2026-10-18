package gateway

// Table-driven tests for the realtime bridge translation layer. Fixture shapes
// come from live protocol recon against the MaaS host (2026-10-18,
// staging/ws-recon/events-*.json).

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/jsonx"
)

func mustJSON(t *testing.T, s string) map[string]any {
	t.Helper()
	m := map[string]any{}
	if err := json.Unmarshal([]byte(s), &m); err != nil {
		t.Fatalf("fixture: %v", err)
	}
	return m
}

// TestNormalizeWebUISession covers the WebUI session.update → DashScope shape.
func TestNormalizeWebUISession(t *testing.T) {
	// WebUI 0.12.3 shape (verified wire format).
	webui := mustJSON(t, `{
		"output_modalities": ["audio"],
		"instructions": "be brief",
		"audio": {
			"input": {
				"format": {"type": "audio/pcm", "rate": 24000},
				"transcription": {"model": "gummy-realtime-v1"},
				"turn_detection": {"type": "server_vad", "threshold": 0.2, "prefix_padding_ms": 300, "silence_duration_ms": 800}
			},
			"output": {
				"format": {"type": "audio/pcm", "rate": 24000},
				"voice": "Chelsie"
			}
		},
		"tools": [{"type": "function", "name": "get_time", "description": "time", "parameters": {"type": "object", "properties": {}, "required": []}}],
		"tool_choice": "auto"
	}`)

	got := normalizeWebUISession(webui, "")

	// modalities derived from output_modalities, text prepended.
	mods, _ := got["modalities"].([]any)
	if len(mods) != 2 || mods[0] != "text" || mods[1] != "audio" {
		t.Fatalf("modalities = %v", mods)
	}
	if _, ok := got["output_modalities"]; ok {
		t.Fatal("output_modalities must be stripped")
	}
	// legacy format strings must not coexist with the new-style audio block
	if _, ok := got["input_audio_format"]; ok {
		t.Fatal("input_audio_format must be dropped when new-style audio present")
	}
	audio := jsonx.AsMap(got["audio"])
	if audio == nil {
		t.Fatal("audio missing")
	}
	inFmt := jsonx.AsMap(jsonx.AsMap(audio["input"])["format"])
	if inFmt["type"] != "pcm" {
		t.Fatalf("input format = %v", inFmt)
	}
	if r, _ := jsonx.Float(inFmt, "sample_rate"); r != 24000 {
		t.Fatalf("input sample_rate = %v", inFmt)
	}
	if inFmt["sample_format"] != "s16le" {
		t.Fatalf("input sample_format = %v", inFmt)
	}
	if r, _ := jsonx.Float(inFmt, "channels"); r != 1 {
		t.Fatalf("input channels = %v", inFmt)
	}
	outFmt := jsonx.AsMap(jsonx.AsMap(audio["output"])["format"])
	if outFmt["type"] != "pcm" {
		t.Fatalf("output format = %v", outFmt)
	}
	if r, _ := jsonx.Float(outFmt, "sample_rate"); r != 24000 {
		t.Fatalf("output sample_rate = %v", outFmt)
	}
	// voice hoisted to top level, removed from audio.output
	if got["voice"] != "Chelsie" {
		t.Fatalf("voice = %v", got["voice"])
	}
	if jsonx.Str(jsonx.AsMap(audio["output"]), "voice") != "" {
		t.Fatal("audio.output.voice must be removed")
	}
	// transcription / turn_detection / tools relayed untouched
	if jsonx.Str(jsonx.AsMap(jsonx.AsMap(audio["input"])["transcription"]), "model") != "gummy-realtime-v1" {
		t.Fatal("input transcription lost")
	}
	inMap := jsonx.AsMap(audio["input"])
	td2 := jsonx.AsMap(inMap["turn_detection"])
	if td2 == nil || td2["type"] != "server_vad" || td2["threshold"] != 0.2 {
		t.Fatalf("turn_detection = %v", td2)
	}
	if _, ok := got["tools"]; !ok {
		t.Fatal("tools must be relayed (upstream accepts them)")
	}
}

func TestNormalizeWebUISessionVoiceOverride(t *testing.T) {
	s := mustJSON(t, `{"voice": "Chelsie", "output_modalities": ["audio"]}`)
	got := normalizeWebUISession(s, "Tina")
	if got["voice"] != "Tina" {
		t.Fatalf("voice override not applied: %v", got["voice"])
	}
}

func TestNormalizeWebUISessionLegacyFormatsKept(t *testing.T) {
	// No audio block: legacy format strings stay (live-verified accepted).
	s := mustJSON(t, `{"input_audio_format": "pcm", "output_audio_format": "pcm", "voice": "Tina", "turn_detection": null}`)
	got := normalizeWebUISession(s, "")
	if got["input_audio_format"] != "pcm" || got["output_audio_format"] != "pcm" {
		t.Fatalf("legacy formats dropped: %v", got)
	}
	if v, ok := got["turn_detection"]; !ok || v != nil {
		t.Fatalf("turn_detection null must survive: %v", got["turn_detection"])
	}
	// default modalities when nothing specified
	mods, _ := got["modalities"].([]any)
	if len(mods) != 2 {
		t.Fatalf("default modalities = %v", mods)
	}
}

func TestNormalizeWebUISessionAudioOnlyModalitiesDedupe(t *testing.T) {
	s := mustJSON(t, `{"output_modalities": ["text"]}`)
	got := normalizeWebUISession(s, "")
	mods, _ := got["modalities"].([]any)
	if len(mods) != 1 || mods[0] != "text" {
		t.Fatalf("modalities = %v", mods)
	}
}

// TestUpstreamRenames covers the rename map with the live-verified upstream
// frame shapes (ids present — must not be overwritten).
func TestUpstreamRenames(t *testing.T) {
	b := &realtimeBridge{}
	// audio delta (ids present upstream)
	frame := mustJSON(t, `{"event_id":"e1","type":"response.audio.delta","response_id":"resp_1","item_id":"item_1","output_index":0,"content_index":0,"delta":"QUJD"}`)
	out := b.translateUpstreamEvent([]byte(mustMarshal(frame)))
	got := mustJSON(t, string(out))
	if got["type"] != "response.output_audio.delta" {
		t.Fatalf("type = %v", got["type"])
	}
	if got["response_id"] != "resp_1" || got["item_id"] != "item_1" {
		t.Fatalf("ids clobbered: %v", got)
	}
	if ci, _ := jsonx.Float(got, "content_index"); ci != 0 {
		t.Fatalf("content_index clobbered: %v", got)
	}
	// transcript delta without ids → injected from tracked state
	b2 := &realtimeBridge{respID: "resp_2", itemID: "item_2", contentIndex: 1}
	frame2 := mustJSON(t, `{"event_id":"e2","type":"response.audio_transcript.delta","delta":"hi"}`)
	got2 := mustJSON(t, string(b2.translateUpstreamEvent([]byte(mustMarshal(frame2)))))
	if got2["type"] != "response.output_audio_transcript.delta" {
		t.Fatalf("type = %v", got2["type"])
	}
	if got2["response_id"] != "resp_2" || got2["item_id"] != "item_2" {
		t.Fatalf("id injection failed: %v", got2)
	}
	if ci, _ := jsonx.Float(got2, "content_index"); ci != 1 {
		t.Fatalf("content_index injection failed: %v", got2)
	}
	// text + done variants
	for _, c := range [][2]string{
		{"response.text.delta", "response.output_text.delta"},
		{"response.text.done", "response.output_text.done"},
		{"response.audio.done", "response.output_audio.done"},
		{"response.audio_transcript.done", "response.output_audio_transcript.done"},
	} {
		b3 := &realtimeBridge{}
		f := map[string]any{"type": c[0], "delta": "x"}
		got3 := mustJSON(t, string(b3.translateUpstreamEvent([]byte(mustMarshal(f)))))
		if got3["type"] != c[1] {
			t.Fatalf("%s → %v, want %s", c[0], got3["type"], c[1])
		}
	}
}

func TestUpstreamSessionCreatedSwallowed(t *testing.T) {
	b := &realtimeBridge{}
	f := mustJSON(t, `{"type":"session.created","session":{"id":"sess_x","voice":"Tina"}}`)
	if out := b.translateUpstreamEvent([]byte(mustMarshal(f))); out != nil {
		t.Fatalf("upstream session.created must be swallowed, got %s", out)
	}
}

func TestUpstreamUnknownEventPassesThrough(t *testing.T) {
	b := &realtimeBridge{}
	raw := `{"type":"input_audio_buffer.speech_started","audio_start_ms":100}`
	if out := b.translateUpstreamEvent([]byte(raw)); string(out) != raw {
		t.Fatalf("passthrough broken: %s", out)
	}
	// transcription completed carries ids; relayed unrenamed
	b2 := &realtimeBridge{}
	f := mustJSON(t, `{"type":"conversation.item.input_audio_transcription.completed","item_id":"item_1","content_index":0,"transcript":"tone"}`)
	got := mustJSON(t, string(b2.translateUpstreamEvent([]byte(mustMarshal(f)))))
	if got["type"] != "conversation.item.input_audio_transcription.completed" || got["transcript"] != "tone" {
		t.Fatalf("transcription event mangled: %v", got)
	}
}

func TestTranslateClientEventPassthrough(t *testing.T) {
	raw := `{"type":"input_audio_buffer.append","audio":"QUJD"}`
	ev := mustJSON(t, raw)
	out := string(translateClientEvent(ev, ""))
	var got map[string]any
	if json.Unmarshal([]byte(out), &got) != nil {
		t.Fatalf("unmarshal: %s", out)
	}
	if got["type"] != "input_audio_buffer.append" || got["audio"] != "QUJD" {
		t.Fatalf("append mangled: %s", out)
	}
}

func TestTranslateClientEventSessionUpdate(t *testing.T) {
	ev := mustJSON(t, `{"type":"session.update","event_id":"ev9","session":{"voice":"Tina","output_modalities":["audio"]}}`)
	out := string(translateClientEvent(ev, ""))
	got := mustJSON(t, out)
	if got["type"] != "session.update" || got["event_id"] != "ev9" {
		t.Fatalf("envelope broken: %s", out)
	}
	if jsonx.Str(jsonx.AsMap(got["session"]), "voice") != "Tina" {
		t.Fatalf("session lost: %s", out)
	}
}

func TestRealtimeUpstreamModel(t *testing.T) {
	cases := [][3]string{
		{"openai/qwen3.8-omni-flash-realtime", "alibaba/qwen3.8-omni-flash-realtime", "qwen3.8-omni-flash-realtime"},
		{"", "alibaba/qwen3.8-flash-maas", "qwen3.8-flash"},
		{"openai/qwen-audio-3.1-realtime-plus", "alibaba/x", "qwen-audio-3.1-realtime-plus"},
	}
	for _, c := range cases {
		if got := realtimeUpstreamModel(c[0], c[1]); got != c[2] {
			t.Fatalf("realtimeUpstreamModel(%q,%q) = %q, want %q", c[0], c[1], got, c[2])
		}
	}
}

func TestRealtimeWSURL(t *testing.T) {
	got, err := realtimeWSURL("https://llm-50geojgn528m31d1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1", "qwen3.8-omni-flash-realtime")
	if err != nil {
		t.Fatal(err)
	}
	want := "wss://llm-50geojgn528m31d1.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=qwen3.8-omni-flash-realtime"
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
	if _, err := realtimeWSURL("", "m"); err == nil {
		t.Fatal("empty api_base must error")
	}
	got2, _ := realtimeWSURL("http://localhost:4000/v1", "m")
	if !strings.HasPrefix(got2, "ws://") {
		t.Fatalf("http→ws failed: %s", got2)
	}
}

func TestRealtimeAllowed(t *testing.T) {
	if !realtimeAllowed("qwen3.8-omni-flash-realtime", "alibaba/qwen3.8-omni-flash-realtime") {
		t.Fatal("-realtime suffix must pass")
	}
	if realtimeAllowed("qwen3.8-flash", "alibaba/qwen3.8-flash") {
		t.Fatal("non-realtime must fail")
	}
	t.Setenv("REALTIME_ALLOWLIST", " glm-5.3-prime , other ")
	if !realtimeAllowed("glm-5.3-prime", "alibaba/glm-5.3-prime") {
		t.Fatal("allowlist entry must pass")
	}
	if realtimeAllowed("nope", "alibaba/nope") {
		t.Fatal("non-listed must fail")
	}
}

func TestIsRealtimeUpgrade(t *testing.T) {
	mk := func(headers map[string]string) *http.Request {
		r, _ := http.NewRequest("GET", "/v1/realtime?model=x", nil)
		for k, v := range headers {
			r.Header.Set(k, v)
		}
		return r
	}
	if !isRealtimeUpgrade(mk(map[string]string{"Upgrade": "websocket", "Connection": "Upgrade"})) {
		t.Fatal("plain upgrade must match")
	}
	if !isRealtimeUpgrade(mk(map[string]string{"Upgrade": "WebSocket", "Connection": "keep-alive, Upgrade"})) {
		t.Fatal("case-insensitive upgrade must match")
	}
	if isRealtimeUpgrade(mk(nil)) {
		t.Fatal("non-upgrade GET must not match")
	}
	r2, _ := http.NewRequest("POST", "/v1/realtime", nil)
	r2.Header.Set("Upgrade", "websocket")
	if isRealtimeUpgrade(r2) {
		t.Fatal("POST must not match")
	}
}

func mustMarshal(m map[string]any) string {
	b, _ := json.Marshal(m)
	return string(b)
}
