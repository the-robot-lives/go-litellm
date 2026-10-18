package gateway

// Translation between the WebUI/OpenAI-Realtime surface (client side) and the
// DashScope session protocol (upstream side).
//
// Client→upstream: normalize the WebUI session object — "audio/pcm"→"pcm",
// rate→sample_rate, inject sample_format/channels, hoist audio.output.voice to
// the session top level (DashScope echoes voice only there), map
// output_modalities→modalities, and never mix legacy (input_audio_format) and
// new-style (audio.input.format) fields.
//
// Upstream→client: rename response.* event types to the OpenAI-Realtime names
// WebUI consumes, and inject response_id/item_id/content_index when a delta
// arrives without them (defensive — the host normally includes them).

import (
	"encoding/json"
	"strings"

	"github.com/noizu-labs/go-litellm/internal/jsonx"
)

// upstreamToClientRenames is the exact event-type rename map (live-verified).
var upstreamToClientRenames = map[string]string{
	"response.audio.delta":             "response.output_audio.delta",
	"response.audio.done":              "response.output_audio.done",
	"response.audio_transcript.delta":  "response.output_audio_transcript.delta",
	"response.audio_transcript.done":   "response.output_audio_transcript.done",
	"response.text.delta":              "response.output_text.delta",
	"response.text.done":               "response.output_text.done",
	"response.audio_transcript":        "response.output_audio_transcript",
	"response.audio_transcript.paused": "response.output_audio_transcript.paused",
}

// translateClientEvent converts one client frame for the upstream. keep=false
// means drop the frame entirely.
func translateClientEvent(ev map[string]any, voiceOverride string) []byte {
	t := jsonx.Str(ev, "type")
	if t == "session.update" {
		out := map[string]any{"type": "session.update"}
		if eid := jsonx.Str(ev, "event_id"); eid != "" {
			out["event_id"] = eid
		}
		if s := jsonx.AsMap(ev["session"]); s != nil {
			out["session"] = normalizeWebUISession(s, voiceOverride)
		}
		return jsonx.EncodeJSON(out)
	}
	// input_audio_buffer.append/commit/clear, response.create/cancel,
	// conversation.item.create/truncate, session.finish … relay verbatim —
	// the DashScope surface accepts the OpenAI names as-is (live-verified).
	return jsonx.EncodeJSON(ev)
}

// normalizeWebUISession rewrites a WebUI session object into DashScope style.
func normalizeWebUISession(s map[string]any, voiceOverride string) map[string]any {
	out := jsonx.Clone(s)

	// modalities ← modalities | output_modalities | default. WebUI sends
	// output_modalities: ["audio"]; DashScope wants ["text","audio"]-style
	// "modalities" (text must be present — audio-only replies still need it
	// for transcripts).
	if _, ok := out["modalities"]; !ok {
		if om, ok := out["output_modalities"].([]any); ok && len(om) > 0 {
			mods := []any{}
			hasText := false
			for _, m := range om {
				if ms, _ := m.(string); ms == "text" {
					hasText = true
				}
				mods = append(mods, m)
			}
			if !hasText {
				mods = append([]any{"text"}, mods...)
			}
			out["modalities"] = mods
		} else {
			out["modalities"] = []any{"text", "audio"}
		}
	}
	delete(out, "output_modalities")

	// New-style audio block. When present, legacy format strings must go —
	// the host rejects mixing styles.
	if audio := jsonx.AsMap(out["audio"]); audio != nil {
		delete(out, "input_audio_format")
		delete(out, "output_audio_format")
		audioOut := map[string]any{}
		if in := jsonx.AsMap(audio["input"]); in != nil {
			inOut := jsonx.Clone(in)
			if f := jsonx.AsMap(in["format"]); f != nil {
				inOut["format"] = normalizeAudioFormat(f)
			}
			audioOut["input"] = inOut
		}
		if o := jsonx.AsMap(audio["output"]); o != nil {
			oOut := jsonx.Clone(o)
			// voice lives at the session top level (live-verified: the host
			// echoes session.voice, never audio.output.voice).
			if v := jsonx.Str(o, "voice"); v != "" {
				out["voice"] = v
			}
			delete(oOut, "voice")
			if f := jsonx.AsMap(o["format"]); f != nil {
				oOut["format"] = normalizeAudioFormat(f)
			}
			audioOut["output"] = oOut
		}
		out["audio"] = audioOut
	} else {
		delete(out, "audio")
	}

	// Query-string voice override wins over everything.
	if voiceOverride != "" {
		out["voice"] = voiceOverride
	}

	// turn_detection: kept exactly as sent — nil means manual mode, a map
	// means server/semantic VAD; both live-verified.
	// tools/tool_choice/instructions/input_audio_transcription: relayed as-is
	// (the host accepts and echoes them; live-verified 2026-10-18).
	return out
}

// normalizeAudioFormat converts WebUI format {type:"audio/pcm",rate:24000}
// into DashScope {type:"pcm",sample_rate:24000,sample_format:"s16le",channels:1}.
func normalizeAudioFormat(f map[string]any) map[string]any {
	out := jsonx.Clone(f)
	if t := jsonx.Str(out, "type"); t != "" {
		out["type"] = strings.TrimPrefix(t, "audio/")
	}
	if r, ok := jsonx.Float(f, "rate"); ok {
		delete(out, "rate")
		if _, present := out["sample_rate"]; !present {
			out["sample_rate"] = int(r)
		}
	}
	switch jsonx.Str(out, "type") {
	case "pcm":
		if _, ok := out["sample_format"]; !ok {
			out["sample_format"] = "s16le"
		}
		if _, ok := out["channels"]; !ok {
			out["channels"] = 1
		}
	}
	return out
}

// translateUpstreamEvent rewrites one upstream frame for the client.
// Returns nil when the frame must be swallowed (upstream's unsolicited
// session.created — the client already got our synthesized one).
func (b *realtimeBridge) translateUpstreamEvent(data []byte) []byte {
	var ev map[string]any
	if json.Unmarshal(data, &ev) != nil {
		return data // not JSON — relay untouched
	}
	t := jsonx.Str(ev, "type")
	if t == "session.created" {
		return nil
	}

	// Track ids for defensive injection.
	switch t {
	case "response.created", "response.done":
		if rd := jsonx.AsMap(ev["response"]); rd != nil {
			if id := jsonx.Str(rd, "id"); id != "" {
				b.respID = id
			}
		}
	case "response.output_item.added":
		if id := jsonx.Str(ev, "response_id"); id != "" {
			b.respID = id
		}
		if it := jsonx.AsMap(ev["item"]); it != nil {
			if id := jsonx.Str(it, "id"); id != "" {
				b.itemID = id
			}
		}
		if oi, ok := jsonx.Int(ev, "output_index"); ok {
			b.outputIndex = oi
		}
	case "response.content_part.added":
		if id := jsonx.Str(ev, "response_id"); id != "" {
			b.respID = id
		}
		if id := jsonx.Str(ev, "item_id"); id != "" {
			b.itemID = id
		}
		if ci, ok := jsonx.Int(ev, "content_index"); ok {
			b.contentIndex = ci
		}
	}

	renamed, ok := upstreamToClientRenames[t]
	if !ok {
		return data
	}
	ev["type"] = renamed
	if jsonx.Str(ev, "response_id") == "" && b.respID != "" {
		ev["response_id"] = b.respID
	}
	if jsonx.Str(ev, "item_id") == "" && b.itemID != "" {
		ev["item_id"] = b.itemID
	}
	if _, present := jsonx.Int(ev, "content_index"); !present {
		ev["content_index"] = b.contentIndex
	}
	return jsonx.EncodeJSON(ev)
}
