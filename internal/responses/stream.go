package responses

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/noizu-labs/go-litellm/internal/jsonx"
	"github.com/noizu-labs/go-litellm/internal/providers"
)

// Stream assembles OpenAI Responses SSE from normalized OpenAI-family chunks,
// mirroring anthropic.Stream's lazy-open design:
//
//   - the message item and its output_text part open on the first text
//     fragment, so a tool-only stream never carries a fake empty message
//   - delta.tool_calls (the OpenAI list shape: {index, id, type:"function",
//     function:{name, arguments}} on the first fragment for an index, then
//     {index, function:{arguments}} on argument deltas) open function_call
//     items keyed on tool_calls[].index; the upstream id is the Responses
//     call_id, the item id is freshly minted (fc_…), matching
//     FromModelResponse
//   - every frame carries a monotonically increasing sequence_number
//
// Reasoning deltas are ignored in v1 — the stateless surface emits no
// reasoning items, matching FromModelResponse (documented limitation).
type Stream struct {
	model     string
	id        string
	createdAt int64
	seq       int
	items     []*streamItem
	msg       *streamItem // lazily opened message item (nil until first text)
	toolIdx   map[int]*streamItem
	finish    string
	usage     map[string]any
}

type streamItem struct {
	kind   string // "message" | "function_call"
	id     string // msg_… / fc_…
	callID string
	name   string
	idx    int // Responses output_index
	text   strings.Builder
	args   strings.Builder
}

// NewStream starts an idle Responses SSE assembler for the requested model id.
func NewStream(model string) *Stream {
	return &Stream{
		model:     model,
		id:        "resp_" + providers.RandID(16),
		createdAt: time.Now().Unix(),
		toolIdx:   map[int]*streamItem{},
	}
}

// Preamble emits response.created only — output items open on the first delta.
func (s *Stream) Preamble() []string {
	return []string{s.frame("response.created", map[string]any{
		"type":     "response.created",
		"response": s.shell("in_progress", nil, nil),
	})}
}

// Push translates one normalized upstream chunk into zero or more SSE frames.
func (s *Stream) Push(chunk providers.StreamChunk) []string {
	if chunk.FinishReason != nil {
		if fr := stringify(chunk.FinishReason); fr != "" {
			s.finish = fr
		}
	}
	if chunk.Usage != nil {
		if u := jsonx.AsMap(chunk.Usage); u != nil {
			s.usage = u
		}
	}
	var frames []string
	frames = append(frames, s.emitText(chunk.Text)...)
	frames = append(frames, s.emitTools(chunk.ToolUse)...)
	return frames
}

// Close finishes every opened item in output_index order, then emits
// response.completed carrying the fully assembled response object with real
// usage. A stream with no output at all still yields one (empty) message
// item, mirroring anthropic.Stream's empty-block fallback.
func (s *Stream) Close() []string {
	var frames []string
	if len(s.items) == 0 {
		frames = append(frames, s.openMessage()...)
	}
	for _, it := range s.items {
		switch it.kind {
		case "message":
			text := it.text.String()
			frames = append(frames,
				s.frame("response.output_text.done", map[string]any{
					"type": "response.output_text.done", "item_id": it.id,
					"output_index": it.idx, "content_index": 0, "text": text,
				}),
				s.frame("response.content_part.done", map[string]any{
					"type": "response.content_part.done", "item_id": it.id,
					"output_index": it.idx, "content_index": 0, "part": textPart(text),
				}),
				s.frame("response.output_item.done", map[string]any{
					"type":         "response.output_item.done",
					"output_index": it.idx,
					"item":         messageItem(it.id, "completed", []any{textPart(text)}),
				}),
			)
		case "function_call":
			args := it.args.String()
			frames = append(frames,
				s.frame("response.function_call_arguments.done", map[string]any{
					"type": "response.function_call_arguments.done", "item_id": it.id,
					"output_index": it.idx, "arguments": args,
				}),
				s.frame("response.output_item.done", map[string]any{
					"type":         "response.output_item.done",
					"output_index": it.idx,
					"item":         toolItem(it.id, it.callID, it.name, args, "completed"),
				}),
			)
		}
	}
	return append(frames, s.frame("response.completed", map[string]any{
		"type":     "response.completed",
		"response": s.terminal(),
	}))
}

func (s *Stream) emitText(text string) []string {
	if text == "" {
		return nil
	}
	frames := s.openMessage()
	s.msg.text.WriteString(text)
	return append(frames, s.frame("response.output_text.delta", map[string]any{
		"type": "response.output_text.delta", "item_id": s.msg.id,
		"output_index": s.msg.idx, "content_index": 0, "delta": text,
	}))
}

// openMessage lazily opens the message item + its output_text part.
func (s *Stream) openMessage() []string {
	if s.msg != nil {
		return nil
	}
	s.msg = &streamItem{kind: "message", id: "msg_" + providers.RandID(12), idx: len(s.items)}
	s.items = append(s.items, s.msg)
	return []string{
		s.frame("response.output_item.added", map[string]any{
			"type": "response.output_item.added", "output_index": s.msg.idx,
			"item": messageItem(s.msg.id, "in_progress", []any{}),
		}),
		s.frame("response.content_part.added", map[string]any{
			"type": "response.content_part.added", "item_id": s.msg.id,
			"output_index": s.msg.idx, "content_index": 0, "part": textPart(""),
		}),
	}
}

// emitTools consumes the OpenAI delta.tool_calls list shape. A tool item
// opens on its first NAMED fragment (the identity carrier — the item payload
// carries the name, per OpenAI convention); argument-only fragments for an
// unseen index are dropped exactly like anthropic.Stream, since a
// function_call without a name is uncallable.
func (s *Stream) emitTools(raw any) []string {
	list := asList(raw)
	if len(list) == 0 {
		return nil
	}
	var frames []string
	for _, item := range list {
		frames = append(frames, s.emitTool(item)...)
	}
	return frames
}

func (s *Stream) emitTool(item any) []string {
	m := jsonx.AsMap(item)
	if m == nil {
		return nil
	}
	fun := jsonx.AsMap(m["function"])
	id := jsonx.Str(m, "id")
	name := jsonx.Str(fun, "name")
	args := argumentsString(fun["arguments"])
	oi, ok := jsonx.Int(m, "index")
	if !ok {
		oi = 0
	}

	ti, exists := s.toolIdx[oi]
	if !exists {
		if name == "" {
			return nil
		}
		callID := id
		if callID == "" {
			callID = "call_" + providers.RandID(12)
		}
		ti = &streamItem{
			kind: "function_call", id: "fc_" + providers.RandID(12),
			callID: callID, name: name, idx: len(s.items),
		}
		s.items = append(s.items, ti)
		s.toolIdx[oi] = ti
		return append([]string{s.frame("response.output_item.added", map[string]any{
			"type": "response.output_item.added", "output_index": ti.idx,
			"item": toolItem(ti.id, ti.callID, ti.name, "", "in_progress"),
		})}, s.argDeltas(ti, args)...)
	}
	return s.argDeltas(ti, args)
}

func (s *Stream) argDeltas(ti *streamItem, args string) []string {
	if args == "" {
		return nil
	}
	ti.args.WriteString(args)
	return []string{s.frame("response.function_call_arguments.delta", map[string]any{
		"type": "response.function_call_arguments.delta", "item_id": ti.id,
		"output_index": ti.idx, "delta": args,
	})}
}

// frame stamps sequence_number and renders one SSE frame.
func (s *Stream) frame(event string, data map[string]any) string {
	data["sequence_number"] = s.seq
	s.seq++
	return sse(event, data)
}

// shell is the response object as it appears mid-stream.
func (s *Stream) shell(status string, incomplete, usage any) map[string]any {
	return map[string]any{
		"id": s.id, "object": "response", "created_at": s.createdAt,
		"status": status, "model": s.model, "output": s.output(),
		"error": nil, "incomplete_details": incomplete,
		"usage": usage, "metadata": map[string]any{},
	}
}

// terminal is the response object carried by response.completed: fully
// assembled output, real usage, and the finish-derived status.
func (s *Stream) terminal() map[string]any {
	status, incomplete := "completed", any(nil)
	switch s.finish {
	case "length":
		status, incomplete = "incomplete", map[string]any{"reason": "max_output_tokens"}
	case "content_filter":
		status, incomplete = "incomplete", map[string]any{"reason": "content_filter"}
	}
	return s.shell(status, incomplete, usageOf(s.usage))
}

func (s *Stream) output() []any {
	out := make([]any, 0, len(s.items))
	for _, it := range s.items {
		switch it.kind {
		case "message":
			out = append(out, messageItem(it.id, "completed", []any{textPart(it.text.String())}))
		case "function_call":
			out = append(out, toolItem(it.id, it.callID, it.name, it.args.String(), "completed"))
		}
	}
	return out
}

// sse formats one Responses SSE frame — same wire shape as anthropic.SSE.
func sse(event string, data any) string {
	b, _ := json.Marshal(data)
	return "event: " + event + "\ndata: " + string(b) + "\n\n"
}
