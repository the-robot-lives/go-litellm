package anthropic

import (
	"encoding/json"
	"strings"

	"github.com/noizu-labs/go-litellm/internal/jsonx"
	"github.com/noizu-labs/go-litellm/internal/providers"
)

// Stream assembles Anthropic SSE from normalized OpenAI-family chunks.
//
// Unlike a naive "always open a text block" translator, this starts content
// blocks lazily so:
//
//   - Groq gpt-oss delta.reasoning becomes thinking blocks
//   - delta.tool_calls become tool_use + input_json_delta
//   - a stream that only reasons or only calls tools does not emit a fake
//     empty text block with stop_reason=end_turn (Claude Code then reports
//     "The model's tool call could not be parsed")
type Stream struct {
	model   string
	kind    blockKind
	index   int
	next    int
	finish  string
	usage   map[string]any
	sawTool bool
	toolIdx map[int]int // OpenAI tool_calls[].index → Anthropic content-block index
}

type blockKind int

const (
	blockNone blockKind = iota
	blockText
	blockThinking
	blockTool
)

// NewStream starts an idle Anthropic SSE assembler for the requested model id.
func NewStream(model string) *Stream {
	return &Stream{model: model, toolIdx: map[int]int{}}
}

// Preamble emits message_start only — content blocks open on the first delta.
func (s *Stream) Preamble() []string {
	message := map[string]any{
		"id": "msg_" + providers.RandID(12), "type": "message", "role": "assistant",
		"model": s.model, "content": []any{}, "stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
	}
	return []string{SSE("message_start", map[string]any{"type": "message_start", "message": message})}
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
	frames = append(frames, s.emitReasoning(chunk.Reasoning)...)
	frames = append(frames, s.emitText(chunk.Text)...)
	frames = append(frames, s.emitTools(chunk.ToolUse)...)
	return frames
}

// Close stops the open block (if any) and emits message_delta + message_stop.
func (s *Stream) Close() []string {
	frames := s.closeBlock()
	if s.next == 0 {
		// Anthropic clients reject a message with no content blocks.
		frames = append(frames,
			StreamContentBlockStart(0, map[string]any{"type": "text", "text": ""}),
			StreamContentBlockStop(0),
		)
		s.next = 1
	}
	finish := s.finish
	if s.sawTool && (finish == "" || finish == "stop") {
		finish = "tool_calls"
	}
	return append(frames, StreamClosingAfterBlocks(finish, s.usage)...)
}

func (s *Stream) emitReasoning(text string) []string {
	if text == "" {
		return nil
	}
	var frames []string
	frames = append(frames, s.ensure(blockThinking)...)
	return append(frames, StreamThinkingDelta(s.index, text))
}

func (s *Stream) emitText(text string) []string {
	if text == "" {
		return nil
	}
	var frames []string
	frames = append(frames, s.ensure(blockText)...)
	return append(frames, StreamTextDeltaAt(s.index, text))
}

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
	args := argsString(fun["arguments"])
	oi, ok := jsonx.Int(m, "index")
	if !ok {
		oi = 0
	}

	var frames []string
	if name != "" {
		if _, exists := s.toolIdx[oi]; !exists {
			if id == "" {
				id = "toolu_" + providers.RandID(12)
			}
			frames = append(frames, s.closeBlock()...)
			i := s.next
			frames = append(frames, SSE("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": i,
				"content_block": map[string]any{
					"type": "tool_use", "id": id, "name": name, "input": map[string]any{},
				},
			}))
			s.kind = blockTool
			s.index = i
			s.next = i + 1
			s.toolIdx[oi] = i
			s.sawTool = true
		}
	}
	if args == "" {
		return frames
	}
	ai, exists := s.toolIdx[oi]
	if !exists || s.kind != blockTool || s.index != ai {
		return frames
	}
	return append(frames, StreamInputJSONDelta(s.index, args))
}

func (s *Stream) ensure(kind blockKind) []string {
	if s.kind == kind {
		return nil
	}
	frames := s.closeBlock()
	i := s.next
	switch kind {
	case blockText:
		frames = append(frames, StreamContentBlockStart(i, map[string]any{"type": "text", "text": ""}))
	case blockThinking:
		frames = append(frames, StreamContentBlockStart(i, map[string]any{"type": "thinking", "thinking": ""}))
	default:
		return frames
	}
	s.kind = kind
	s.index = i
	s.next = i + 1
	return frames
}

func (s *Stream) closeBlock() []string {
	if s.kind == blockNone {
		return nil
	}
	frame := StreamContentBlockStop(s.index)
	s.kind = blockNone
	return []string{frame}
}

func asList(v any) []any {
	switch t := v.(type) {
	case nil:
		return nil
	case []any:
		return t
	default:
		if m := jsonx.AsMap(t); m != nil {
			return []any{m}
		}
		return nil
	}
}

func argsString(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	default:
		b, err := json.Marshal(t)
		if err != nil {
			return ""
		}
		return string(b)
	}
}

func stringify(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	b, _ := json.Marshal(v)
	return strings.Trim(string(b), `"`)
}
