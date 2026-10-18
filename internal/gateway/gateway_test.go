package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/noizu-labs/go-litellm/internal/config"
	"github.com/noizu-labs/go-litellm/internal/runtime"
)

func testApp(t *testing.T, master string) *App {
	t.Helper()
	cfg := config.Empty()
	app := New(runtime.Settings{
		Host:        "127.0.0.1",
		Port:        0,
		MasterKey:   master,
		DatabaseURL: ":memory:",
	}, cfg)
	t.Cleanup(app.Close)
	return app
}

func TestHealthAndBanner(t *testing.T) {
	app := testApp(t, "")
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(b) != "LiteLLM: RUNNING" {
		t.Fatalf("%q", b)
	}

	resp, err = http.Get(ts.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatal(resp.Status)
	}
	resp, err = http.Get(ts.URL + "/health/readiness")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var body map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&body)
	if body["status"] != "connected" || body["litellm_version"] == nil {
		t.Fatalf("%+v", body)
	}
}

func TestMasterKeyAuth(t *testing.T) {
	app := testApp(t, "sk-test")
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	resp, _ := http.Get(ts.URL + "/v1/models")
	if resp.StatusCode != 401 {
		t.Fatalf("unauthed %d", resp.StatusCode)
	}
	resp.Body.Close()

	req, _ := http.NewRequest("GET", ts.URL+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer sk-test")
	resp, _ = http.DefaultClient.Do(req)
	if resp.StatusCode != 200 {
		t.Fatalf("authed %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestModelCRUD(t *testing.T) {
	app := testApp(t, "sk")
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	payload := `{"model_name":"groq-llama","litellm_params":{"model":"groq/llama","api_key":"secret"}}`
	req, _ := http.NewRequest("POST", ts.URL+"/model/new", strings.NewReader(payload))
	req.Header.Set("Authorization", "Bearer sk")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	var created map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&created)
	resp.Body.Close()
	if created["model_id"] == nil {
		t.Fatalf("%+v", created)
	}

	req, _ = http.NewRequest("GET", ts.URL+"/model/info", nil)
	req.Header.Set("Authorization", "Bearer sk")
	resp, _ = http.DefaultClient.Do(req)
	var info map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&info)
	resp.Body.Close()
	data := info["data"].([]any)
	row := data[0].(map[string]any)
	lp := row["litellm_params"].(map[string]any)
	if lp["api_key"] != "****" {
		t.Fatalf("key not redacted: %v", lp["api_key"])
	}

	req, _ = http.NewRequest("GET", ts.URL+"/api/claude_cli/bootstrap", nil)
	resp, _ = http.DefaultClient.Do(req)
	var boot map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&boot)
	resp.Body.Close()
	opts := boot["additional_model_options"].([]any)
	if len(opts) != 1 {
		t.Fatalf("%+v", boot)
	}
}

func TestChatCompletionsViaFakeUpstream(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer up.Close()

	app := testApp(t, "sk")
	app.Router.Add(map[string]any{
		"model_name": "echo",
		"litellm_params": map[string]any{
			"model":    "openai/gpt-4o-mini",
			"api_base": up.URL,
			"api_key":  "k",
		},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	body := `{"model":"echo","messages":[{"role":"user","content":"hi"}]}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Authorization", "Bearer sk")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, raw)
	}
	if !strings.Contains(string(raw), "pong") {
		t.Fatalf("%s", raw)
	}
}

func TestDashScopeChatViaFakeUpstream(t *testing.T) {
	// Native DashScope outbound path: the upstream sees the generation endpoint,
	// {model, input, parameters} body (result_format forced), Bearer auth, and
	// x-dashscope-sse: enable iff streaming; the client always sees OpenAI shape.
	var mu struct {
		path, sse, auth string
		body            map[string]any
	}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.path = r.URL.Path
		mu.sse = r.Header.Get("X-DashScope-SSE")
		mu.auth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(b, &body)
		mu.body = body
		params, _ := body["parameters"].(map[string]any)
		if params != nil && params["incremental_output"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			// Frame 1 replays the prompt history (role-carrying) — the adapter
			// must drop it; frames 2+ are incremental deltas, frame 4 finalizes.
			_, _ = io.WriteString(w, "id:1\nevent:result\n:HTTP_STATUS/200\n"+
				`data:{"output":{"choices":[{"message":{"role":"user","content":"hi"}}]},"request_id":"req-s","usage":{"input_tokens":3,"output_tokens":0,"total_tokens":3}}`+"\n\n")
			_, _ = io.WriteString(w, "id:2\nevent:result\n:HTTP_STATUS/200\n"+
				`data:{"output":{"choices":[{"message":{"content":"he"}}]},"request_id":"req-s","usage":{"input_tokens":3,"output_tokens":1,"total_tokens":4}}`+"\n\n")
			_, _ = io.WriteString(w, "id:3\nevent:result\n:HTTP_STATUS/200\n"+
				`data:{"output":{"choices":[{"message":{"content":"llo"}}]},"request_id":"req-s","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}`+"\n\n")
			_, _ = io.WriteString(w, "id:4\nevent:result\n:HTTP_STATUS/200\n"+
				`data:{"output":{"choices":[{"finish_reason":"stop","message":{"content":""}}],"finish_reason":"stop"},"request_id":"req-s","usage":{"input_tokens":3,"output_tokens":3,"total_tokens":6}}`+"\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"request_id":"req-1","model":"qwen-max","output":{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"pong"}}],"finish_reason":"stop"},"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}`))
	}))
	defer up.Close()

	app := testApp(t, "sk")
	app.Router.Add(map[string]any{
		"model_name": "dashscope-fake",
		"litellm_params": map[string]any{
			"model":    "dashscope/qwen-max",
			"api_base": up.URL,
			"api_key":  "k",
		},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	post := func(body string) string {
		t.Helper()
		req, err := http.NewRequest("POST", ts.URL+"/v1/chat/completions", bytes.NewReader([]byte(body)))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Authorization", "Bearer sk")
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 {
			t.Fatalf("%d %s", resp.StatusCode, raw)
		}
		return string(raw)
	}

	// Non-stream: native path + OpenAI-shaped answer.
	raw := post(`{"model":"dashscope-fake","messages":[{"role":"user","content":"hi"}],"max_tokens":8}`)
	if mu.path != "/api/v1/services/aigc/text-generation/generation" {
		t.Fatalf("upstream path=%s", mu.path)
	}
	if mu.auth != "Bearer k" {
		t.Fatalf("upstream auth=%q", mu.auth)
	}
	if mu.sse != "" {
		t.Fatalf("x-dashscope-sse set on non-stream: %q", mu.sse)
	}
	if mu.body["model"] != "qwen-max" {
		t.Fatalf("upstream model=%v", mu.body["model"])
	}
	input, _ := mu.body["input"].(map[string]any)
	if input == nil {
		t.Fatalf("upstream input missing: %s", raw)
	}
	if _, has := mu.body["messages"]; has {
		t.Fatal("top-level messages leaked into native body")
	}
	params, _ := mu.body["parameters"].(map[string]any)
	if params == nil || params["result_format"] != "message" {
		t.Fatalf("parameters=%v", mu.body["parameters"])
	}
	// R1: native deprecated max_tokens never emitted.
	if params["max_completion_tokens"] != float64(8) {
		t.Fatalf("max_completion_tokens=%v", params["max_completion_tokens"])
	}
	if _, has := params["max_tokens"]; has {
		t.Fatal("deprecated native max_tokens emitted")
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatalf("client body: %v", err)
	}
	if out["id"] != "req-1" || out["model"] != "qwen-max" {
		t.Fatalf("client id/model: %s", raw)
	}
	choices, _ := out["choices"].([]any)
	if len(choices) != 1 {
		t.Fatalf("choices=%v", out["choices"])
	}
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "pong" {
		t.Fatalf("content=%v", msg["content"])
	}
	usage := out["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(3) || usage["completion_tokens"] != float64(2) || usage["total_tokens"] != float64(5) {
		t.Fatalf("usage=%v", out["usage"])
	}

	// Stream: SSE opt-in header + incremental_output; history frame dropped;
	// client sees synthesized OpenAI SSE with [DONE].
	raw = post(`{"model":"dashscope-fake","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if mu.sse != "enable" {
		t.Fatalf("stream x-dashscope-sse=%q", mu.sse)
	}
	params, _ = mu.body["parameters"].(map[string]any)
	if params == nil || params["incremental_output"] != true {
		t.Fatalf("stream parameters=%v", mu.body["parameters"])
	}
	if strings.Contains(raw, `"content":"hi"`) {
		t.Fatalf("history frame leaked into client stream: %s", raw)
	}
	for _, needle := range []string{`"content":"he"`, `"content":"llo"`, `"finish_reason":"stop"`, `"usage"`, "data: [DONE]"} {
		if !strings.Contains(raw, needle) {
			t.Fatalf("missing %s in client stream: %s", needle, raw)
		}
	}
}

func TestMessagesClaudePassthrough(t *testing.T) {
	var gotKey, gotPath string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("X-Api-Key")
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_x","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}`))
	}))
	defer up.Close()

	app := testApp(t, "sk")
	// Swap Anthropic host by using a custom passthrough isn't built-in for claude-*
	// (hardcoded api.anthropic.com). Instead register a non-claude anthropic-family
	// deployment pointing at the fake upstream.
	app.Router.Add(map[string]any{
		"model_name": "glm",
		"litellm_params": map[string]any{
			"model":    "anthropic/glm-5.2",
			"api_base": up.URL,
			"api_key":  "upstream-key",
		},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	// Deliberate update (security fix): non-claude-* /v1/messages requests now
	// require the master key, so this deployment-path test authenticates.
	body := `{"model":"glm","messages":[{"role":"user","content":"hi"}],"max_tokens":8}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, raw)
	}
	if gotKey != "upstream-key" {
		t.Fatalf("x-api-key=%q path=%s body=%s", gotKey, gotPath, raw)
	}
	if gotPath != "/v1/messages" {
		t.Fatalf("path=%s", gotPath)
	}
	if !strings.Contains(string(raw), "ok") {
		t.Fatalf("body %s", raw)
	}
}

func TestKeysSwitchChangesUpstreamAuth(t *testing.T) {
	var gotKey string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("X-Api-Key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_x","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}]}`))
	}))
	defer up.Close()

	app := testApp(t, "sk")
	app.Router.Keys.Put("zai", "default-secret", "ZAI_SUB_KEY", "env")
	app.Router.Keys.Put("tyna", "tyna-secret", "ZAI_SUB_KEY_TYNA", "env")
	app.Router.Add(map[string]any{
		"model_name": "zai/opus",
		"litellm_params": map[string]any{
			"model":        "anthropic/glm-5.3",
			"api_base":     up.URL,
			"api_key":      "default-secret",
			"api_key_name": "zai",
		},
	})
	app.Router.Add(map[string]any{
		"model_name": "zai-tyna/opus",
		"litellm_params": map[string]any{
			"model":        "anthropic/glm-5.3",
			"api_base":     up.URL,
			"api_key":      "tyna-secret",
			"api_key_name": "tyna",
		},
	})

	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	call := func(model string) {
		t.Helper()
		body := `{"model":"` + model + `","messages":[{"role":"user","content":"hi"}],"max_tokens":8}`
		req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		// Deliberate update (security fix): the deployment path now authenticates.
		req.Header.Set("Authorization", "Bearer sk")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s status %d", model, resp.StatusCode)
		}
	}

	call("zai/opus")
	if gotKey != "default-secret" {
		t.Fatalf("before switch key=%q", gotKey)
	}

	payload := `{"target":"zai","key":"tyna"}`
	req, _ := http.NewRequest("POST", ts.URL+"/keys/switch", strings.NewReader(payload))
	req.Header.Set("Authorization", "Bearer sk")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("switch %d %s", resp.StatusCode, raw)
	}
	if !strings.Contains(string(raw), "zai/opus") {
		t.Fatalf("switch body %s", raw)
	}
	if strings.Contains(string(raw), "zai-tyna/opus") {
		t.Fatalf("must not rebind tyna family: %s", raw)
	}

	call("zai/opus")
	if gotKey != "tyna-secret" {
		t.Fatalf("after switch key=%q", gotKey)
	}

	req, _ = http.NewRequest("GET", ts.URL+"/keys", nil)
	req.Header.Set("Authorization", "Bearer sk")
	resp, _ = http.DefaultClient.Do(req)
	var listing map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&listing)
	resp.Body.Close()
	keys, _ := listing["keys"].([]any)
	if len(keys) < 2 {
		t.Fatalf("keys %+v", listing)
	}
}

const anthropicToolSSE = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":3,"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}
`

// sseDataFrames pulls the data payloads out of an SSE body in order; [DONE]
// is reported separately.
func sseDataFrames(t *testing.T, body string) (frames []map[string]any, sawDone bool) {
	t.Helper()
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			sawDone = true
			continue
		}
		var f map[string]any
		if err := json.Unmarshal([]byte(data), &f); err != nil {
			t.Fatalf("bad SSE data %q: %v", data, err)
		}
		frames = append(frames, f)
	}
	return frames, sawDone
}

func TestChatCompletionsAnthropicUpstreamStreamTools(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("upstream path %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, anthropicToolSSE)
	}))
	defer up.Close()

	app := testApp(t, "sk")
	app.Router.Add(map[string]any{
		"model_name": "claude-test",
		"litellm_params": map[string]any{
			"model":    "anthropic/claude-x",
			"api_base": up.URL,
			"api_key":  "k",
		},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	body := `{"model":"claude-test","messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}],"stream":true}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/chat/completions", bytes.NewReader([]byte(body)))
	req.Header.Set("Authorization", "Bearer sk")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, raw)
	}

	frames, sawDone := sseDataFrames(t, string(raw))
	if !sawDone {
		t.Fatalf("no terminal [DONE]: %s", raw)
	}

	var id, name, args string
	var finish any
	firstToolFrameChecked := false
	for _, f := range frames {
		choices, _ := f["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		ch := choices[0].(map[string]any)
		if fr := ch["finish_reason"]; fr != nil {
			finish = fr
		}
		delta := jsonxMap(ch["delta"])
		tcs, _ := delta["tool_calls"].([]any)
		for _, item := range tcs {
			tc := jsonxMap(item)
			if v, ok := tc["id"].(string); ok && v != "" {
				id = v
			}
			fn := jsonxMap(tc["function"])
			if v, ok := fn["name"].(string); ok && v != "" {
				name = v
			}
			if v, ok := fn["arguments"].(string); ok {
				args += v
			}
			if !firstToolFrameChecked {
				firstToolFrameChecked = true
				if id != "toolu_01" || name != "get_weather" {
					t.Fatalf("first tool_calls frame lost identity: id=%q name=%q frame=%v", id, name, tc)
				}
			}
		}
	}
	if id != "toolu_01" || name != "get_weather" {
		t.Fatalf("tool identity lost on stream: id=%q name=%q body=%s", id, name, raw)
	}
	if args != `{"city":"SF"}` {
		t.Fatalf("arguments=%q body=%s", args, raw)
	}
	if finish != "tool_calls" {
		t.Fatalf("finish_reason=%v body=%s", finish, raw)
	}
}

func TestMessagesOpenAIUpstreamStreamTools(t *testing.T) {
	// Regression lock: /v1/messages streaming against an OpenAI-shaped upstream
	// must round-trip tools — delta.tool_calls in, content_block_start tool_use
	// + input_json_delta + stop_reason tool_use out.
	openaiSSE := strings.Join([]string{
		`data: {"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Bash","arguments":""}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":\"ls\"}"}}]}}]}`,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
	}, "\n\n") + "\n\n"
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, openaiSSE)
	}))
	defer up.Close()

	app := testApp(t, "sk")
	app.Router.Add(map[string]any{
		"model_name": "groq-chat",
		"litellm_params": map[string]any{
			"model":    "groq/llama-3.3-70b",
			"api_base": up.URL,
			"api_key":  "k",
		},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	body := `{"model":"groq-chat","messages":[{"role":"user","content":"ls please"}],"stream":true}`
	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer sk")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("%d %s", resp.StatusCode, raw)
	}
	out := string(raw)
	for _, needle := range []string{
		"event: message_start",
		`"type":"tool_use"`,
		`"id":"call_1"`,
		`"name":"Bash"`,
		"input_json_delta",
		`{\"command\":\"ls\"}`,
		`"stop_reason":"tool_use"`,
		"event: message_stop",
	} {
		if !strings.Contains(out, needle) {
			t.Fatalf("missing %q in\n%s", needle, out)
		}
	}
}

func jsonxMap(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return map[string]any{}
}

func TestKeysPutAndDelete(t *testing.T) {
	app := testApp(t, "sk")
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	payload := `{"name":"extra","api_key":"sk-extra"}`
	req, _ := http.NewRequest("POST", ts.URL+"/keys", strings.NewReader(payload))
	req.Header.Set("Authorization", "Bearer sk")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	resp.Body.Close()

	if app.Router.Keys.Value("extra") != "sk-extra" {
		t.Fatal("stored")
	}

	req, _ = http.NewRequest("POST", ts.URL+"/keys/delete", strings.NewReader(`{"name":"extra"}`))
	req.Header.Set("Authorization", "Bearer sk")
	req.Header.Set("Content-Type", "application/json")
	resp, _ = http.DefaultClient.Do(req)
	resp.Body.Close()
	if app.Router.Keys.Get("extra") != nil {
		t.Fatal("still present")
	}
}
