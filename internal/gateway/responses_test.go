package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type capturedRequest struct {
	path          string
	authorization string
	body          map[string]any
}

func postResponses(t *testing.T, url, key, payload string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, url+"/v1/responses", strings.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

// gatewaySSEFrames splits the gateway's "event: X\ndata: {json}\n\n" stream.
func gatewaySSEFrames(t *testing.T, raw string) []map[string]string {
	t.Helper()
	var frames []map[string]string
	for _, block := range strings.Split(raw, "\n\n") {
		if block == "" {
			continue
		}
		f := map[string]string{}
		for _, line := range strings.Split(block, "\n") {
			if strings.HasPrefix(line, "event: ") {
				f["event"] = strings.TrimPrefix(line, "event: ")
			}
			if strings.HasPrefix(line, "data: ") {
				f["data"] = strings.TrimPrefix(line, "data: ")
			}
		}
		if f["event"] == "" || f["data"] == "" {
			t.Fatalf("malformed frame %q", block)
		}
		frames = append(frames, f)
	}
	return frames
}

func decoded(t *testing.T, resp *http.Response) map[string]any {
	t.Helper()
	defer resp.Body.Close()
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return body
}

// TestResponsesRoundTrip drives POST /v1/responses against a fake OpenAI
// upstream: non-stream translate → Responses JSON; stream → Responses SSE;
// master key required.
func TestResponsesRoundTrip(t *testing.T) {
	var up capturedRequest
	upSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		up.path = r.URL.Path
		up.authorization = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &up.body)
		if strings.Contains(string(b), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"pon\"},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"g\"},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`))
	}))
	defer upSrv.Close()

	app := testApp(t, "sk-test")
	app.Router.Add(map[string]any{
		"model_name":     "echo",
		"litellm_params": map[string]any{"model": "openai/gpt-4o-mini", "api_base": upSrv.URL, "api_key": "k"},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	// Master key required.
	if resp := postResponses(t, ts.URL, "", `{"model":"echo","input":"hi"}`); resp.StatusCode != 401 {
		resp.Body.Close()
		t.Fatalf("unauthed status %d", resp.StatusCode)
	} else {
		resp.Body.Close()
	}

	// Non-stream round trip.
	resp := postResponses(t, ts.URL, "sk-test", `{"model":"echo","instructions":"be brief","input":"hi"}`)
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	body := decoded(t, resp)
	if body["object"] != "response" || body["model"] != "echo" || body["status"] != "completed" {
		t.Fatalf("envelope %v", body)
	}
	if _, ok := body["store"]; ok {
		t.Fatal("store must never be echoed")
	}
	output := body["output"].([]any)
	if text := output[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]; text != "pong" {
		t.Fatalf("output text %v", text)
	}
	usage := body["usage"].(map[string]any)
	if usage["input_tokens"] != 2.0 || usage["output_tokens"] != 3.0 || usage["total_tokens"] != 5.0 {
		t.Fatalf("usage %v", usage)
	}
	if up.path != "/chat/completions" || up.authorization != "Bearer k" {
		t.Fatalf("upstream path/auth %s/%s", up.path, up.authorization)
	}
	msgs := up.body["messages"].([]any)
	if msgs[0].(map[string]any)["role"] != "system" || msgs[0].(map[string]any)["content"] != "be brief" {
		t.Fatalf("instructions message %v", msgs[0])
	}
	if msgs[1].(map[string]any)["role"] != "user" || msgs[1].(map[string]any)["content"] != "hi" {
		t.Fatalf("input message %v", msgs[1])
	}
	if up.body["model"] != "gpt-4o-mini" {
		t.Fatalf("upstream model %v", up.body["model"])
	}

	// Stream round trip: synthesized Responses SSE.
	up.path = ""
	resp = postResponses(t, ts.URL, "sk-test", `{"model":"echo","input":"hi","stream":true}`)
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("%d %s", resp.StatusCode, b)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type %q", ct)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	frames := gatewaySSEFrames(t, string(raw))
	if frames[0]["event"] != "response.created" {
		t.Fatalf("first event %v", frames[0]["event"])
	}
	if frames[len(frames)-1]["event"] != "response.completed" {
		t.Fatalf("last event %v", frames[len(frames)-1]["event"])
	}
	var completed map[string]any
	if err := json.Unmarshal([]byte(frames[len(frames)-1]["data"]), &completed); err != nil {
		t.Fatal(err)
	}
	response := completed["response"].(map[string]any)
	if response["status"] != "completed" {
		t.Fatalf("stream status %v", response["status"])
	}
	streamOut := response["output"].([]any)
	if text := streamOut[0].(map[string]any)["content"].([]any)[0].(map[string]any)["text"]; text != "pong" {
		t.Fatalf("stream assembled text %v", text)
	}
	if u := response["usage"].(map[string]any); u["input_tokens"] != 2.0 {
		t.Fatalf("stream usage %v", u)
	}
	if up.path != "/chat/completions" {
		t.Fatalf("stream upstream path %s", up.path)
	}
}

// TestResponsesToolRoundTrip: function_call/function_call_output items go in
// as chat tool messages + tool_calls; a tool_calls upstream answer comes back
// as function_call items (JSON and SSE).
func TestResponsesToolRoundTrip(t *testing.T) {
	var up capturedRequest
	upSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &up.body)
		if strings.Contains(string(b), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_9\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}]},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":6,\"total_tokens\":10}}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_9","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}`))
	}))
	defer upSrv.Close()

	app := testApp(t, "sk-test")
	app.Router.Add(map[string]any{
		"model_name":     "echo",
		"litellm_params": map[string]any{"model": "openai/gpt-4o-mini", "api_base": upSrv.URL, "api_key": "k"},
	})
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	in := `{"model":"echo","input":[
      {"type":"message","role":"user","content":[{"type":"input_text","text":"weather?"}]},
      {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"SF\"}"},
      {"type":"function_call_output","call_id":"call_1","output":"72F and sunny"}
    ]}`
	streamIn := strings.TrimSuffix(in, "}") + `,"stream":true}`
	resp := postResponses(t, ts.URL, "sk-test", in)
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("%d %s", resp.StatusCode, b)
	}

	// Upstream saw the chat-completions tool shape.
	msgs := up.body["messages"].([]any)
	u0 := msgs[0].(map[string]any)
	if u0["role"] != "user" {
		t.Fatalf("msg0 %v", u0)
	}
	if blk := u0["content"].([]any)[0].(map[string]any); blk["type"] != "text" || blk["text"] != "weather?" {
		t.Fatalf("msg0 content %v", u0["content"])
	}
	u1 := msgs[1].(map[string]any)
	if u1["role"] != "assistant" {
		t.Fatalf("msg1 %v", u1)
	}
	tc := u1["tool_calls"].([]any)[0].(map[string]any)
	if tc["id"] != "call_1" || tc["type"] != "function" {
		t.Fatalf("upstream tool_call %v", tc)
	}
	fn := tc["function"].(map[string]any)
	if fn["name"] != "get_weather" || fn["arguments"] != `{"city":"SF"}` {
		t.Fatalf("upstream function %v", fn)
	}
	u2 := msgs[2].(map[string]any)
	if u2["role"] != "tool" || u2["tool_call_id"] != "call_1" || u2["content"] != "72F and sunny" {
		t.Fatalf("tool output message %v", u2)
	}

	// …and the function_call came back as a Responses item.
	body := decoded(t, resp)
	var fc map[string]any
	for _, item := range body["output"].([]any) {
		if m := item.(map[string]any); m["type"] == "function_call" {
			fc = m
		}
	}
	if fc == nil || fc["call_id"] != "call_9" || fc["name"] != "get_weather" || fc["arguments"] != `{"city":"SF"}` {
		t.Fatalf("response function_call %v", fc)
	}

	// Streamed tool round trip: SSE carries the named function_call open, arg
	// deltas, and a completed envelope with the assembled function_call item.
	up.body = nil
	resp = postResponses(t, ts.URL, "sk-test", streamIn)
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	frames := gatewaySSEFrames(t, string(raw))
	var sawNameOpen, sawArgDelta bool
	for _, f := range frames {
		switch f["event"] {
		case "response.output_item.added":
			if strings.Contains(f["data"], `"name":"get_weather"`) && strings.Contains(f["data"], `"call_id":"call_9"`) {
				sawNameOpen = true
			}
		case "response.function_call_arguments.delta":
			sawArgDelta = true
		}
	}
	if !sawNameOpen || !sawArgDelta {
		t.Fatalf("stream tool events: nameOpen=%v argDelta=%v", sawNameOpen, sawArgDelta)
	}
	var completed map[string]any
	if err := json.Unmarshal([]byte(frames[len(frames)-1]["data"]), &completed); err != nil {
		t.Fatal(err)
	}
	response := completed["response"].(map[string]any)
	var streamFC map[string]any
	for _, item := range response["output"].([]any) {
		if m := item.(map[string]any); m["type"] == "function_call" {
			streamFC = m
		}
	}
	if streamFC == nil || streamFC["call_id"] != "call_9" || streamFC["arguments"] != `{"city":"SF"}` {
		t.Fatalf("stream assembled function_call %v", streamFC)
	}
}

// TestResponsesStubsAndAuth: stateless stubs for the id-addressed routes and
// proof the new dispatch cases don't shadow sibling routes.
func TestResponsesStubsAndAuth(t *testing.T) {
	app := testApp(t, "sk-test")
	ts := httptest.NewServer(app.Handler())
	defer ts.Close()

	get := func(path string) *http.Response {
		req, _ := http.NewRequest(http.MethodGet, ts.URL+path, nil)
		req.Header.Set("Authorization", "Bearer sk-test")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	// GET /v1/responses/{id} → documented 404 not_found (stateless), both spellings.
	for _, path := range []string{"/v1/responses/resp_x", "/responses/resp_x"} {
		resp := get(path)
		if resp.StatusCode != 404 {
			b, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			t.Fatalf("GET %s: %d %s", path, resp.StatusCode, b)
		}
		body := decoded(t, resp)
		errBody := body["error"].(map[string]any)
		if errBody["type"] != "not_found_error" {
			t.Fatalf("GET %s: envelope %v", path, body)
		}
	}

	// DELETE / PUT / POST on the id shape → 405, never forwarded.
	for _, method := range []string{http.MethodDelete, http.MethodPut, http.MethodPost} {
		req, _ := http.NewRequest(method, ts.URL+"/v1/responses/resp_x", nil)
		req.Header.Set("Authorization", "Bearer sk-test")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != 405 {
			b, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			t.Fatalf("%s id route: %d %s", method, resp.StatusCode, b)
		}
		resp.Body.Close()
	}

	// previous_response_id → 400 naming the field.
	resp := postResponses(t, ts.URL, "sk-test", `{"model":"echo","input":"hi","previous_response_id":"resp_old"}`)
	if resp.StatusCode != 400 {
		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("previous_response_id: %d %s", resp.StatusCode, b)
	}
	body := decoded(t, resp)
	if msg := body["error"].(map[string]any)["message"]; !strings.Contains(msg.(string), "previous_response_id") {
		t.Fatalf("400 message %v", msg)
	}

	// Sibling routes unaffected: embeddings still requires auth, models works.
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/v1/embeddings", strings.NewReader(`{"model":"echo","input":"x"}`))
	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if resp2.StatusCode != 401 {
		b, _ := io.ReadAll(resp2.Body)
		resp2.Body.Close()
		t.Fatalf("embeddings without key: %d %s", resp2.StatusCode, b)
	}
	resp2.Body.Close()
	if resp3 := get("/v1/models"); resp3.StatusCode != 200 {
		b, _ := io.ReadAll(resp3.Body)
		resp3.Body.Close()
		t.Fatalf("models: %d %s", resp3.StatusCode, b)
	} else {
		resp3.Body.Close()
	}

	// GET /v1/responses (collection) is NOT intercepted by the responses
	// dispatch — it falls to passthrough (never our 404/405 stubs).
	resp4 := get("/v1/responses")
	if resp4.StatusCode == 404 || resp4.StatusCode == 405 {
		b, _ := io.ReadAll(resp4.Body)
		resp4.Body.Close()
		t.Fatalf("collection GET intercepted: %d %s", resp4.StatusCode, b)
	}
	resp4.Body.Close()
}
