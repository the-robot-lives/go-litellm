package gateway

// Env-gated end-to-end test against the real DashScope realtime host. NEVER
// runs in CI: requires GO_LITELLM_REALTIME_E2E=1 and ALIBABA_MAAS_API_KEY in
// the environment (optional ALIBABA_MAAS_API_BASE overrides the host).
//
//	GO_LITELLM_REALTIME_E2E=1 ALIBABA_MAAS_API_KEY=sk-… go test ./internal/gateway -run TestRealtimeE2E -v

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"

	"github.com/noizu-labs/go-litellm/internal/config"
	"github.com/noizu-labs/go-litellm/internal/runtime"
)

const e2eModel = "alibaba/qwen3.8-omni-flash-realtime"

func e2eApp(t *testing.T) *App {
	t.Helper()
	apiBase := os.Getenv("ALIBABA_MAAS_API_BASE")
	if apiBase == "" {
		apiBase = "https://llm-50geojgn528m31d1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
	}
	cfgYAML := "model_list:\n" +
		"  - model_name: \"" + e2eModel + "\"\n" +
		"    litellm_params:\n" +
		"      model: openai/qwen3.8-omni-flash-realtime\n" +
		"      api_key: \"os.environ/ALIBABA_MAAS_API_KEY\"\n" +
		"      api_base: " + apiBase + "\n" +
		"general_settings:\n" +
		"  master_key: os.environ/GO_LITELLM_TEST_MASTER_KEY\n"
	path := t.TempDir() + "/config.yaml"
	if err := os.WriteFile(path, []byte(cfgYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("GO_LITELLM_TEST_MASTER_KEY", "test-master-key-e2e")
	app := New(runtime.Settings{Host: "127.0.0.1", Port: 0, MasterKey: "test-master-key-e2e", DatabaseURL: ":memory:"}, cfg)
	t.Cleanup(app.Close)
	return app
}

func e2eTone(n int, freq float64) string {
	buf := make([]byte, 0, n*2)
	for i := 0; i < n; i++ {
		v := int16(9000 * math.Sin(2*math.Pi*freq*float64(i)/24000.0))
		buf = append(buf, byte(v), byte(v>>8))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

func TestRealtimeE2EManualMode(t *testing.T) {
	if os.Getenv("GO_LITELLM_REALTIME_E2E") != "1" || os.Getenv("ALIBABA_MAAS_API_KEY") == "" {
		t.Skip("GO_LITELLM_REALTIME_E2E=1 + ALIBABA_MAAS_API_KEY required")
	}
	app := e2eApp(t)
	srv := httptest.NewServer(app.Handler())
	defer srv.Close()
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http") + "/v1/realtime?model=" + e2eModel

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	// Negative: bad key must be rejected before any upgrade.
	badCtx, badCancel := context.WithTimeout(ctx, 10*time.Second)
	_, _, err := websocket.Dial(badCtx, wsURL, &websocket.DialOptions{
		HTTPHeader: map[string][]string{"Authorization": {"Bearer wrong-key"}},
	})
	badCancel()
	if err == nil {
		t.Fatal("bad key must be rejected")
	}
	t.Logf("negative test: bad key rejected (%v)", err)

	conn, _, err := websocket.Dial(ctx, wsURL, &websocket.DialOptions{
		HTTPHeader: map[string][]string{"Authorization": {"Bearer test-master-key-e2e"}},
	})
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close(websocket.StatusNormalClosure, "")

	read := func() map[string]any {
		typ, data, err := conn.Read(ctx)
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		if typ != websocket.MessageText {
			t.Fatalf("non-text frame")
		}
		m := map[string]any{}
		if err := json.Unmarshal(data, &m); err != nil {
			t.Fatalf("frame: %s", data)
		}
		return m
	}
	send := func(v any) {
		b, _ := json.Marshal(v)
		if err := conn.Write(ctx, websocket.MessageText, b); err != nil {
			t.Fatalf("write: %v", err)
		}
	}

	// 1. server speaks first: session.created
	first := read()
	if first["type"] != "session.created" {
		t.Fatalf("first frame = %v, want session.created", first["type"])
	}

	// 2. manual-mode session (turn_detection null, legacy pcm, Tina)
	send(map[string]any{
		"type": "session.update",
		"session": map[string]any{
			"modalities":          []string{"text", "audio"},
			"voice":               "Tina",
			"input_audio_format":  "pcm",
			"output_audio_format": "pcm",
			"turn_detection":      nil,
		},
	})

	// 3. tone + commit + response.create
	send(map[string]any{"type": "input_audio_buffer.append", "audio": e2eTone(3200, 440)})
	send(map[string]any{"type": "input_audio_buffer.append", "audio": e2eTone(3200, 660)})
	send(map[string]any{"type": "input_audio_buffer.commit"})
	send(map[string]any{"type": "response.create", "response": map[string]any{"modalities": []string{"text", "audio"}}})

	// 4. drain until response.done / error; assert renamed deltas arrive.
	audioDeltas, transcriptDeltas, done, sawUpdated := 0, 0, false, false
	for {
		ev := read()
		switch ev["type"] {
		case "error":
			t.Fatalf("upstream error: %v", ev["error"])
		case "session.updated":
			sawUpdated = true
		case "response.output_audio.delta":
			audioDeltas++
		case "response.output_audio_transcript.delta":
			transcriptDeltas++
		case "response.done":
			done = true
			resp := ev["response"].(map[string]any)
			t.Logf("response.done status=%v usage=%v", resp["status"], resp["usage"])
		}
		if done {
			break
		}
	}
	if !sawUpdated {
		t.Error("session.updated not observed")
	}
	if audioDeltas == 0 {
		t.Error("no response.output_audio.delta observed")
	}
	if transcriptDeltas == 0 {
		t.Error("no response.output_audio_transcript.delta observed")
	}
	t.Logf("e2e ok: audio_deltas=%d transcript_deltas=%d", audioDeltas, transcriptDeltas)
}
