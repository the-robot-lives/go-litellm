package core

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/noizu-labs/go-litellm/internal/errx"
	"github.com/noizu-labs/go-litellm/internal/jsonx"
	"github.com/noizu-labs/go-litellm/internal/providers"
)

// HTTPDoer is the outbound client surface (injectable in tests).
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// DefaultClient never advertises compression (bodies are relayed verbatim).
var DefaultClient = &http.Client{
	Timeout: 10 * time.Minute,
	Transport: &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		MaxIdleConns:        200,
		IdleConnTimeout:     30 * time.Second,
		DisableCompression:  true,
		TLSHandshakeTimeout: 15 * time.Second,
	},
}

// DoCompletion runs validate → URL → transform → POST → transform_response.
func DoCompletion(client HTTPDoer, p *Prepared) (*providers.ModelResponse, *errx.Error) {
	body, e := DoRaw(client, p.Adapter, p.Req)
	if e != nil {
		return nil, e
	}
	return p.Adapter.TransformResponse(body, p.Req), nil
}

// DoRaw POSTs and returns the decoded upstream JSON body.
// Honors litellm_params.num_retries (default 2, litellm convention); retries
// cover transport errors and 408/429/5xx responses with capped backoff.
func DoRaw(client HTTPDoer, ad providers.Adapter, req *providers.Request) (map[string]any, *errx.Error) {
	if client == nil {
		client = DefaultClient
	}
	headers, e := ad.ValidateEnvironment(req, map[string]string{})
	if e != nil {
		return nil, e
	}
	url := ad.CompleteURL(req)
	payload := ad.TransformRequest(req)
	raw, _ := json.Marshal(payload)

	timeout := timeoutOf(req.LiteLLMParams, 10*time.Minute)
	maxRetries := retriesOf(req.LiteLLMParams)
	var last *errx.Error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			time.Sleep(retryBackoff(attempt))
		}
		httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(raw))
		if err != nil {
			return nil, errx.New(502, "upstream request failed: "+err.Error())
		}
		for k, v := range headers {
			httpReq.Header.Set(k, v)
		}
		httpReq.Header.Del("Accept-Encoding")
		c := client
		if hc, ok := client.(*http.Client); ok && timeout > 0 {
			clone := *hc
			clone.Timeout = timeout
			c = &clone
		}
		resp, err := c.Do(httpReq)
		if err != nil {
			if attempt < maxRetries && retryable(err) {
				last = errx.New(502, "upstream request failed: "+err.Error())
				continue
			}
			return nil, errx.New(502, "upstream request failed: "+err.Error())
		}
		b, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		decoded := decodeBody(b)
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return decoded, nil
		}
		last = ad.ErrorClass(resp.StatusCode, decoded, nil)
		if attempt < maxRetries && retryableStatus(resp.StatusCode) {
			continue
		}
		return nil, last
	}
	return nil, last
}

// retriesOf reads litellm_params.num_retries; default 2 (litellm convention),
// clamped to [0, 8] so a wild config value can't pin a worker.
func retriesOf(lp map[string]any) int {
	n := 2
	if lp != nil {
		if v, ok := jsonx.Float(lp, "num_retries"); ok {
			n = int(v)
		}
	}
	if n < 0 {
		n = 0
	}
	if n > 8 {
		n = 8
	}
	return n
}

// retryableStatus mirrors litellm's retry posture: timeout, rate limit, and 5xx.
func retryableStatus(code int) bool {
	return code == http.StatusRequestTimeout ||
		code == http.StatusTooManyRequests ||
		code >= 500
}

// retryBackoff is a capped exponential: 500ms, 1s, 2s, 4s, 4s, …
func retryBackoff(attempt int) time.Duration {
	d := 500 * time.Millisecond
	for i := 1; i < attempt && d < 4*time.Second; i++ {
		d *= 2
	}
	if d > 4*time.Second {
		d = 4 * time.Second
	}
	return d
}

func decodeBody(b []byte) map[string]any {
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return map[string]any{"error": map[string]any{"message": string(b)}}
	}
	return m
}

func timeoutOf(lp map[string]any, def time.Duration) time.Duration {
	if n, ok := jsonx.Float(lp, "timeout"); ok {
		return time.Duration(n * float64(time.Second))
	}
	return def
}

// streamTimeoutOf prefers litellm_params.stream_timeout, then timeout; 0 means
// "no override" (the caller's client timeout applies).
func streamTimeoutOf(lp map[string]any) time.Duration {
	if n, ok := jsonx.Float(lp, "stream_timeout"); ok {
		return time.Duration(n * float64(time.Second))
	}
	return timeoutOf(lp, 0)
}

func retryable(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	for _, needle := range []string{
		"connection reset", "broken pipe", "eof", "use of closed network connection",
		"connection refused", "server closed idle connection",
	} {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}

// HeaderList converts a header map to a slice of pairs (for forwarding).
func HeaderMap(h http.Header) map[string]string {
	out := map[string]string{}
	for k, vs := range h {
		if len(vs) > 0 {
			out[k] = vs[0]
		}
	}
	return out
}
