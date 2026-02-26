// Package testutil provides testing utilities and fixtures for any-llm.
package testutil

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

// FakeCompletionServer creates an httptest server that captures the raw JSON
// request body and returns a minimal valid OpenAI-compatible chat completion
// response. The captured body is returned so callers can assert on the exact
// JSON field names sent over the wire.
//
// The handler uses t.Errorf for error reporting, which is safe because the
// blocking HTTP client call in the test goroutine synchronises the handler
// goroutine — the handler always completes before the test returns.
func FakeCompletionServer(t *testing.T) (serverURL string, capturedBody func() map[string]any) {
	t.Helper()

	var (
		mu   sync.Mutex
		body map[string]any
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		mu.Lock()
		defer mu.Unlock()

		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("unmarshalling request body: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		// Minimal valid chat completion response.
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-test",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "test-model",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "hello"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
		}`))
	}))

	t.Cleanup(srv.Close)

	return srv.URL, func() map[string]any {
		mu.Lock()
		defer mu.Unlock()

		cp := make(map[string]any, len(body))
		maps.Copy(cp, body)

		return cp
	}
}

// FakeStreamingServer creates an httptest server that captures the raw JSON
// request body and returns a minimal valid OpenAI-compatible streaming (SSE)
// response. The captured body is returned so callers can assert on the exact
// JSON field names sent over the wire.
//
// The handler uses t.Errorf for error reporting, which is safe because the
// blocking HTTP client call in the test goroutine synchronises the handler
// goroutine — the handler always completes before the test returns.
func FakeStreamingServer(t *testing.T) (serverURL string, capturedBody func() map[string]any) {
	t.Helper()

	var (
		mu   sync.Mutex
		body map[string]any
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		mu.Lock()
		defer mu.Unlock()

		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("unmarshalling request body: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// Minimal valid SSE streaming response.
		chunk := `{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`
		done := `{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`

		_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", done)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))

	t.Cleanup(srv.Close)

	return srv.URL, func() map[string]any {
		mu.Lock()
		defer mu.Unlock()

		cp := make(map[string]any, len(body))
		maps.Copy(cp, body)

		return cp
	}
}
