package anthropic_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/anthropic"
)

func TestCompletionStreamToolReplayPreservesRawInput(t *testing.T) {
	for _, arguments := range []string{`{"n":9007199254740993}`, `{ "n": 9007199254740993 }`} {
		t.Run(arguments, func(t *testing.T) {
			wire := make(chan string, 2)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				wire <- string(body)
				var request struct {
					Stream bool `json:"stream"`
				}
				_ = json.Unmarshal(body, &request)
				if !request.Stream {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"id":"m2","type":"message","role":"assistant",`+
						`"model":"claude-sonnet-4-20250514","content":[{"type":"text","text":"done"}],`+
						`"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				argJSON, _ := json.Marshal(arguments)
				events := []string{
					`{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"stop_reason":null,"usage":{"input_tokens":7,"output_tokens":1}}}`,
					`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":"","future":{"version":2}}}`,
					`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"consider"}}`,
					`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-a"}}`,
					`{"type":"content_block_stop","index":0}`,
					`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call-a","name":"lookup","input":{},"future_flag":true}}`,
					`{"type":"content_block_delta","index":1,` +
						`"delta":{"type":"input_json_delta","partial_json":` + string(argJSON) + `}}`,
					`{"type":"content_block_stop","index":1}`,
					`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":11}}`,
					`{"type":"message_stop"}`,
				}
				for _, event := range events {
					var v struct{ Type string }
					_ = json.Unmarshal([]byte(event), &v)
					_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", v.Type, event)
				}
			}))
			defer srv.Close()
			p, newErr := anthropic.New(config.WithBaseURL(srv.URL), config.WithAPIKey("audit-fake-key"))
			if newErr != nil {
				t.Fatal(newErr)
			}
			params := providers.CompletionParams{
				Model:           "claude-sonnet-4-20250514",
				Messages:        []providers.Message{{Role: "user", Content: "lookup"}},
				ReasoningEffort: providers.ReasoningEffortLow,
				Tools: []providers.Tool{{Type: "function", Function: providers.Function{
					Name: "lookup", Parameters: map[string]any{
						"type":       "object",
						"properties": map[string]any{"n": map[string]any{"type": "integer"}},
						"required":   []string{"n"},
					},
				}}},
			}
			streamBefore, _ := json.Marshal(params)
			chunks, errs := p.CompletionStream(t.Context(), params)
			assembled := providers.Message{Role: "assistant", Content: "", Reasoning: &providers.Reasoning{}}
			for c := range chunks {
				b, _ := json.Marshal(c)
				t.Logf("chunk=%s", b)
				d := c.Choices[0].Delta
				if d.Reasoning != nil {
					assembled.Reasoning.Content += d.Reasoning.Content
					if len(d.Reasoning.ProviderRaw) > 0 {
						assembled.Reasoning.ProviderRaw = d.Reasoning.ProviderRaw
						assembled.Reasoning.Provider = d.Reasoning.Provider
					}
				}
				for _, call := range d.ToolCalls {
					if call.Function.Name != "" {
						assembled.ToolCalls = append(assembled.ToolCalls, call)
					} else {
						assembled.ToolCalls[len(assembled.ToolCalls)-1].Function.Arguments += call.Function.Arguments
					}
				}
			}
			if err := <-errs; err != nil {
				t.Fatal(err)
			}
			streamAfter, _ := json.Marshal(params)
			if string(streamBefore) != string(streamAfter) {
				t.Fatal("stream input mutated")
			}
			t.Logf("request=%s", <-wire)
			if len(assembled.Reasoning.ProviderRaw) == 0 {
				t.Fatal("missing full snapshot")
			}
			// Derive expected complete blocks directly from the mock SSE fixture.
			// Number-preserving decoding detects integer rounding and injected fields.
			wantRaw := []byte(`[
				{"type":"thinking","thinking":"consider","signature":"sig-a","future":{"version":2}},
				{"type":"tool_use","id":"call-a","name":"lookup","input":{"n":9007199254740993},"future_flag":true}
			]`)
			var decoded [2]any
			for i, data := range [][]byte{wantRaw, assembled.Reasoning.ProviderRaw} {
				decoder := json.NewDecoder(bytes.NewReader(data))
				decoder.UseNumber()
				if err := decoder.Decode(&decoded[i]); err != nil {
					t.Fatal(err)
				}
			}
			if !reflect.DeepEqual(decoded[0], decoded[1]) {
				t.Errorf("snapshot is not wire-faithful: want=%s got=%s", wantRaw, assembled.Reasoning.ProviderRaw)
			}
			params.Messages = append(params.Messages, assembled, providers.Message{
				Role: "tool", ToolCallID: "call-a", Content: "result",
			})
			before, _ := json.Marshal(params)
			_, replayErr := p.Completion(t.Context(), params)
			after, _ := json.Marshal(params)
			if string(before) != string(after) {
				t.Fatal("input mutated")
			}
			t.Logf("replay input=%s error=%v", before, replayErr)
			select {
			case body := <-wire:
				t.Logf("replay wire=%s", body)
				var request struct {
					Messages []struct {
						Content json.RawMessage `json:"content"`
					} `json:"messages"`
				}
				if err := json.Unmarshal([]byte(body), &request); err != nil || len(request.Messages) != 3 {
					t.Fatalf("invalid replay request: %v", err)
				}
				var replay any
				decoder := json.NewDecoder(bytes.NewReader(request.Messages[1].Content))
				decoder.UseNumber()
				if err := decoder.Decode(&replay); err != nil || !reflect.DeepEqual(decoded[0], replay) {
					t.Errorf("replayed content changed: %s error=%v", request.Messages[1].Content, err)
				}
			default:
				t.Error("replay produced no HTTP request")
			}
			if replayErr != nil {
				t.Fatalf("unaltered streamed tool replay failed: %v", replayErr)
			}
		})
	}
}
