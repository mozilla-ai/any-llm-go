package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"testing/synctest"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const anthropicMessageStartSSE = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant",` +
	`"content":[],"model":"claude-opus-5","stop_reason":null,"stop_sequence":null,` +
	`"usage":{"input_tokens":3,"output_tokens":0}}}` + "\n\n"

func TestCompletionStreamKeepsIndexedToolArgumentsAcrossTextBlocks(t *testing.T) {
	t.Parallel()

	events := []string{
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool-A","name":"alpha","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"preface29"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tool-B","name":"beta","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"17}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"y\":31}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":11}}`,
		`{"type":"message_stop"}`,
	}
	provider := newStreamTestProvider(t, func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(writer, anthropicMessageStartSSE)
		for _, event := range events {
			var envelope struct{ Type string }
			if err := json.Unmarshal([]byte(event), &envelope); err != nil {
				t.Error(err)
				return
			}
			_, _ = fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", envelope.Type, event)
		}
	})
	chunks, errs := provider.CompletionStream(t.Context(), streamTestParams())
	arguments := map[string]string{}
	var text string
	for chunk := range chunks {
		for _, choice := range chunk.Choices {
			text += choice.Delta.Content
			for _, tool := range choice.Delta.ToolCalls {
				arguments[tool.ID] += tool.Function.Arguments
			}
		}
	}
	for err := range errs {
		require.NoError(t, err)
	}
	require.Equal(t, "preface29", text)
	require.Equal(t, map[string]string{"tool-A": `{"x":17}`, "tool-B": `{"y":31}`}, arguments)
}

func TestCompletionPreservesModelNotFoundError(t *testing.T) {
	t.Parallel()

	provider := newStreamTestProvider(t, func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusNotFound)
		_, _ = fmt.Fprint(
			writer,
			`{"type":"error","error":{"type":"not_found_error","message":"model: missing-model"}}`,
		)
	})
	_, err := provider.Completion(t.Context(), streamTestParams())
	require.ErrorIs(t, err, errors.ErrModelNotFound)
}

func TestCompletionStreamRequiresMessageStop(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		events    string
		wantError bool
	}{
		{
			name: "complete stream",
			events: anthropicMessageStartSSE +
				"event: content_block_start\n" +
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
				"event: content_block_delta\n" +
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}` + "\n\n" +
				"event: content_block_stop\n" +
				`data: {"type":"content_block_stop","index":0}` + "\n\n" +
				"event: message_delta\n" +
				`data: {"type":"message_delta",` +
				`"delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}` + "\n\n" +
				"event: message_stop\n" +
				`data: {"type":"message_stop"}` + "\n\n",
		},
		{
			name:      "EOF before message_stop",
			events:    anthropicMessageStartSSE,
			wantError: true,
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			provider := newStreamTestProvider(t, func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")

				if _, err := fmt.Fprint(writer, testCase.events); err != nil {
					t.Errorf("write stream events: %v", err)
				}
			})

			chunks, errs := provider.CompletionStream(t.Context(), streamTestParams())

			var received []providers.ChatCompletionChunk
			for chunk := range chunks {
				received = append(received, chunk)
			}

			err := <-errs

			if testCase.wantError {
				require.ErrorIs(t, err, errors.ErrProvider)

				return
			}

			require.NoError(t, err)
			require.Len(t, received, 3)
			require.Equal(t, "hello", received[1].Choices[0].Delta.Content)
			require.Equal(t, providers.FinishReasonStop, received[2].Choices[0].FinishReason)
		})
	}
}

func TestCompletionStreamReportsSSEError(t *testing.T) {
	t.Parallel()

	provider := newStreamTestProvider(t, func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")

		if _, err := fmt.Fprint(
			writer,
			"event: error\n"+
				`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`+"\n\n",
		); err != nil {
			t.Errorf("write stream error: %v", err)
		}
	})

	chunks, errs := provider.CompletionStream(t.Context(), streamTestParams())
	for range chunks {
	}

	require.ErrorIs(t, <-errs, errors.ErrProvider)
}

func TestSendChunkCancellationUnblocksOutput(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()

		chunks := make(chan providers.ChatCompletionChunk)

		done := make(chan bool, 1)
		go func() { done <- sendChunk(ctx, chunks, providers.ChatCompletionChunk{}) }()

		synctest.Wait()
		require.Empty(t, done)
		cancel()
		require.False(t, <-done)
	})
}

func TestCompletionStreamCancelledRequest(t *testing.T) {
	t.Parallel()

	provider := newStreamTestProvider(t, func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")

		if _, err := fmt.Fprint(writer, anthropicMessageStartSSE); err != nil {
			t.Errorf("write stream start: %v", err)
		}

		if err := http.NewResponseController(writer).Flush(); err != nil {
			t.Errorf("flush stream start: %v", err)

			return
		}

		<-request.Context().Done()
	})

	ctx, cancel := context.WithCancel(t.Context())
	_, errs := provider.CompletionStream(ctx, streamTestParams())

	cancel()

	require.ErrorIs(t, <-errs, context.Canceled)
}

func TestCompletionStreamToolArgumentsAreDeltas(t *testing.T) {
	t.Parallel()

	// Independently constructed from the documented event sequence, including a
	// second tool with no arguments: metadata must not depend on a JSON delta.
	// https://platform.claude.com/docs/en/build-with-claude/streaming#input-json-delta
	events := []string{
		`{"type":"content_block_start","index":0,` +
			`"content_block":{"type":"tool_use","id":"tool_a","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"key\":"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"value\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,` +
			`"content_block":{"type":"tool_use","id":"tool_b","name":"clock","input":{}}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`,
	}
	provider := newStreamTestProvider(t, func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")

		if _, err := fmt.Fprint(writer, anthropicMessageStartSSE); err != nil {
			t.Errorf("write message start: %v", err)
		}

		for _, event := range events {
			var envelope struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal([]byte(event), &envelope); err != nil {
				t.Errorf("decode fixture: %v", err)

				return
			}

			if _, err := fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", envelope.Type, event); err != nil {
				t.Errorf("write event: %v", err)
			}
		}
	})

	chunks, errs := provider.CompletionStream(t.Context(), streamTestParams())

	var calls []providers.ToolCall

	for chunk := range chunks {
		for _, choice := range chunk.Choices {
			calls = append(calls, choice.Delta.ToolCalls...)
		}
	}

	require.NoError(t, <-errs)
	require.Equal(t, []providers.ToolCall{
		{ID: "tool_a", Type: "function", Function: providers.FunctionCall{Name: "lookup"}},
		{ID: "tool_a", Type: "function", Function: providers.FunctionCall{Arguments: `{"key":`}},
		{ID: "tool_a", Type: "function", Function: providers.FunctionCall{Arguments: `"value"}`}},
		{ID: "tool_b", Type: "function", Function: providers.FunctionCall{Name: "clock"}},
	}, calls)
}

func newStreamTestProvider(t *testing.T, handler http.HandlerFunc) *Provider {
	t.Helper()

	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	return provider
}

func streamTestParams() providers.CompletionParams {
	return providers.CompletionParams{
		Model:    "claude-opus-5",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	}
}
