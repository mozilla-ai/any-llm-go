package mistral

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestCompletionStreamRejectsMalformedEvent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		event   string
		wantErr string
	}{
		{name: "invalid JSON", event: `{"id":`, wantErr: "unexpected end"},
		{
			name: "invalid thinking chunk",
			event: `{
				"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,
				"model":"mistral-small-latest","choices":[{"index":0,"delta":{
					"content":[{"type":"thinking","thinking":[{"type":"text"}]}]
				},"finish_reason":null}]
			}`,
			wantErr: "thinking text chunk is missing text",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			event := test.event
			if json.Valid([]byte(event)) {
				event = compactJSON(t, event)
			}

			server := httptest.NewServer(http.HandlerFunc(func(responseWriter http.ResponseWriter, _ *http.Request) {
				responseWriter.Header().Set("Content-Type", "text/event-stream")
				_, err := fmt.Fprintf(responseWriter, "data: %s\n\n", event)
				if err != nil {
					t.Errorf("writing malformed Mistral stream event: %v", err)
				}
			}))
			t.Cleanup(server.Close)

			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL(server.URL),
			)
			require.NoError(t, err)

			chunks, errs := provider.CompletionStream(t.Context(), providers.CompletionParams{
				Model:    mistralReasoningModel,
				Messages: testutil.SimpleMessages(),
			})
			for range chunks {
			}
			require.ErrorContains(t, <-errs, test.wantErr)
		})
	}
}

func TestCompletionStreamHonorsDeadlineAfterThinkingChunk(t *testing.T) {
	t.Parallel()

	const rawEvent = `{
		"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,
		"model":"mistral-small-latest","choices":[{"index":0,"delta":{
			"content":[{"type":"thinking","thinking":[
				{"type":"text","text":"step one"}
			],"closed":false}]
		},"finish_reason":null}]
	}`
	event := compactJSON(t, rawEvent)

	server := httptest.NewServer(http.HandlerFunc(func(responseWriter http.ResponseWriter, request *http.Request) {
		responseWriter.Header().Set("Content-Type", "text/event-stream")
		_, err := fmt.Fprintf(responseWriter, "data: %s\n\n", event)
		if err != nil {
			t.Errorf("writing Mistral stream event: %v", err)
			return
		}
		if flusher, ok := responseWriter.(http.Flusher); ok {
			flusher.Flush()
		}
		<-request.Context().Done()
	}))
	t.Cleanup(server.Close)

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(t.Context(), 500*time.Millisecond)
	defer cancel()
	chunks, errs := provider.CompletionStream(ctx, providers.CompletionParams{
		Model:    mistralReasoningModel,
		Messages: testutil.SimpleMessages(),
	})
	first, ok := <-chunks
	require.True(t, ok)
	require.NotNil(t, first.Choices[0].Delta.Reasoning)
	require.Equal(t, "step one", first.Choices[0].Delta.Reasoning.Content)
	for range chunks {
	}
	require.ErrorIs(t, <-errs, context.DeadlineExceeded)
}
