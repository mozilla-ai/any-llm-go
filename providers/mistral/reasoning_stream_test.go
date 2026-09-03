package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func compactJSON(t *testing.T, value string) string {
	t.Helper()

	var compact bytes.Buffer
	require.NoError(t, json.Compact(&compact, []byte(value)))

	return compact.String()
}

func TestCompletionStreamPreservesThinkingChunks(t *testing.T) {
	t.Parallel()

	// Mistral streams thinking arrays, a mixed thinking-to-answer transition,
	// then ordinary answer strings.
	// https://docs.mistral.ai/studio-api/conversations/reasoning
	events := []string{
		`{
			"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,
			"model":"mistral-small-latest","choices":[{"index":0,"delta":{
				"role":"assistant","content":[
					{"type":"thinking","thinking":[
						{"type":"text","text":"step one","reference_ids":[{}]},
						{"type":"reference","reference_ids":[1,1e2,999999999999999999999999,"ref-1"],"text":{}}
					],"signature":null,"closed":false},
					{"type":"future_chunk","text":{},"future":true}
				]
			},"finish_reason":null}]
		}`,
		`{
			"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,
			"model":"mistral-small-latest","choices":[{"index":0,"delta":{"content":[
				{"type":"thinking","thinking":[{"type":"text","text":"step two"}],
				 "signature":"sig-2","closed":true},
				{"type":"text","text":"the "}
			]},"finish_reason":null}]
		}`,
		`{
			"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,
			"model":"mistral-small-latest","choices":[{
				"index":0,"delta":{"content":"answer"},"finish_reason":null
			}]
		}`,
		`{
			"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,
			"model":"mistral-small-latest","choices":[{
				"index":0,"delta":{},"finish_reason":"error"
			}]
		}`,
	}
	for i := range events {
		events[i] = compactJSON(t, events[i])
	}

	server := httptest.NewServer(http.HandlerFunc(func(responseWriter http.ResponseWriter, _ *http.Request) {
		responseWriter.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			_, err := fmt.Fprintf(responseWriter, "data: %s\n\n", event)
			if err != nil {
				t.Errorf("writing Mistral stream event: %v", err)
				return
			}
		}
		_, err := fmt.Fprint(responseWriter, "data: [DONE]\n\n")
		if err != nil {
			t.Errorf("writing Mistral stream terminator: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	chunkChannel, errChannel := provider.CompletionStream(t.Context(), providers.CompletionParams{
		Model:    mistralReasoningModel,
		Messages: testutil.SimpleMessages(),
	})
	chunks := make([]providers.ChatCompletionChunk, 0, len(events))
	for chunk := range chunkChannel {
		chunks = append(chunks, chunk)
	}
	require.NoError(t, <-errChannel)
	require.Len(t, chunks, len(events))

	first := chunks[0].Choices[0].Delta
	require.Empty(t, first.Content)
	require.NotNil(t, first.Reasoning)
	require.Equal(t, "step one", first.Reasoning.Content)
	require.JSONEq(
		t,
		`[
			{"type":"thinking","thinking":[
				{"type":"text","text":"step one","reference_ids":[{}]},
				{"type":"reference","reference_ids":[1,1e2,999999999999999999999999,"ref-1"],"text":{}}
			],"signature":null,"closed":false},
			{"type":"future_chunk","text":{},"future":true}
		]`,
		string(first.Reasoning.ProviderRaw),
	)

	transition := chunks[1].Choices[0].Delta
	require.Equal(t, "the ", transition.Content)
	require.NotNil(t, transition.Reasoning)
	require.Equal(t, "step two", transition.Reasoning.Content)
	require.JSONEq(
		t,
		`[
			{"type":"thinking","thinking":[{"type":"text","text":"step two"}],
			 "signature":"sig-2","closed":true},
			{"type":"text","text":"the "}
		]`,
		string(transition.Reasoning.ProviderRaw),
	)

	require.Equal(t, "answer", chunks[2].Choices[0].Delta.Content)
	require.Nil(t, chunks[2].Choices[0].Delta.Reasoning)
	require.Equal(t, "error", chunks[3].Choices[0].FinishReason)
}
