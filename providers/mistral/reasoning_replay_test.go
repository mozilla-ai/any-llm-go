package mistral

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

type capturedMistralRequest struct {
	Messages []struct {
		Content json.RawMessage `json:"content"`
		Role    string          `json:"role"`
	} `json:"messages"`
}

func mistralReplayServer(t *testing.T) (string, func() capturedMistralRequest) {
	t.Helper()

	var (
		mutex   sync.Mutex
		request capturedMistralRequest
	)
	server := httptest.NewServer(http.HandlerFunc(func(responseWriter http.ResponseWriter, requestBody *http.Request) {
		mutex.Lock()
		err := json.NewDecoder(requestBody.Body).Decode(&request)
		mutex.Unlock()
		if err != nil {
			t.Errorf("decoding Mistral request: %v", err)
			http.Error(responseWriter, "bad request", http.StatusBadRequest)
			return
		}

		responseWriter.Header().Set("Content-Type", "application/json")
		_, err = fmt.Fprintf(responseWriter, `{
			"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
			"model":"%s","choices":[{"index":0,
			"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]
		}`, mistralReasoningModel)
		if err != nil {
			t.Errorf("writing Mistral response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	return server.URL, func() capturedMistralRequest {
		mutex.Lock()
		defer mutex.Unlock()

		return request
	}
}

func TestCompletionReplaysThinkingContent(t *testing.T) {
	t.Parallel()

	// Mistral requires the complete assistant content array in later turns.
	// https://docs.mistral.ai/studio-api/conversations/reasoning
	serverURL, capturedRequest := mistralReplayServer(t)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model: mistralReasoningModel,
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "first question"},
			{
				Role:    providers.RoleAssistant,
				Content: "the answer",
				Reasoning: &providers.Reasoning{
					Content:     "step one\nstep two",
					ProviderRaw: json.RawMessage(mistralThinkingContent),
				},
			},
			{Role: providers.RoleUser, Content: "follow-up"},
		},
	})
	require.NoError(t, err)
	require.Equal(t, "done", completion.Choices[0].Message.Content)

	request := capturedRequest()
	require.Len(t, request.Messages, 3)
	require.Equal(t, providers.RoleAssistant, request.Messages[1].Role)
	require.JSONEq(t, mistralThinkingContent, string(request.Messages[1].Content))
}

func TestCompletionStreamReplaysThinkingContent(t *testing.T) {
	t.Parallel()

	serverURL, capturedBody := testutil.FakeStreamingServer(t)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	chunks, errs := provider.CompletionStream(t.Context(), providers.CompletionParams{
		Model: mistralReasoningModel,
		Messages: []providers.Message{{
			Role:    providers.RoleAssistant,
			Content: "the answer",
			Reasoning: &providers.Reasoning{
				Content:     "step one\nstep two",
				ProviderRaw: json.RawMessage(mistralThinkingContent),
			},
		}},
	})
	for range chunks {
	}
	require.NoError(t, <-errs)

	rawRequest, err := json.Marshal(capturedBody())
	require.NoError(t, err)
	var request capturedMistralRequest
	require.NoError(t, json.Unmarshal(rawRequest, &request))
	require.Len(t, request.Messages, 1)
	require.JSONEq(t, mistralThinkingContent, string(request.Messages[0].Content))
}

func TestCompletionStreamRejectsInvalidThinkingReplay(t *testing.T) {
	t.Parallel()

	serverURL, capturedBody := testutil.FakeStreamingServer(t)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	chunks, errs := provider.CompletionStream(t.Context(), providers.CompletionParams{
		Model: mistralReasoningModel,
		Messages: []providers.Message{{
			Role: providers.RoleAssistant,
			Reasoning: &providers.Reasoning{
				ProviderRaw: json.RawMessage(`{"type":"thinking"}`),
			},
		}},
	})
	for range chunks {
	}
	require.ErrorContains(t, <-errs, "invalid reasoning provider_raw")
	require.Empty(t, capturedBody())
}

func TestCompletionRejectsInvalidThinkingReplay(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		raw     json.RawMessage
		wantErr string
	}{
		{name: "not an array", raw: json.RawMessage(`{"type":"thinking"}`), wantErr: "invalid reasoning provider_raw"},
		{name: "malformed JSON", raw: json.RawMessage(`[`), wantErr: "unexpected end of JSON input"},
		{
			name:    "unknown nested thinking type",
			raw:     json.RawMessage(`[{"type":"thinking","thinking":[{"type":"future_thinking"}]}]`),
			wantErr: `unsupported thinking chunk type "future_thinking"`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			serverURL, capturedRequest := mistralReplayServer(t)
			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL(serverURL),
			)
			require.NoError(t, err)

			_, err = provider.Completion(t.Context(), providers.CompletionParams{
				Model: mistralReasoningModel,
				Messages: []providers.Message{{
					Role:    providers.RoleAssistant,
					Content: "answer",
					Reasoning: &providers.Reasoning{
						Content:     "thinking",
						ProviderRaw: test.raw,
					},
				}},
			})
			require.ErrorContains(t, err, test.wantErr)
			require.Empty(t, capturedRequest().Messages)
		})
	}
}
