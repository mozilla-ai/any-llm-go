package mistral

import (
	"bytes"
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
	sourceProvider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(mistralCompletionServer(t, mistralThinkingContent)),
	)
	require.NoError(t, err)
	source, err := sourceProvider.Completion(t.Context(), providers.CompletionParams{
		Model: mistralReasoningModel, Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)

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
			source.Choices[0].Message,
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

func TestCompletionUsesPortableReasoningForForeignProviderRaw(t *testing.T) {
	t.Parallel()

	serverURL, capturedRequest := mistralReplayServer(t)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(serverURL))
	require.NoError(t, err)

	_, err = provider.Completion(t.Context(), providers.CompletionParams{
		Model: mistralReasoningModel,
		Messages: []providers.Message{{
			Role:    providers.RoleAssistant,
			Content: "answer",
			Reasoning: &providers.Reasoning{
				Content:     "thought",
				ProviderRaw: json.RawMessage(`[{"type":"thinking","thinking":"anthropic"}]`),
				Provider:    "anthropic",
			},
		}},
	})
	require.NoError(t, err)
	require.JSONEq(t, `[
		{"type":"thinking","thinking":[{"type":"text","text":"thought"}]},
		{"type":"text","text":"answer"}
	]`, string(capturedRequest().Messages[0].Content))
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
				Provider:    providerName,
			},
		}},
	})
	for range chunks {
		// Drain the channel before checking the terminal stream error.
	}

	require.NoError(t, <-errs)

	rawRequest, err := json.Marshal(capturedBody())
	require.NoError(t, err)

	var request capturedMistralRequest

	require.NoError(t, json.Unmarshal(rawRequest, &request))
	require.Len(t, request.Messages, 1)
	require.JSONEq(t, mistralThinkingContent, string(request.Messages[0].Content))
}

func TestCompletionReplaysUnsignedNormalizedReasoningFullRequest(t *testing.T) {
	t.Parallel()

	serverURL, capturedBody := testutil.FakeCompletionServer(t)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	params := providers.CompletionParams{
		Model: mistralReasoningModel,
		Messages: []providers.Message{
			{
				Role:      providers.RoleAssistant,
				Content:   "Answer.",
				Reasoning: &providers.Reasoning{Content: "Prior thinking."},
			},
			{Role: providers.RoleUser, Content: "Continue."},
		},
	}
	before, err := json.Marshal(params)
	require.NoError(t, err)

	_, err = provider.Completion(t.Context(), params)
	require.NoError(t, err)

	after, err := json.Marshal(params)
	require.NoError(t, err)
	require.True(t, bytes.Equal(before, after), "completion mutated caller params")
	require.Equal(t, map[string]any{
		"model": mistralReasoningModel,
		"messages": []any{
			map[string]any{
				"role": providers.RoleAssistant,
				"content": []any{
					map[string]any{
						"type": "thinking",
						"thinking": []any{
							map[string]any{"type": "text", "text": "Prior thinking."},
						},
					},
					map[string]any{"type": "text", "text": "Answer."},
				},
			},
			map[string]any{"role": providers.RoleUser, "content": "Continue."},
		},
	}, capturedBody())
}

func TestCompletionStreamReplaysUnsignedNormalizedReasoningFullRequest(t *testing.T) {
	t.Parallel()

	serverURL, capturedBody := testutil.FakeStreamingServer(t)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	params := providers.CompletionParams{
		Model: mistralReasoningModel,
		Messages: []providers.Message{{
			Role:      providers.RoleAssistant,
			Content:   "Answer.",
			Reasoning: &providers.Reasoning{Content: "Prior thinking."},
		}},
	}
	before, err := json.Marshal(params)
	require.NoError(t, err)

	chunks, errs := provider.CompletionStream(t.Context(), params)
	for range chunks {
		// Drain the channel before checking the terminal stream error.
	}
	require.NoError(t, <-errs)

	after, err := json.Marshal(params)
	require.NoError(t, err)
	require.True(t, bytes.Equal(before, after), "streaming completion mutated caller params")
	require.Equal(t, map[string]any{
		"model":  mistralReasoningModel,
		"stream": true,
		"messages": []any{map[string]any{
			"role": providers.RoleAssistant,
			"content": []any{
				map[string]any{
					"type": "thinking",
					"thinking": []any{
						map[string]any{"type": "text", "text": "Prior thinking."},
					},
				},
				map[string]any{"type": "text", "text": "Answer."},
			},
		}},
	}, capturedBody())
}

func TestCompletionLeavesNilAndEmptyReasoningUnchanged(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		reasoning *providers.Reasoning
	}{
		{name: "nil reasoning"},
		{name: "empty reasoning", reasoning: &providers.Reasoning{}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			serverURL, capturedBody := testutil.FakeCompletionServer(t)
			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL(serverURL),
			)
			require.NoError(t, err)

			params := providers.CompletionParams{
				Model: mistralReasoningModel,
				Messages: []providers.Message{{
					Role: providers.RoleAssistant, Content: "Answer.", Reasoning: test.reasoning,
				}},
			}
			before, err := json.Marshal(params)
			require.NoError(t, err)

			_, err = provider.Completion(t.Context(), params)
			require.NoError(t, err)

			after, err := json.Marshal(params)
			require.NoError(t, err)
			require.True(t, bytes.Equal(before, after), "completion mutated caller params")
			require.Equal(t, map[string]any{
				"model": mistralReasoningModel,
				"messages": []any{map[string]any{
					"role": providers.RoleAssistant, "content": "Answer.",
				}},
			}, capturedBody())
		})
	}
}

func TestCompletionRejectsInvalidThinkingReplay(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		raw     json.RawMessage
		content any
		wantErr string
	}{
		{name: "not an array", raw: json.RawMessage(`{"type":"thinking"}`), wantErr: "invalid reasoning provider_raw"},
		{name: "malformed JSON", raw: json.RawMessage(`[`), wantErr: "unexpected end of JSON input"},
		{
			name:    "unknown nested thinking type",
			raw:     json.RawMessage(`[{"type":"thinking","thinking":[{"type":"future_thinking"}]}]`),
			wantErr: `unsupported thinking chunk type "future_thinking"`,
		},
		{
			name: "missing answer", raw: json.RawMessage(mistralThinkingContent),
			wantErr: "reasoning provider_raw does not match assistant content",
		},
		{
			name: "different answer", raw: json.RawMessage(mistralThinkingContent), content: "different",
			wantErr: "reasoning provider_raw does not match assistant content",
		},
		{
			name: "unexpected answer", raw: json.RawMessage(`[{"type":"thinking","thinking":[]}]`), content: "answer",
			wantErr: "reasoning provider_raw does not match assistant content",
		},
		{
			name: "multipart content", raw: json.RawMessage(mistralThinkingContent),
			content: []providers.ContentPart{{Type: "text", Text: "the answer"}},
			wantErr: "reasoning replay requires string or nil content",
		},
		{
			name:    "unsigned multipart content",
			content: []providers.ContentPart{{Type: "text", Text: "the answer"}},
			wantErr: "reasoning replay requires string or nil content",
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

			params := providers.CompletionParams{
				Model: mistralReasoningModel,
				Messages: []providers.Message{{
					Role:    providers.RoleAssistant,
					Content: test.content,
					Reasoning: &providers.Reasoning{
						Content:     "thinking",
						ProviderRaw: test.raw,
						Provider:    providerName,
					},
				}},
			}
			_, err = provider.Completion(t.Context(), params)
			require.ErrorContains(t, err, test.wantErr)

			chunks, errs := provider.CompletionStream(t.Context(), params)
			for range chunks {
				// Drain the channel before checking the terminal stream error.
			}

			require.ErrorContains(t, <-errs, test.wantErr)
			require.Empty(t, capturedRequest().Messages)
		})
	}
}
