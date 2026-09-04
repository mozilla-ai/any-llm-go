package deepseek

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	llmerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func deepSeekCompletionServer(
	t *testing.T,
	response string,
) (serverURL string, requestBody <-chan map[string]any) {
	t.Helper()

	captured := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding DeepSeek request: %v", err)
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		captured <- body

		w.Header().Set("Content-Type", "application/json")
		// DeepSeek can send blank keep-alive lines before a non-streaming body.
		_, err := fmt.Fprint(w, "\n\n", response)
		if err != nil {
			t.Errorf("writing DeepSeek response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	return server.URL, captured
}

func TestCompletionMapsCurrentThinkingRequest(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		effort       providers.ReasoningEffort
		wantThinking string
		wantEffort   string
	}{
		{name: "provider default", effort: providers.ReasoningEffortAuto},
		{name: "disabled", effort: providers.ReasoningEffortNone, wantThinking: "disabled"},
		{name: "low", effort: providers.ReasoningEffortLow, wantThinking: "enabled", wantEffort: "low"},
		{
			name:         "medium maps high",
			effort:       providers.ReasoningEffortMedium,
			wantThinking: "enabled",
			wantEffort:   "high",
		},
		{name: "high", effort: providers.ReasoningEffortHigh, wantThinking: "enabled", wantEffort: "high"},
		{name: "xhigh maps high", effort: providers.ReasoningEffortXHigh, wantThinking: "enabled", wantEffort: "high"},
		{name: "max", effort: providers.ReasoningEffortMax, wantThinking: "enabled", wantEffort: "max"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			serverURL, requestBody := deepSeekCompletionServer(t, `{
				"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
				"model":"deepseek-v4-pro","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]
			}`)
			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL(serverURL),
			)
			require.NoError(t, err)

			_, err = provider.Completion(t.Context(), providers.CompletionParams{
				Model: "deepseek-v4-pro",
				Messages: []providers.Message{
					{Role: providers.RoleUser, Content: "first"},
					{
						Role:      providers.RoleAssistant,
						Content:   "answer",
						Reasoning: &providers.Reasoning{Content: "reasoning"},
					},
					{Role: providers.RoleUser, Content: "next"},
				},
				ReasoningEffort: test.effort,
				Tools: []providers.Tool{{
					Type: "function",
					Function: providers.Function{
						Name:       "lookup",
						Parameters: map[string]any{"type": "object"},
					},
				}},
				User: "account_42",
			})
			require.NoError(t, err)

			body := <-requestBody
			require.Equal(t, "account_42", body["user_id"])
			require.NotContains(t, body, "user")
			require.NotContains(t, body, "parallel_tool_calls")
			require.NotContains(t, body, "seed")
			if test.wantThinking == "" {
				require.NotContains(t, body, "thinking")
			} else {
				require.Equal(t, map[string]any{"type": test.wantThinking}, body["thinking"])
			}
			if test.wantEffort == "" {
				require.NotContains(t, body, "reasoning_effort")
			} else {
				require.Equal(t, test.wantEffort, body["reasoning_effort"])
			}

			messages, ok := body["messages"].([]any)
			require.True(t, ok)
			assistant, ok := messages[1].(map[string]any)
			require.True(t, ok)
			require.Equal(t, "reasoning", assistant["reasoning_content"])
		})
	}
}

func TestCompletionOmitsReasoningReplayWithoutTools(t *testing.T) {
	t.Parallel()

	serverURL, requestBody := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
		"model":"deepseek-v4-pro","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]
	}`)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	_, err = provider.Completion(t.Context(), providers.CompletionParams{
		Model: "deepseek-v4-pro",
		Messages: []providers.Message{{
			Role:      providers.RoleAssistant,
			Content:   "answer",
			Reasoning: &providers.Reasoning{Content: "ignored reasoning"},
		}},
	})
	require.NoError(t, err)

	messages, ok := (<-requestBody)["messages"].([]any)
	require.True(t, ok)
	assistant, ok := messages[0].(map[string]any)
	require.True(t, ok)
	require.NotContains(t, assistant, "reasoning_content")
}

func TestCompletionStreamMapsCurrentThinkingRequest(t *testing.T) {
	t.Parallel()

	serverURL, capturedBody := testutil.FakeStreamingServer(t)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	chunks, errs := provider.CompletionStream(t.Context(), providers.CompletionParams{
		Model:           "deepseek-v4-flash",
		Messages:        testutil.SimpleMessages(),
		ReasoningEffort: providers.ReasoningEffortMax,
		User:            "account_42",
	})
	for range chunks {
	}
	require.NoError(t, <-errs)

	body := capturedBody()
	require.Equal(t, map[string]any{"type": "enabled"}, body["thinking"])
	require.Equal(t, "max", body["reasoning_effort"])
	require.Equal(t, "account_42", body["user_id"])
	require.NotContains(t, body, "user")
}

func TestCompletionRejectsUnsupportedParamsBeforeTransport(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		params providers.CompletionParams
		want   error
	}{
		{
			name: "unknown reasoning effort",
			params: providers.CompletionParams{
				ReasoningEffort: providers.ReasoningEffort("unsupported"),
			},
			want: llmerrors.ErrInvalidRequest,
		},
		{
			name:   "parallel tool calls",
			params: providers.CompletionParams{ParallelToolCalls: new(false)},
			want:   llmerrors.ErrUnsupportedParam,
		},
		{
			name:   "seed",
			params: providers.CompletionParams{Seed: new(7)},
			want:   llmerrors.ErrUnsupportedParam,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				requests.Add(1)
			}))
			t.Cleanup(server.Close)
			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL(server.URL),
			)
			require.NoError(t, err)

			test.params.Model = "deepseek-v4-pro"
			test.params.Messages = testutil.SimpleMessages()
			_, err = provider.Completion(t.Context(), test.params)
			require.ErrorIs(t, err, test.want)

			chunks, errs := provider.CompletionStream(t.Context(), test.params)
			for range chunks {
			}
			require.ErrorIs(t, <-errs, test.want)
			require.Zero(t, requests.Load())
		})
	}
}
