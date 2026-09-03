package deepseek

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	llmerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestCompletionPreservesReasoningResponse(t *testing.T) {
	t.Parallel()

	serverURL, _ := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
		"model":"deepseek-v4-pro","choices":[{"index":0,"message":{
			"role":"assistant","content":"answer","reasoning_content":"","future_field":true
		},"finish_reason":"insufficient_system_resource"}]
	}`)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)
	require.Equal(t, "answer", completion.Choices[0].Message.Content)
	require.Equal(t, "insufficient_system_resource", completion.Choices[0].FinishReason)
	require.NotNil(t, completion.Choices[0].Message.Reasoning)
	require.Empty(t, completion.Choices[0].Message.Reasoning.Content)
}

func TestCompletionPreservesNullableContent(t *testing.T) {
	t.Parallel()

	serverURL, _ := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
		"model":"deepseek-v4-pro","choices":[{"index":0,"message":{
			"role":"assistant","content":null,"reasoning_content":"reasoning"
		},"finish_reason":"stop"}]
	}`)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)
	require.Nil(t, completion.Choices[0].Message.Content)
}

func TestCompletionPreservesCacheUsage(t *testing.T) {
	t.Parallel()

	serverURL, _ := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
		"model":"deepseek-v4-pro","choices":[{"index":0,"message":{
			"role":"assistant","content":"answer"
		},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":5,
		"total_tokens":13,"prompt_cache_hit_tokens":3,"prompt_cache_miss_tokens":5}
	}`)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)
	require.Equal(t, &providers.Usage{
		PromptTokens:     8,
		CompletionTokens: 5,
		TotalTokens:      13,
		CachedTokens:     3,
	}, completion.Usage)
}

func TestCompletionPreservesPresentZeroCacheUsage(t *testing.T) {
	t.Parallel()

	serverURL, _ := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
		"model":"deepseek-v4-pro","choices":[{"index":0,"message":{
			"role":"assistant","content":"answer"
		},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,
		"total_tokens":0,"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":0}
	}`)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)
	require.Equal(t, &providers.Usage{}, completion.Usage)
}

func TestCompletionRejectsMalformedCacheUsage(t *testing.T) {
	t.Parallel()

	serverURL, _ := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
		"model":"deepseek-v4-pro","choices":[{"index":0,"message":{
			"role":"assistant","content":"answer"
		},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":5,
		"total_tokens":13,"prompt_cache_hit_tokens":"3","prompt_cache_miss_tokens":5}
	}`)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	_, err = provider.Completion(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: testutil.SimpleMessages(),
	})
	require.ErrorContains(t, err, "decoding DeepSeek usage")
}

func TestCompletionRejectsMalformedContent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		message string
		want    string
	}{
		{name: "missing answer", message: `"reasoning_content":"reasoning"`, want: "missing required field"},
		{name: "invalid answer", message: `"content":{}`, want: "cannot unmarshal object"},
		{
			name:    "invalid reasoning",
			message: `"content":"answer","reasoning_content":{}`,
			want:    "cannot unmarshal object",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			serverURL, _ := deepSeekCompletionServer(t, fmt.Sprintf(`{
				"id":"chatcmpl-test","object":"chat.completion","created":1700000000,
				"model":"deepseek-v4-pro","choices":[{"index":0,"message":{
					"role":"assistant",%s
				},"finish_reason":"stop"}]
			}`, test.message))
			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL(serverURL),
			)
			require.NoError(t, err)

			_, err = provider.Completion(t.Context(), providers.CompletionParams{
				Model:    "deepseek-v4-pro",
				Messages: testutil.SimpleMessages(),
			})
			require.ErrorContains(t, err, test.want)
		})
	}
}

func TestCompletionStreamPreservesReasoningAndTerminalUsage(t *testing.T) {
	t.Parallel()

	events := []string{
		`{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"reasoning"},"finish_reason":null}],"usage":null}`,
		`{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}],"usage":null}`,
		`{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":5,"total_tokens":13,"prompt_cache_hit_tokens":3,"prompt_cache_miss_tokens":5}}`,
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := fmt.Fprint(w, ": keep-alive\n\n")
		if err != nil {
			t.Errorf("writing DeepSeek keep-alive: %v", err)
			return
		}
		for _, event := range events {
			_, err = fmt.Fprintf(w, "data: %s\n\n", event)
			if err != nil {
				t.Errorf("writing DeepSeek stream event: %v", err)
				return
			}
		}
		_, err = fmt.Fprint(w, "data: [DONE]\n\n")
		if err != nil {
			t.Errorf("writing DeepSeek stream terminator: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	chunkChannel, errChannel := provider.CompletionStream(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: testutil.SimpleMessages(),
	})
	chunks := make([]providers.ChatCompletionChunk, 0, len(events))
	for chunk := range chunkChannel {
		chunks = append(chunks, chunk)
	}
	require.NoError(t, <-errChannel)
	require.Len(t, chunks, len(events))
	require.NotNil(t, chunks[0].Choices[0].Delta.Reasoning)
	require.Equal(t, "reasoning", chunks[0].Choices[0].Delta.Reasoning.Content)
	require.Equal(t, "answer", chunks[1].Choices[0].Delta.Content)
	require.Equal(t, "stop", chunks[2].Choices[0].FinishReason)
	require.Equal(t, &providers.Usage{
		PromptTokens:     8,
		CompletionTokens: 5,
		TotalTokens:      13,
		CachedTokens:     3,
	}, chunks[2].Usage)
}

func TestCompletionStreamRejectsMalformedCacheUsage(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := fmt.Fprint(w, `data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":5,"total_tokens":13,"prompt_cache_hit_tokens":"3","prompt_cache_miss_tokens":5}}

data: [DONE]

`)
		if err != nil {
			t.Errorf("writing DeepSeek stream: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	chunks, errs := provider.CompletionStream(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: testutil.SimpleMessages(),
	})
	for range chunks {
	}
	require.ErrorContains(t, <-errs, "decoding DeepSeek usage")
}

func TestCompletionMapsDocumentedDeepSeekErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		status int
		want   error
	}{
		{status: http.StatusBadRequest, want: llmerrors.ErrInvalidRequest},
		{status: http.StatusUnauthorized, want: llmerrors.ErrAuthentication},
		{status: http.StatusPaymentRequired, want: llmerrors.ErrInsufficientFunds},
		{status: http.StatusUnprocessableEntity, want: llmerrors.ErrInvalidRequest},
		{status: http.StatusTooManyRequests, want: llmerrors.ErrRateLimit},
		{status: http.StatusInternalServerError, want: llmerrors.ErrProvider},
		{status: http.StatusServiceUnavailable, want: llmerrors.ErrProvider},
		{status: http.StatusNotFound, want: llmerrors.ErrProvider},
	}
	for _, test := range tests {
		t.Run(http.StatusText(test.status), func(t *testing.T) {
			t.Parallel()

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(test.status)
				_, err := fmt.Fprint(w, `{"error":{"message":"request failed","type":"invalid_request_error"}}`)
				if err != nil {
					t.Errorf("writing DeepSeek error: %v", err)
				}
			}))
			t.Cleanup(server.Close)
			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL(server.URL),
			)
			require.NoError(t, err)

			_, err = provider.Completion(t.Context(), providers.CompletionParams{
				Model:    "deepseek-v4-pro",
				Messages: testutil.SimpleMessages(),
			})
			require.Error(t, err)
			require.ErrorIs(t, err, test.want)
		})
	}
}
