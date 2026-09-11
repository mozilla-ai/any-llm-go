package openai_test

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/providers"
	anyopenai "github.com/mozilla-ai/any-llm-go/providers/openai"
)

func TestCompatibleChatCompletionRequestTransformRejectsBeforeHTTP(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name string
		call func(*testing.T, *anyopenai.CompatibleProvider) error
	}{
		{
			name: "completion",
			call: func(t *testing.T, provider *anyopenai.CompatibleProvider) error {
				t.Helper()

				_, err := provider.Completion(t.Context(), callbackCompletionParams(false))

				return err
			},
		},
		{
			name: "completion stream",
			call: func(t *testing.T, provider *anyopenai.CompatibleProvider) error {
				t.Helper()

				chunks, errs := provider.CompletionStream(t.Context(), callbackCompletionParams(true))
				require.Empty(t, collectCallbackChunks(chunks))

				return <-errs
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			var requested atomic.Bool

			server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				requested.Store(true)
			}))
			t.Cleanup(server.Close)

			transformErr := errors.New("request transform rejected asymmetric-model")
			provider := newCallbackCompatibleProvider(t, server.URL, anyopenai.CompatibleConfig{
				ChatCompletionRequestTransform: func(
					params providers.CompletionParams,
					request *openaisdk.ChatCompletionNewParams,
				) error {
					if params.Model != "callback-model-17" || request.Model != "callback-model-17" {
						return errors.New("request transform received unexpected values")
					}

					return transformErr
				},
			})

			err := test.call(t, provider)
			require.Same(t, transformErr, err)
			require.False(t, requested.Load())
		})
	}
}

func TestCompatibleChatCompletionResponseTransformMutatesResult(t *testing.T) {
	t.Parallel()

	server := callbackCompletionServer(t)
	provider := newCallbackCompatibleProvider(t, server.URL, anyopenai.CompatibleConfig{
		ChatCompletionResponseTransform: func(source *openaisdk.ChatCompletion, result *providers.ChatCompletion) error {
			result.ID = source.ID + "-adapted"
			result.Model = "foundation-" + source.Model
			result.Choices[0].Message.Content = source.Choices[0].Message.Content + " / transformed"

			return nil
		},
	})

	response, err := provider.Completion(t.Context(), callbackCompletionParams(false))
	require.NoError(t, err)
	require.Equal(t, "response-id-29-adapted", response.ID)
	require.Equal(t, "foundation-upstream-model-31", response.Model)
	require.Equal(t, "violet response / transformed", response.Choices[0].Message.Content)
}

func TestCompatibleChatCompletionResponseTransformPropagatesExactError(t *testing.T) {
	t.Parallel()

	server := callbackCompletionServer(t)
	transformErr := errors.New("response transform sentinel 37")
	provider := newCallbackCompatibleProvider(t, server.URL, anyopenai.CompatibleConfig{
		ChatCompletionResponseTransform: func(*openaisdk.ChatCompletion, *providers.ChatCompletion) error {
			return transformErr
		},
	})

	response, err := provider.Completion(t.Context(), callbackCompletionParams(false))
	require.Nil(t, response)
	require.Same(t, transformErr, err)
}

func TestCompatibleChatCompletionChunkTransformMutatesResults(t *testing.T) {
	t.Parallel()

	server := callbackStreamServer(t)
	provider := newCallbackCompatibleProvider(t, server.URL, anyopenai.CompatibleConfig{
		ChatCompletionChunkTransform: func(
			source *openaisdk.ChatCompletionChunk,
			result *providers.ChatCompletionChunk,
		) error {
			result.ID = source.ID + "-adapted"
			result.Model = "stream-" + source.Model
			result.Choices[0].Delta.Content = source.Choices[0].Delta.Content + "!"

			return nil
		},
	})

	chunks, errs := provider.CompletionStream(t.Context(), callbackCompletionParams(true))
	got := collectCallbackChunks(chunks)

	require.NoError(t, <-errs)
	require.Equal(t, []providers.ChatCompletionChunk{
		{
			ID:      "chunk-id-41-adapted",
			Object:  "chat.completion.chunk",
			Created: 43,
			Model:   "stream-upstream-stream-47",
			Choices: []providers.ChunkChoice{
				{
					Index: 0,
					Delta: providers.ChunkDelta{Content: "amber!", Role: providers.RoleAssistant},
				},
			},
		},
		{
			ID:      "chunk-id-53-adapted",
			Object:  "chat.completion.chunk",
			Created: 59,
			Model:   "stream-upstream-stream-61",
			Choices: []providers.ChunkChoice{
				{Index: 2, Delta: providers.ChunkDelta{Content: "teal!"}, FinishReason: "stop"},
			},
		},
	}, got)
}

func TestCompatibleChatCompletionChunkTransformPropagatesExactErrorAndTerminates(t *testing.T) {
	t.Parallel()

	server := callbackStreamServer(t)
	transformErr := errors.New("chunk transform sentinel 67")

	var calls atomic.Int64

	provider := newCallbackCompatibleProvider(t, server.URL, anyopenai.CompatibleConfig{
		ChatCompletionChunkTransform: func(*openaisdk.ChatCompletionChunk, *providers.ChatCompletionChunk) error {
			calls.Add(1)

			return transformErr
		},
	})

	chunks, errs := provider.CompletionStream(t.Context(), callbackCompletionParams(true))
	require.Empty(t, collectCallbackChunks(chunks))
	require.Same(t, transformErr, <-errs)

	_, chunksOpen := <-chunks
	_, errsOpen := <-errs

	require.False(t, chunksOpen)
	require.False(t, errsOpen)
	require.Equal(t, int64(1), calls.Load())
}

func newCallbackCompatibleProvider(
	t *testing.T,
	baseURL string,
	callbacks anyopenai.CompatibleConfig,
) *anyopenai.CompatibleProvider {
	t.Helper()

	callbacks.Name = "callback-contract-test"
	callbacks.DefaultAPIKey = "callback-key"
	callbacks.DefaultBaseURL = baseURL
	provider, err := anyopenai.NewCompatible(callbacks)
	require.NoError(t, err)

	return provider
}

func callbackCompletionParams(stream bool) providers.CompletionParams {
	return providers.CompletionParams{
		Model:    "callback-model-17",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "request-cyan-23"}},
		Stream:   stream,
	}
}

func callbackCompletionServer(t *testing.T) *httptest.Server {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(
			w,
			`{"id":"response-id-29","object":"chat.completion","created":27,"model":"upstream-model-31",`+
				`"choices":[{"index":0,"message":{"role":"assistant","content":"violet response"},`+
				`"finish_reason":"stop"}]}`,
		)
	}))
	t.Cleanup(server.Close)

	return server
}

func callbackStreamServer(t *testing.T) *httptest.Server {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")

		for _, payload := range []string{
			`{"id":"chunk-id-41","object":"chat.completion.chunk","created":43,"model":"upstream-stream-47",` +
				`"choices":[{"index":0,"delta":{"role":"assistant","content":"amber"},"finish_reason":null}]}`,
			`{"id":"chunk-id-53","object":"chat.completion.chunk","created":59,"model":"upstream-stream-61",` +
				`"choices":[{"index":2,"delta":{"content":"teal"},"finish_reason":"stop"}]}`,
		} {
			_, _ = fmt.Fprintf(writer, "data: %s\n\n", payload)
		}

		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)

	return server
}

func collectCallbackChunks(
	chunks <-chan providers.ChatCompletionChunk,
) []providers.ChatCompletionChunk {
	var collected []providers.ChatCompletionChunk
	for chunk := range chunks {
		collected = append(collected, chunk)
	}

	return collected
}
