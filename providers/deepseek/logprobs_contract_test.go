package deepseek

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestCompletionPreservesReasoningLogprobs(t *testing.T) {
	t.Parallel()

	serverURL, requestBody := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"deepseek-v4-pro",
		"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"A"},
		"logprobs":{"content":[{"token":"A","bytes":[65],"logprob":-0.1,"top_logprobs":[]}],
		"reasoning_content":[{"token":"think","bytes":null,"logprob":-0.2,"top_logprobs":[
			{"token":"plan","bytes":[112,108,97,110],"logprob":-9999.0}
		]}],"future_extension":{"enabled":true}}}]
	}`)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(serverURL))
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:       "deepseek-v4-pro",
		Messages:    []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
		Logprobs:    new(true),
		TopLogprobs: new(1),
	})
	require.NoError(t, err)
	body := <-requestBody
	require.Equal(t, true, body["logprobs"])
	require.Equal(t, float64(1), body["top_logprobs"])
	require.Equal(t, []providers.ChatCompletionTokenLogprob{{
		Token: "think", Logprob: -0.2,
		TopLogprobs: []providers.ChatCompletionTopLogprob{{
			Token: "plan", Bytes: []int{112, 108, 97, 110}, Logprob: -9999,
		}},
	}}, completion.Choices[0].Logprobs.ReasoningContent)
}

func TestCompletionStreamPreservesReasoningLogprobs(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := fmt.Fprint(
			w,
			`data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"finish_reason":null,"delta":{"reasoning_content":"think"},"logprobs":{"content":null,"reasoning_content":[{"token":"think","bytes":null,"logprob":-0.2,"top_logprobs":[]}]}}]}`+"\n\ndata: [DONE]\n\n",
		)
		if err != nil {
			t.Errorf("writing DeepSeek logprobs stream: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	chunks, errs := provider.CompletionStream(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
		Logprobs: new(true),
	})
	chunk := <-chunks
	require.NoError(t, <-errs)
	require.Equal(t, []providers.ChatCompletionTokenLogprob{{
		Token: "think", Logprob: -0.2, TopLogprobs: []providers.ChatCompletionTopLogprob{},
	}}, chunk.Choices[0].Logprobs.ReasoningContent)
}

func TestCompletionRejectsMalformedReasoningLogprobs(t *testing.T) {
	t.Parallel()

	serverURL, _ := deepSeekCompletionServer(t, `{
		"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"deepseek-v4-pro",
		"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"A"},
		"logprobs":{"content":null,"reasoning_content":{"token":"think"}}}]
	}`)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(serverURL))
	require.NoError(t, err)

	_, err = provider.Completion(t.Context(), providers.CompletionParams{
		Model:    "deepseek-v4-pro",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
		Logprobs: new(true),
	})
	require.ErrorContains(t, err, "reasoning_content")
}
