package azureopenai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	anyerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestAzureV1CoreWireContract(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "ambient-openai-key")

	srv := httptest.NewTLSServer(http.HandlerFunc(func(responseWriter http.ResponseWriter, request *http.Request) {
		assert.Equal(t, "wire-key", request.Header.Get("Api-Key"))
		assert.NotContains(t, request.Header, "Authorization")
		assert.Equal(t, "trace-value", request.Header.Get("X-Test-Trace"))
		assert.Empty(t, request.URL.RawQuery)

		responseWriter.Header().Set("Content-Type", "application/json")

		switch request.URL.Path {
		case "/openai/v1/chat/completions":
			body, err := io.ReadAll(request.Body)
			assert.NoError(t, err)
			assert.JSONEq(t, `{
				"messages": [
					{"role": "developer", "content": "Follow the schema.", "name": "policy"}
				],
				"model": "deployment",
				"reasoning_effort": "none"
			}`, string(body))

			_, _ = io.WriteString(
				responseWriter,
				`{"id":"chat","object":"chat.completion","created":1,"model":"deployment","choices":[]}`,
			)
		case "/openai/v1/embeddings":
			_, _ = io.WriteString(
				responseWriter,
				`{"object":"list","data":[],"model":"embedding-deployment",`+
					`"usage":{"prompt_tokens":0,"total_tokens":0}}`,
			)
		case "/openai/v1/models":
			_, _ = io.WriteString(
				responseWriter,
				`{"object":"list","data":`+
					`[{"id":"deployment","object":"model","created":7,"owned_by":"azure",`+
					`"future_metadata":{"region":"unknown"}}]}`,
			)
		default:
			http.NotFound(responseWriter, request)
		}
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithAPIKey("wire-key"),
		config.WithBaseURL(srv.URL),
		config.WithHTTPClient(srv.Client()),
		config.WithHeader("X-Test-Trace", "trace-value"),
	)
	require.NoError(t, err)

	_, err = provider.Completion(t.Context(), providers.CompletionParams{
		Model: "deployment",
		Messages: []providers.Message{
			{Role: providers.RoleDeveloper, Content: "Follow the schema.", Name: "policy"},
		},
		ReasoningEffort: providers.ReasoningEffortNone,
	})
	require.NoError(t, err)

	_, err = provider.Embedding(t.Context(), providers.EmbeddingParams{Model: "embedding-deployment", Input: "hello"})
	require.NoError(t, err)

	models, err := provider.ListModels(t.Context())
	require.NoError(t, err)
	require.Len(t, models.Data, 1)
	require.EqualValues(t, 7, models.Data[0].Created)
}

func TestAzureV1RejectsCredentialRedirect(t *testing.T) {
	t.Parallel()

	var redirectedRequests atomic.Int32

	destination := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		redirectedRequests.Add(1)
	}))
	t.Cleanup(destination.Close)

	endpoint := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "azure-key", r.Header.Get("Api-Key"))
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(endpoint.Close)

	provider, err := New(
		config.WithAPIKey("azure-key"),
		config.WithBaseURL(endpoint.URL),
		config.WithHTTPClient(endpoint.Client()),
	)
	require.NoError(t, err)

	_, err = provider.ListModels(t.Context())
	require.Error(t, err)
	require.Zero(t, redirectedRequests.Load())
}

func TestAzureV1MapsServiceError(t *testing.T) {
	t.Parallel()

	endpoint := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "azure-key", r.Header.Get("Api-Key"))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"code":"invalid_api_key","message":"invalid credential"}}`)
	}))
	t.Cleanup(endpoint.Close)

	provider, err := New(
		config.WithAPIKey("azure-key"),
		config.WithBaseURL(endpoint.URL),
		config.WithHTTPClient(endpoint.Client()),
	)
	require.NoError(t, err)

	_, err = provider.ListModels(t.Context())
	require.ErrorIs(t, err, anyerrors.ErrAuthentication)
}

func TestAzureV1StreamWireContract(t *testing.T) {
	t.Parallel()

	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/openai/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "wire-key", r.Header.Get("Api-Key"))
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(
			w,
			"data: {\"id\":\"chat\",\"object\":\"chat.completion.chunk\",\"created\":1,"+
				"\"model\":\"deployment\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n"+
				"data: [DONE]\n\n",
		)
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithAPIKey("wire-key"),
		config.WithBaseURL(srv.URL),
		config.WithHTTPClient(srv.Client()),
	)
	require.NoError(t, err)

	chunks, errs := provider.CompletionStream(t.Context(), providers.CompletionParams{
		Model:    "deployment",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	})
	chunk := <-chunks
	require.Equal(t, "hi", chunk.Choices[0].Delta.Content)
	require.NoError(t, <-errs)
}
