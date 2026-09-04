package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/require"

	anyerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const responseFixture = `{"id":"resp_123","object":"response","created_at":1,"model":"gpt-5.6-sol","status":"completed","output":[]}`

func TestCreateResponsePreservesSDKParamsAndCapability(t *testing.T) {
	t.Parallel()

	unsupported, err := NewCompatible(CompatibleConfig{Name: "unsupported"})
	require.NoError(t, err)
	got, err := unsupported.CreateResponse(t.Context(), responses.ResponseNewParams{})
	require.Nil(t, got)
	var unsupportedErr *anyerrors.UnsupportedOperationError
	require.ErrorAs(t, err, &unsupportedErr)
	events, errs := unsupported.StreamResponse(t.Context(), responses.ResponseNewParams{})
	require.Empty(t, events)
	require.ErrorAs(t, <-errs, &unsupportedErr)

	var request map[string]json.RawMessage
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/v1/responses", r.URL.Path)
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		w.Header().Set("Content-Type", "application/json")
		_, writeErr := io.WriteString(w, responseFixture)
		require.NoError(t, writeErr)
	}))
	t.Cleanup(server.Close)

	provider, err := NewCompatible(CompatibleConfig{
		Capabilities:   providers.Capabilities{Responses: true},
		DefaultAPIKey:  "test-key",
		DefaultBaseURL: server.URL + "/v1",
		Name:           "test-provider",
	})
	require.NoError(t, err)

	resp, err := provider.CreateResponse(t.Context(), responses.ResponseNewParams{
		Instructions: openaisdk.String("native"),
		Temperature:  openaisdk.Float(0.25),
	})
	require.NoError(t, err)
	require.Equal(t, "resp_123", resp.ID)
	require.Contains(t, request, "instructions")
	require.Contains(t, request, "temperature")
	require.NotContains(t, request, "input")
	require.NotContains(t, request, "model")
}

func TestStreamResponsePreservesEventsAndTerminalStates(t *testing.T) {
	t.Parallel()

	for _, terminal := range []string{"response.completed", "response.incomplete", "response.failed"} {
		t.Run(terminal, func(t *testing.T) {
			t.Parallel()

			var request map[string]json.RawMessage
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
				w.Header().Set("Content-Type", "text/event-stream")
				_, err := io.WriteString(w,
					"data: {\"type\":\"future.event\",\"sequence_number\":1,\"future_field\":true}\n\n"+
						"data: {\"type\":\""+terminal+"\",\"sequence_number\":2,\"response\":"+responseFixture+"}\n\n",
				)
				require.NoError(t, err)
			}))
			t.Cleanup(server.Close)

			provider, err := NewCompatible(CompatibleConfig{
				Capabilities:   providers.Capabilities{ResponsesStreaming: true},
				DefaultAPIKey:  "test-key",
				DefaultBaseURL: server.URL + "/v1",
				Name:           "test-provider",
			})
			require.NoError(t, err)

			events, errs := provider.StreamResponse(t.Context(), responses.ResponseNewParams{})
			var types []string
			for event := range events {
				types = append(types, event.Type)
				if event.Type == "future.event" {
					require.Contains(t, event.RawJSON(), `"future_field":true`)
				}
			}
			require.NoError(t, <-errs)
			require.Equal(t, []string{"future.event", terminal}, types)
			require.JSONEq(t, `true`, string(request["stream"]))
		})
	}
}

func TestStreamResponseReportsTruncationAndCancellation(t *testing.T) {
	t.Parallel()

	t.Run("clean EOF without terminal event", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, err := io.WriteString(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n")
			require.NoError(t, err)
		}))
		t.Cleanup(server.Close)

		provider, err := NewCompatible(CompatibleConfig{
			Capabilities:   providers.Capabilities{ResponsesStreaming: true},
			DefaultAPIKey:  "test-key",
			DefaultBaseURL: server.URL + "/v1",
			Name:           "test-provider",
		})
		require.NoError(t, err)

		events, errs := provider.StreamResponse(t.Context(), responses.ResponseNewParams{})
		for range events {
		}
		require.ErrorIs(t, <-errs, io.ErrUnexpectedEOF)
	})

	t.Run("caller cancellation", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			<-r.Context().Done()
		}))
		t.Cleanup(server.Close)

		provider, err := NewCompatible(CompatibleConfig{
			Capabilities:   providers.Capabilities{ResponsesStreaming: true},
			DefaultAPIKey:  "test-key",
			DefaultBaseURL: server.URL + "/v1",
			Name:           "test-provider",
		})
		require.NoError(t, err)

		ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
		defer cancel()
		events, errs := provider.StreamResponse(ctx, responses.ResponseNewParams{})
		for range events {
		}
		require.ErrorIs(t, <-errs, context.DeadlineExceeded)
	})
}

func TestResponseTransportMapsAPIErrors(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, err := io.WriteString(
			w,
			`{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limit_exceeded"}}`,
		)
		require.NoError(t, err)
	}))
	t.Cleanup(server.Close)

	provider, err := NewCompatible(CompatibleConfig{
		Capabilities: providers.Capabilities{
			Responses:          true,
			ResponsesStreaming: true,
		},
		DefaultAPIKey:  "test-key",
		DefaultBaseURL: server.URL + "/v1",
		Name:           "test-provider",
	})
	require.NoError(t, err)

	resp, err := provider.CreateResponse(t.Context(), responses.ResponseNewParams{})
	require.Nil(t, resp)
	require.ErrorIs(t, err, anyerrors.ErrRateLimit)

	events, errs := provider.StreamResponse(t.Context(), responses.ResponseNewParams{})
	for range events {
	}
	require.ErrorIs(t, <-errs, anyerrors.ErrRateLimit)
}

func TestCreateResponseHonorsCallerTimeout(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(release) })

	provider, err := NewCompatible(CompatibleConfig{
		Capabilities:   providers.Capabilities{Responses: true},
		DefaultAPIKey:  "test-key",
		DefaultBaseURL: server.URL + "/v1",
		Name:           "test-provider",
	})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	result, err := provider.CreateResponse(ctx, responses.ResponseNewParams{})
	require.Nil(t, result)
	require.ErrorIs(t, err, context.DeadlineExceeded)
}
