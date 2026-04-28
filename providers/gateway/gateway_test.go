package gateway

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Object type constants for test assertions.
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
)

func TestNew(t *testing.T) {
	// Note: Not using t.Parallel() here because child tests use t.Setenv.

	t.Run("returns error when base URL is missing", func(t *testing.T) {
		t.Setenv(envAPIBase, "")

		provider, err := New()
		require.Nil(t, provider)
		require.Error(t, err)
		require.Contains(t, err.Error(), "gateway base URL is required")
	})

	t.Run("creates provider with base URL from env", func(t *testing.T) {
		t.Setenv(envAPIBase, "http://localhost:8000/v1")
		t.Setenv(envPlatformToken, "")
		t.Setenv(envAPIKey, "")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, providerName, provider.Name())
	})

	t.Run("creates provider with explicit base URL", func(t *testing.T) {
		t.Setenv(envAPIBase, "")
		t.Setenv(envPlatformToken, "")

		provider, err := New(config.WithBaseURL("http://localhost:8000/v1"))
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("creates platform mode provider with explicit API key", func(t *testing.T) {
		t.Setenv(envAPIBase, "")
		t.Setenv(envPlatformToken, "")

		provider, err := New(
			config.WithBaseURL("http://localhost:8000/v1"),
			config.WithAPIKey("tk_test_token"),
			WithPlatformMode(),
		)
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.True(t, provider.platformMode)
	})

	t.Run("auto-detects platform mode from env token", func(t *testing.T) {
		t.Setenv(envAPIBase, "http://localhost:8000/v1")
		t.Setenv(envPlatformToken, "tk_auto_detected")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.True(t, provider.platformMode)
	})

	t.Run("returns error when platform mode has no token", func(t *testing.T) {
		t.Setenv(envAPIBase, "http://localhost:8000/v1")
		t.Setenv(envPlatformToken, "")

		provider, err := New(WithPlatformMode())
		require.Nil(t, provider)
		require.Error(t, err)
		require.Contains(t, err.Error(), "platform mode requires a token")
	})

	t.Run("creates non-platform provider with gateway key from env", func(t *testing.T) {
		t.Setenv(envAPIBase, "http://localhost:8000/v1")
		t.Setenv(envAPIKey, "gw_test_key")
		t.Setenv(envPlatformToken, "")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.False(t, provider.platformMode)
	})

	t.Run("creates non-platform provider with WithGatewayKey", func(t *testing.T) {
		t.Setenv(envAPIBase, "")
		t.Setenv(envPlatformToken, "")

		provider, err := New(
			config.WithBaseURL("http://localhost:8000/v1"),
			WithGatewayKey("gw_explicit_key"),
		)
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.False(t, provider.platformMode)
	})

	t.Run("forwards custom timeout to underlying provider", func(t *testing.T) {
		t.Setenv(envAPIBase, "")
		t.Setenv(envPlatformToken, "")

		provider, err := New(
			config.WithBaseURL("http://localhost:8000/v1"),
			config.WithTimeout(30*time.Second),
		)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("forwards custom HTTP client to platform mode provider", func(t *testing.T) {
		t.Setenv(envAPIBase, "")
		t.Setenv(envPlatformToken, "")

		var capturedHeaders http.Header
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capturedHeaders = r.Header.Clone()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(mockCompletionResponse("test")))
		}))
		t.Cleanup(srv.Close)

		customClient := &http.Client{Timeout: 5 * time.Second}
		provider, err := New(
			config.WithBaseURL(srv.URL),
			config.WithAPIKey("tk_test"),
			config.WithHTTPClient(customClient),
			WithPlatformMode(),
		)
		require.NoError(t, err)

		ctx := context.Background()
		_, err = provider.Completion(ctx, mockCompletionParams())
		require.NoError(t, err)

		// Platform mode: API key sent as standard Bearer auth.
		require.Equal(t, bearerPrefix+"tk_test", capturedHeaders.Get("Authorization"))
	})

	t.Run("forwards custom HTTP client transport in non-platform mode", func(t *testing.T) {
		t.Setenv(envAPIBase, "")
		t.Setenv(envPlatformToken, "")

		var capturedHeaders http.Header
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			capturedHeaders = r.Header.Clone()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(mockCompletionResponse("test")))
		}))
		t.Cleanup(srv.Close)

		customTransport := &mockRoundTripper{base: http.DefaultTransport}
		customClient := &http.Client{Transport: customTransport}
		provider, err := New(
			config.WithBaseURL(srv.URL),
			config.WithHTTPClient(customClient),
			WithGatewayKey("gw_key"),
		)
		require.NoError(t, err)

		ctx := context.Background()
		_, err = provider.Completion(ctx, mockCompletionParams())
		require.NoError(t, err)

		// Non-platform mode: gateway key sent via custom header.
		require.Equal(t, bearerPrefix+"gw_key", capturedHeaders.Get(apiKeyHeaderName))
		// Custom transport should have been used (wrapped by headerTransport).
		require.True(t, customTransport.called, "custom transport should be used as base")
	})
}

func TestProviderName(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithBaseURL("http://localhost:8000/v1"))
	require.NoError(t, err)
	require.Equal(t, providerName, provider.Name())
}

func TestCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithBaseURL("http://localhost:8000/v1"))
	require.NoError(t, err)

	caps := provider.Capabilities()

	require.True(t, caps.Completion)
	require.True(t, caps.CompletionImage)
	require.True(t, caps.CompletionPDF)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.True(t, caps.Embedding)
	require.True(t, caps.ListModels)
	require.True(t, caps.Moderation)
	require.True(t, caps.Rerank)
}

func TestPlatformModeDetection(t *testing.T) {
	// Note: Not using t.Parallel() because subtests use t.Setenv.

	tests := []struct {
		name             string
		envPlatformToken string
		envAPIKey        string
		apiKey           string
		gatewayKey       string
		wantPlatform     bool
	}{
		{
			name:             "does not auto-detect when API key is explicitly set",
			envPlatformToken: "tk_auto",
			apiKey:           "explicit_key",
			wantPlatform:     false,
		},
		{
			name:             "does not auto-detect when gateway key is explicitly set",
			envPlatformToken: "tk_auto",
			gatewayKey:       "gw_key",
			wantPlatform:     false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(envAPIBase, "http://localhost:8000/v1")
			t.Setenv(envPlatformToken, tc.envPlatformToken)
			t.Setenv(envAPIKey, tc.envAPIKey)

			var opts []config.Option
			if tc.apiKey != "" {
				opts = append(opts, config.WithAPIKey(tc.apiKey))
			}
			if tc.gatewayKey != "" {
				opts = append(opts, WithGatewayKey(tc.gatewayKey))
			}

			provider, err := New(opts...)
			require.NoError(t, err)
			require.Equal(t, tc.wantPlatform, provider.platformMode)
		})
	}
}

// TestExtraValueHandling verifies that the config.Extra mechanism correctly
// ignores wrong-typed values (silent fallthrough) and honours valid ones.
// Assertions are made on the wire to prove the mode actually took effect,
// not just that the provider constructed without error.
func TestExtraValueHandling(t *testing.T) {
	// Note: Not using t.Parallel() because subtests use t.Setenv.

	tests := []struct {
		name              string
		extraKey          string
		extraValue        any
		envAPIKey         string
		apiKey            string
		gatewayKey        string
		wantAPIKeyHeader  string // expected AnyLLM-Key value; empty means the header must not be sent.
		wantAuthorization string // expected Authorization value.
	}{
		{
			name:              "string gateway_key is forwarded as the gateway key header",
			extraKey:          extraKeyGatewayKey,
			extraValue:        "valid_key",
			wantAPIKeyHeader:  bearerPrefix + "valid_key",
			wantAuthorization: bearerPrefix + placeholderAPIKey,
		},
		{
			name:              "int gateway_key is silently ignored",
			extraKey:          extraKeyGatewayKey,
			extraValue:        123,
			wantAPIKeyHeader:  "",
			wantAuthorization: bearerPrefix + placeholderAPIKey,
		},
		{
			name:              "empty-string gateway_key is treated as unset",
			extraKey:          extraKeyGatewayKey,
			extraValue:        "",
			wantAPIKeyHeader:  "",
			wantAuthorization: bearerPrefix + placeholderAPIKey,
		},
		{
			name:              "empty-string gateway_key falls through to GATEWAY_API_KEY env var",
			extraKey:          extraKeyGatewayKey,
			extraValue:        "",
			envAPIKey:         "env_fallback_key",
			wantAPIKeyHeader:  bearerPrefix + "env_fallback_key",
			wantAuthorization: bearerPrefix + placeholderAPIKey,
		},
		{
			name:              "bool platform_mode enables platform-mode Bearer auth",
			extraKey:          extraKeyPlatformMode,
			extraValue:        true,
			apiKey:            "platform_token",
			wantAPIKeyHeader:  "",
			wantAuthorization: bearerPrefix + "platform_token",
		},
		{
			// Passing a string instead of bool must not flip into platform mode.
			// Combining with a gateway key proves the non-platform path was
			// taken: an honoured platform_mode would suppress the gateway
			// header. WithAPIKey is also passed to verify it does NOT leak
			// into Authorization in non-platform mode — the placeholder is
			// used instead.
			name:              "string platform_mode is silently ignored",
			extraKey:          extraKeyPlatformMode,
			extraValue:        "true",
			apiKey:            "platform_token",
			gatewayKey:        "gw_key",
			wantAPIKeyHeader:  bearerPrefix + "gw_key",
			wantAuthorization: bearerPrefix + placeholderAPIKey,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(envAPIBase, "")
			t.Setenv(envAPIKey, tc.envAPIKey)
			t.Setenv(envPlatformToken, "")

			var (
				mu              sync.Mutex
				capturedHeaders http.Header
			)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				capturedHeaders = r.Header.Clone()
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(mockCompletionResponse("ok")))
			}))
			t.Cleanup(srv.Close)

			opts := []config.Option{
				config.WithBaseURL(srv.URL),
				config.WithExtra(tc.extraKey, tc.extraValue),
			}
			if tc.apiKey != "" {
				opts = append(opts, config.WithAPIKey(tc.apiKey))
			}
			if tc.gatewayKey != "" {
				opts = append(opts, WithGatewayKey(tc.gatewayKey))
			}

			provider, err := New(opts...)
			require.NoError(t, err)

			_, err = provider.Completion(context.Background(), mockCompletionParams())
			require.NoError(t, err)

			mu.Lock()
			defer mu.Unlock()

			require.Equal(t, tc.wantAPIKeyHeader, capturedHeaders.Get(apiKeyHeaderName))
			require.Equal(t, tc.wantAuthorization, capturedHeaders.Get("Authorization"))
		})
	}
}

func TestHeaderTransport(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		header     string
		value      string
		wantHeader string
		wantValue  string
	}{
		{
			name:       "injects gateway header",
			header:     apiKeyHeaderName,
			value:      bearerPrefix + "test-key",
			wantHeader: apiKeyHeaderName,
			wantValue:  bearerPrefix + "test-key",
		},
		{
			name:       "overwrites existing header value",
			header:     "Authorization",
			value:      bearerPrefix + "new-token",
			wantHeader: "Authorization",
			wantValue:  bearerPrefix + "new-token",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var capturedHeaders http.Header
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				capturedHeaders = r.Header.Clone()
				w.WriteHeader(http.StatusOK)
			}))
			t.Cleanup(srv.Close)

			transport := &headerTransport{
				base:   http.DefaultTransport,
				header: tc.header,
				value:  tc.value,
			}

			req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, srv.URL, nil)
			require.NoError(t, err)

			resp, err := transport.RoundTrip(req)
			require.NoError(t, err)
			defer func() { _ = resp.Body.Close() }()

			require.Equal(t, tc.wantValue, capturedHeaders.Get(tc.wantHeader))
		})
	}

	t.Run("does not mutate original request", func(t *testing.T) {
		t.Parallel()

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		t.Cleanup(srv.Close)

		transport := &headerTransport{
			base:   http.DefaultTransport,
			header: apiKeyHeaderName,
			value:  bearerPrefix + "key",
		}

		req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, srv.URL, nil)
		require.NoError(t, err)

		resp, err := transport.RoundTrip(req)
		require.NoError(t, err)
		defer func() { _ = resp.Body.Close() }()

		// Original request should not have the injected header.
		require.Empty(t, req.Header.Get(apiKeyHeaderName))
	})
}

func TestNonPlatformModeSendsCustomHeader(t *testing.T) {
	t.Parallel()

	var (
		mu              sync.Mutex
		capturedHeaders http.Header
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		capturedHeaders = r.Header.Clone()
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockCompletionResponse("hello")))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("gw_test_key_123"),
	)
	require.NoError(t, err)

	ctx := context.Background()
	_, err = provider.Completion(ctx, mockCompletionParams())
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()

	require.Equal(t, bearerPrefix+"gw_test_key_123", capturedHeaders.Get(apiKeyHeaderName))
}

func TestPlatformModeSendsBearerAuth(t *testing.T) {
	t.Parallel()

	var (
		mu              sync.Mutex
		capturedHeaders http.Header
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		capturedHeaders = r.Header.Clone()
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockCompletionResponse("hello")))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		config.WithAPIKey("tk_platform_token"),
		WithPlatformMode(),
	)
	require.NoError(t, err)

	ctx := context.Background()
	_, err = provider.Completion(ctx, mockCompletionParams())
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()

	require.Equal(t, bearerPrefix+"tk_platform_token", capturedHeaders.Get("Authorization"))
	require.Empty(t, capturedHeaders.Get(apiKeyHeaderName),
		"platform mode should not send gateway key header")
}

func TestCompletion(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-gw",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "openai:gpt-4o-mini",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Hello from gateway!"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.Completion(ctx, providers.CompletionParams{
		Model:    "openai:gpt-4o-mini",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	})
	require.NoError(t, err)

	require.Equal(t, "chatcmpl-gw", resp.ID)
	require.Equal(t, objectChatCompletion, resp.Object)
	require.Len(t, resp.Choices, 1)
	require.Equal(t, "Hello from gateway!", resp.Choices[0].Message.ContentString())
	require.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
	require.Equal(t, providers.FinishReasonStop, resp.Choices[0].FinishReason)
	require.NotNil(t, resp.Usage)
	require.Equal(t, 15, resp.Usage.TotalTokens)
}

func TestCompletionStream(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		chunk := `{"id":"chatcmpl-gw","object":"chat.completion.chunk","created":1700000000,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`
		done := `{"id":"chatcmpl-gw","object":"chat.completion.chunk","created":1700000000,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`

		_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", done)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	ctx := context.Background()
	chunks, errs := provider.CompletionStream(ctx, providers.CompletionParams{
		Model:    "openai:gpt-4o-mini",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	})

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		require.Equal(t, objectChatCompletionChunk, chunk.Object)
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	require.Greater(t, chunkCount, 0)
	require.Equal(t, "hello", content.String())
}

func TestStreamingContextCancellation(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		// Send chunks slowly so context cancellation happens mid-stream.
		for i := range 100 {
			chunk := fmt.Sprintf(
				`{"id":"chatcmpl-gw","object":"chat.completion.chunk","created":1700000000,"model":"test-model","choices":[{"index":0,"delta":{"content":"chunk-%d"},"finish_reason":null}]}`,
				i,
			)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", chunk)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			time.Sleep(10 * time.Millisecond)
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	chunks, errs := provider.CompletionStream(ctx, mockCompletionParams())

	// Read a few chunks then cancel.
	chunkCount := 0
	for range chunks {
		chunkCount++
		if chunkCount >= 3 {
			cancel()
			break
		}
	}

	// Drain remaining chunks (channel will close after goroutine detects cancellation).
	for range chunks {
	}

	err = <-errs
	// After context cancellation, we must get a context error, not nil.
	require.Error(t, err)
	require.ErrorIs(t, err, context.Canceled)
}

func TestConvertError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		body       string
		wantErr    error
	}{
		{
			name:       "401 returns AuthenticationError",
			statusCode: http.StatusUnauthorized,
			body:       `{"error": {"message": "invalid token", "type": "auth_error", "code": "invalid_api_key"}}`,
			wantErr:    errors.ErrAuthentication,
		},
		{
			name:       "402 returns InsufficientFundsError",
			statusCode: http.StatusPaymentRequired,
			body:       `{"error": {"message": "payment required", "type": "insufficient_funds", "code": "insufficient_funds"}}`,
			wantErr:    errors.ErrInsufficientFunds,
		},
		{
			name:       "404 returns ModelNotFoundError",
			statusCode: http.StatusNotFound,
			body:       `{"error": {"message": "model not found", "type": "not_found", "code": "model_not_found"}}`,
			wantErr:    errors.ErrModelNotFound,
		},
		{
			name:       "429 returns RateLimitError",
			statusCode: http.StatusTooManyRequests,
			body:       `{"error": {"message": "rate limit exceeded", "type": "rate_limit", "code": "rate_limit_exceeded"}}`,
			wantErr:    errors.ErrRateLimit,
		},
		{
			name:       "502 returns UpstreamProviderError",
			statusCode: http.StatusBadGateway,
			body:       `{"error": {"message": "upstream error", "type": "upstream_error", "code": "upstream_error"}}`,
			wantErr:    ErrUpstreamProvider,
		},
		{
			name:       "504 returns TimeoutError",
			statusCode: http.StatusGatewayTimeout,
			body:       `{"error": {"message": "gateway timeout", "type": "timeout", "code": "timeout"}}`,
			wantErr:    ErrTimeout,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.statusCode)
				_, _ = w.Write([]byte(tc.body))
			}))
			t.Cleanup(srv.Close)

			provider, err := New(
				config.WithBaseURL(srv.URL),
				config.WithAPIKey("tk_test"),
				WithPlatformMode(),
			)
			require.NoError(t, err)

			ctx := context.Background()
			_, err = provider.Completion(ctx, mockCompletionParams())

			require.Error(t, err)
			require.ErrorIs(t, err, tc.wantErr)
		})
	}
}

func TestNonPlatformModeAlsoConvertsGatewayErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		body       string
		wantErr    error
	}{
		{
			name:       "402 in non-platform mode",
			statusCode: http.StatusPaymentRequired,
			body:       `{"error": {"message": "payment required", "type": "insufficient_funds", "code": "insufficient_funds"}}`,
			wantErr:    errors.ErrInsufficientFunds,
		},
		{
			name:       "502 in non-platform mode",
			statusCode: http.StatusBadGateway,
			body:       `{"error": {"message": "upstream error", "type": "upstream_error", "code": "upstream_error"}}`,
			wantErr:    ErrUpstreamProvider,
		},
		{
			name:       "504 in non-platform mode",
			statusCode: http.StatusGatewayTimeout,
			body:       `{"error": {"message": "gateway timeout", "type": "timeout", "code": "timeout"}}`,
			wantErr:    ErrTimeout,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.statusCode)
				_, _ = w.Write([]byte(tc.body))
			}))
			t.Cleanup(srv.Close)

			provider, err := New(config.WithBaseURL(srv.URL))
			require.NoError(t, err)
			require.False(t, provider.platformMode)

			ctx := context.Background()
			_, err = provider.Completion(ctx, mockCompletionParams())

			require.Error(t, err)
			require.ErrorIs(t, err, tc.wantErr)
		})
	}
}

func TestStreamingErrorConversion(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusPaymentRequired)
		_, _ = w.Write(
			[]byte(
				`{"error": {"message": "payment required", "type": "insufficient_funds", "code": "insufficient_funds"}}`,
			),
		)
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		config.WithAPIKey("tk_test"),
		WithPlatformMode(),
	)
	require.NoError(t, err)

	ctx := context.Background()
	chunks, errs := provider.CompletionStream(ctx, mockCompletionParams())

	// Drain chunks channel.
	for range chunks {
	}

	err = <-errs
	require.Error(t, err)
	require.ErrorIs(t, err, errors.ErrInsufficientFunds)
}

func TestCompletionRequestBody(t *testing.T) {
	t.Parallel()

	var (
		mu   sync.Mutex
		body map[string]any
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		mu.Lock()
		_ = json.Unmarshal(raw, &body)
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(mockCompletionResponse("ok")))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	temp := 0.7
	ctx := context.Background()
	_, err = provider.Completion(ctx, providers.CompletionParams{
		Model:       "openai:gpt-4o-mini",
		Messages:    []providers.Message{{Role: providers.RoleUser, Content: "test"}},
		Temperature: &temp,
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()

	require.Equal(t, "openai:gpt-4o-mini", body["model"])
	require.InDelta(t, 0.7, body["temperature"], 0.01)
}

func TestValidationErrors(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithBaseURL("http://localhost:9999/v1"))
	require.NoError(t, err)

	ctx := context.Background()

	t.Run("empty model returns error", func(t *testing.T) {
		t.Parallel()

		_, err := provider.Completion(ctx, providers.CompletionParams{
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "model is required")
	})

	t.Run("empty messages returns error", func(t *testing.T) {
		t.Parallel()

		_, err := provider.Completion(ctx, providers.CompletionParams{
			Model:    "openai:gpt-4o-mini",
			Messages: []providers.Message{},
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "at least one message is required")
	})
}

func TestConvertErrorNilPassthrough(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithBaseURL("http://localhost:8000/v1"))
	require.NoError(t, err)

	require.Nil(t, provider.ConvertError(nil))
}

// --- Batch tests ---

func TestCreateBatchSuccess(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/v1/batches", r.URL.Path)

		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(raw, &capturedBody))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "batch_abc123",
			"object": "batch",
			"endpoint": "/v1/chat/completions",
			"status": "validating",
			"created_at": 1714502400,
			"completion_window": "24h",
			"request_counts": {"total": 2, "completed": 0, "failed": 0},
			"metadata": {"key": "value"},
			"provider": "openai"
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	batch, err := provider.CreateBatch(context.Background(), providers.CreateBatchParams{
		Model: "openai:gpt-4o-mini",
		Requests: []providers.BatchRequestItem{
			{
				CustomID: "req-1",
				Body: map[string]any{
					"messages":   []any{map[string]any{"role": "user", "content": "Hello"}},
					"max_tokens": 100,
				},
			},
		},
		CompletionWindow: "24h",
		Metadata:         map[string]string{"key": "value"},
	})
	require.NoError(t, err)
	require.Equal(t, "batch_abc123", batch.ID)
	require.Equal(t, "batch", batch.Object)
	require.Equal(t, "/v1/chat/completions", batch.Endpoint)
	require.Equal(t, providers.BatchStatusValidating, batch.Status)
	require.Equal(t, int64(1714502400), batch.CreatedAt)
	require.Equal(t, "24h", batch.CompletionWindow)
	require.NotNil(t, batch.RequestCounts)
	require.Equal(t, 2, batch.RequestCounts.Total)
	require.Equal(t, 0, batch.RequestCounts.Completed)
	require.Equal(t, 0, batch.RequestCounts.Failed)
	require.Equal(t, "openai", batch.Provider)
	require.Equal(t, map[string]string{"key": "value"}, batch.Metadata)

	// Verify the request body.
	require.Equal(t, "openai:gpt-4o-mini", capturedBody["model"])
	require.Equal(t, "24h", capturedBody["completion_window"])
}

func TestCreateBatchSendsAuthHeader(t *testing.T) {
	t.Parallel()

	var capturedAuthHeader string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuthHeader = r.Header.Get(apiKeyHeaderName)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "batch_test",
			"object": "batch",
			"endpoint": "/v1/chat/completions",
			"status": "validating",
			"created_at": 1714502400,
			"completion_window": "24h"
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("batch-api-key"),
	)
	require.NoError(t, err)

	_, err = provider.CreateBatch(context.Background(), providers.CreateBatchParams{
		Model: "openai:gpt-4o-mini",
		Requests: []providers.BatchRequestItem{
			{CustomID: "req-1", Body: map[string]any{"messages": []any{}}},
		},
	})
	require.NoError(t, err)
	require.Equal(t, "Bearer batch-api-key", capturedAuthHeader)
}

func TestRetrieveBatchSendsProviderParam(t *testing.T) {
	t.Parallel()

	var capturedPath string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.RequestURI()
		require.Equal(t, http.MethodGet, r.Method)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "batch_abc123",
			"object": "batch",
			"endpoint": "/v1/chat/completions",
			"status": "in_progress",
			"created_at": 1714502400,
			"completion_window": "24h",
			"provider": "openai"
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	batch, err := provider.RetrieveBatch(context.Background(), "batch_abc123", "openai")
	require.NoError(t, err)
	require.Equal(t, "batch_abc123", batch.ID)
	require.Equal(t, providers.BatchStatusInProgress, batch.Status)
	require.Contains(t, capturedPath, "provider=openai")
	require.Contains(t, capturedPath, "/v1/batches/batch_abc123")
}

func TestCancelBatchSuccess(t *testing.T) {
	t.Parallel()

	var capturedMethod string
	var capturedPath string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedMethod = r.Method
		capturedPath = r.URL.RequestURI()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "batch_abc123",
			"object": "batch",
			"endpoint": "/v1/chat/completions",
			"status": "cancelling",
			"created_at": 1714502400,
			"completion_window": "24h"
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	batch, err := provider.CancelBatch(context.Background(), "batch_abc123", "openai")
	require.NoError(t, err)
	require.Equal(t, "batch_abc123", batch.ID)
	require.Equal(t, providers.BatchStatusCancelling, batch.Status)
	require.Equal(t, http.MethodPost, capturedMethod)
	require.Contains(t, capturedPath, "/v1/batches/batch_abc123/cancel")
	require.Contains(t, capturedPath, "provider=openai")
}

func TestListBatchesPagination(t *testing.T) {
	t.Parallel()

	var capturedPath string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.RequestURI()
		require.Equal(t, http.MethodGet, r.Method)

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"data": [
				{
					"id": "batch_1",
					"object": "batch",
					"endpoint": "/v1/chat/completions",
					"status": "completed",
					"created_at": 1714502400,
					"completion_window": "24h"
				},
				{
					"id": "batch_2",
					"object": "batch",
					"endpoint": "/v1/chat/completions",
					"status": "in_progress",
					"created_at": 1714502500,
					"completion_window": "24h"
				}
			]
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	limit := 10
	batches, err := provider.ListBatches(context.Background(), "openai", providers.ListBatchesOptions{
		After: "batch_0",
		Limit: &limit,
	})
	require.NoError(t, err)
	require.Len(t, batches, 2)
	require.Equal(t, "batch_1", batches[0].ID)
	require.Equal(t, "batch_2", batches[1].ID)

	// Verify query params.
	require.Contains(t, capturedPath, "provider=openai")
	require.Contains(t, capturedPath, "after=batch_0")
	require.Contains(t, capturedPath, "limit=10")
}

func TestListBatchesWithoutPagination(t *testing.T) {
	t.Parallel()

	var capturedPath string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.RequestURI()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data": []}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	batches, err := provider.ListBatches(context.Background(), "openai", providers.ListBatchesOptions{})
	require.NoError(t, err)
	require.Empty(t, batches)

	// Verify only provider param is sent (no after or limit).
	require.Contains(t, capturedPath, "provider=openai")
	require.NotContains(t, capturedPath, "after=")
	require.NotContains(t, capturedPath, "limit=")
}

func TestRetrieveBatchResultsSuccess(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodGet, r.Method)
		require.Contains(t, r.URL.Path, "/v1/batches/batch_abc123/results")
		require.Equal(t, "openai", r.URL.Query().Get("provider"))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"results": [
				{
					"custom_id": "req-1",
					"result": {
						"id": "chatcmpl-1",
						"object": "chat.completion",
						"created": 1700000000,
						"model": "gpt-4o-mini",
						"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
						"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
					},
					"error": null
				},
				{
					"custom_id": "req-2",
					"result": null,
					"error": {"code": "rate_limit", "message": "Rate limit exceeded"}
				}
			]
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	result, err := provider.RetrieveBatchResults(context.Background(), "batch_abc123", "openai")
	require.NoError(t, err)
	require.Len(t, result.Results, 2)

	// First result: successful.
	require.Equal(t, "req-1", result.Results[0].CustomID)
	require.NotNil(t, result.Results[0].Result)
	require.Equal(t, "chatcmpl-1", result.Results[0].Result.ID)
	require.Nil(t, result.Results[0].Error)

	// Second result: error.
	require.Equal(t, "req-2", result.Results[1].CustomID)
	require.Nil(t, result.Results[1].Result)
	require.NotNil(t, result.Results[1].Error)
	require.Equal(t, "rate_limit", result.Results[1].Error.Code)
	require.Equal(t, "Rate limit exceeded", result.Results[1].Error.Message)
}

// --- Batch error tests ---

func TestBatchError409(t *testing.T) {
	t.Parallel()

	t.Run("gateway returns structured status field", func(t *testing.T) {
		t.Parallel()

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"detail": "batch not yet complete", "status": "in_progress"}`))
		}))
		t.Cleanup(srv.Close)

		provider, err := New(
			config.WithBaseURL(srv.URL),
			WithGatewayKey("test-key"),
		)
		require.NoError(t, err)

		_, err = provider.RetrieveBatchResults(context.Background(), "batch_xyz", "openai")
		require.Error(t, err)

		require.True(t, stderrors.Is(err, ErrBatchNotComplete),
			"expected error to match ErrBatchNotComplete, got %v", err)

		// Batch ID comes from the caller; status comes from the structured
		// JSON field so we don't rely on free-text parsing.
		var batchErr *BatchNotCompleteError
		require.True(t, stderrors.As(err, &batchErr))
		require.Equal(t, "batch_xyz", batchErr.BatchID)
		require.Equal(t, "in_progress", batchErr.Status)
	})

	t.Run("gateway returns detail only", func(t *testing.T) {
		t.Parallel()

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"detail": "batch not yet complete"}`))
		}))
		t.Cleanup(srv.Close)

		provider, err := New(
			config.WithBaseURL(srv.URL),
			WithGatewayKey("test-key"),
		)
		require.NoError(t, err)

		_, err = provider.RetrieveBatchResults(context.Background(), "batch_xyz", "openai")
		require.Error(t, err)

		require.True(t, stderrors.Is(err, ErrBatchNotComplete))

		var batchErr *BatchNotCompleteError
		require.True(t, stderrors.As(err, &batchErr))
		require.Equal(t, "batch_xyz", batchErr.BatchID)
		require.Empty(t, batchErr.Status, "status should be empty when gateway omits it")
	})
}

func TestBatchError404(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"detail": "not found"}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	_, err = provider.RetrieveBatch(context.Background(), "batch_abc123", "openai")
	require.Error(t, err)
	require.True(t, stderrors.Is(err, errors.ErrProvider))
	require.Contains(t, err.Error(), "upgrade your gateway")
}

func TestBatchError401(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"detail": "invalid api key"}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("bad-key"),
	)
	require.NoError(t, err)

	_, err = provider.CreateBatch(context.Background(), providers.CreateBatchParams{
		Model: "openai:gpt-4o-mini",
		Requests: []providers.BatchRequestItem{
			{CustomID: "req-1", Body: map[string]any{"messages": []any{}}},
		},
	})
	require.Error(t, err)
	require.True(t, stderrors.Is(err, errors.ErrAuthentication))
}

func TestBatchError422(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"detail": "unsupported provider"}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	_, err = provider.CreateBatch(context.Background(), providers.CreateBatchParams{
		Model: "unsupported:model",
		Requests: []providers.BatchRequestItem{
			{CustomID: "req-1", Body: map[string]any{"messages": []any{}}},
		},
	})
	require.Error(t, err)
	require.True(t, stderrors.Is(err, errors.ErrProvider))
	require.Contains(t, err.Error(), "unsupported provider")
}

func TestBatchError502(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"detail": "upstream timeout"}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	_, err = provider.RetrieveBatch(context.Background(), "batch_abc123", "openai")
	require.Error(t, err)
	require.True(t, stderrors.Is(err, errors.ErrProvider))
	require.Contains(t, err.Error(), "upstream provider error")
}

func TestBatchError429(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"detail": "rate limit exceeded"}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	_, err = provider.ListBatches(context.Background(), "openai", providers.ListBatchesOptions{})
	require.Error(t, err)
	require.True(t, stderrors.Is(err, errors.ErrRateLimit))
}

// --- Helper function tests ---

func TestCompletionHTTPError409IsNotBatchError(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_, _ = w.Write([]byte(`{"detail": "conflict on completion endpoint"}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	_, err = provider.Completion(context.Background(), providers.CompletionParams{
		Model:    "test-model",
		Messages: []providers.Message{{Role: "user", Content: "hi"}},
	})
	require.Error(t, err)

	// 409 on a completion endpoint should NOT produce a BatchNotCompleteError.
	require.False(t, stderrors.Is(err, ErrBatchNotComplete),
		"completion 409 should not map to ErrBatchNotComplete, got %v", err)
}

func TestCompletionHTTPError404IsModelNotFound(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"detail": "model not found"}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	_, err = provider.Completion(context.Background(), providers.CompletionParams{
		Model:    "nonexistent:model",
		Messages: []providers.Message{{Role: "user", Content: "hi"}},
	})
	require.Error(t, err)

	// 404 on a completion endpoint should not contain "upgrade your gateway".
	require.NotContains(t, err.Error(), "upgrade your gateway")
}

// --- Rerank tests ---

func TestRerank(t *testing.T) {
	t.Parallel()

	responseJSON := `{
		"id": "rerank-test-123",
		"results": [
			{"index": 0, "relevance_score": 0.95},
			{"index": 2, "relevance_score": 0.80},
			{"index": 1, "relevance_score": 0.30}
		],
		"meta": {
			"billed_units": {"search_units": 1.0},
			"tokens": {"input_tokens": 100}
		},
		"usage": {"total_tokens": 100}
	}`

	var capturedBody []byte
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/rerank" && r.Method == http.MethodPost {
			mu.Lock()
			capturedBody, _ = io.ReadAll(r.Body)
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(responseJSON))
			return
		}
		// Handle OpenAI SDK model list or other calls.
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data": [], "object": "list"}`))
	}))
	defer server.Close()

	provider, err := New(
		config.WithBaseURL(server.URL),
		WithGatewayKey("test-key"),
	)
	require.NoError(t, err)

	rerankProvider, ok := any(provider).(providers.RerankProvider)
	require.True(t, ok, "gateway should implement RerankProvider")

	topN := 2
	resp, err := rerankProvider.Rerank(context.Background(), providers.RerankParams{
		Model:     "cohere:rerank-v3.5",
		Query:     "test query",
		Documents: []string{"doc1", "doc2", "doc3"},
		TopN:      &topN,
	})
	require.NoError(t, err)
	require.Equal(t, "rerank-test-123", resp.ID)
	require.Len(t, resp.Results, 3)
	require.Equal(t, 0.95, resp.Results[0].RelevanceScore)
	require.NotNil(t, resp.Usage)
	require.Equal(t, 100, *resp.Usage.TotalTokens)

	// Verify request body.
	mu.Lock()
	defer mu.Unlock()
	var sent providers.RerankParams
	require.NoError(t, json.Unmarshal(capturedBody, &sent))
	require.Equal(t, "cohere:rerank-v3.5", sent.Model)
	require.Equal(t, "test query", sent.Query)
	require.Equal(t, []string{"doc1", "doc2", "doc3"}, sent.Documents)
	require.Equal(t, 2, *sent.TopN)
}

func TestRerankError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		body       string
		checkErr   func(t *testing.T, err error)
	}{
		{
			name:       "401 → AuthenticationError",
			statusCode: http.StatusUnauthorized,
			body:       `{"detail": "Invalid API key"}`,
			checkErr: func(t *testing.T, err error) {
				t.Helper()
				require.ErrorIs(t, err, errors.ErrAuthentication)
			},
		},
		{
			name:       "429 → RateLimitError",
			statusCode: http.StatusTooManyRequests,
			body:       `{"detail": "Rate limit exceeded"}`,
			checkErr: func(t *testing.T, err error) {
				t.Helper()
				require.ErrorIs(t, err, errors.ErrRateLimit)
			},
		},
		{
			name:       "402 → InsufficientFundsError",
			statusCode: http.StatusPaymentRequired,
			body:       `{"detail": "Payment required"}`,
			checkErr: func(t *testing.T, err error) {
				t.Helper()
				require.ErrorIs(t, err, errors.ErrInsufficientFunds)
			},
		},
		{
			name:       "500 → ProviderError",
			statusCode: http.StatusInternalServerError,
			body:       `{"detail": "Internal server error"}`,
			checkErr: func(t *testing.T, err error) {
				t.Helper()
				require.ErrorIs(t, err, errors.ErrProvider)
			},
		},
		{
			name:       "502 → UpstreamProviderError",
			statusCode: http.StatusBadGateway,
			body:       `{"detail": "Bad gateway"}`,
			checkErr: func(t *testing.T, err error) {
				t.Helper()
				require.ErrorIs(t, err, ErrUpstreamProvider)
			},
		},
		{
			name:       "504 → TimeoutError",
			statusCode: http.StatusGatewayTimeout,
			body:       `{"detail": "Gateway timeout"}`,
			checkErr: func(t *testing.T, err error) {
				t.Helper()
				require.ErrorIs(t, err, ErrTimeout)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/rerank" {
					w.WriteHeader(tc.statusCode)
					_, _ = w.Write([]byte(tc.body))
					return
				}
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"data": [], "object": "list"}`))
			}))
			defer server.Close()

			provider, err := New(
				config.WithBaseURL(server.URL),
				WithGatewayKey("test-key"),
			)
			require.NoError(t, err)

			_, err = provider.Rerank(context.Background(), providers.RerankParams{
				Model:     "cohere:rerank-v3.5",
				Query:     "test",
				Documents: []string{"doc1"},
			})
			require.Error(t, err)
			tc.checkErr(t, err)
		})
	}
}

func TestRerankSendsGatewayHeader(t *testing.T) {
	t.Parallel()

	var (
		mu              sync.Mutex
		capturedHeaders http.Header
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/rerank" {
			mu.Lock()
			capturedHeaders = r.Header.Clone()
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"r-1","results":[]}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data": [], "object": "list"}`))
	}))
	defer server.Close()

	provider, err := New(
		config.WithBaseURL(server.URL),
		WithGatewayKey("gw_rerank_key"),
	)
	require.NoError(t, err)

	_, err = provider.Rerank(context.Background(), providers.RerankParams{
		Model:     "cohere:rerank-v3.5",
		Query:     "test",
		Documents: []string{"doc1"},
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()

	// Non-platform mode: gateway key sent via custom header.
	require.Equal(t, bearerPrefix+"gw_rerank_key", capturedHeaders.Get(apiKeyHeaderName))
}

func TestRerankValidation(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithBaseURL("http://localhost:9999/v1"))
	require.NoError(t, err)

	ctx := context.Background()

	t.Run("empty model returns error", func(t *testing.T) {
		t.Parallel()

		_, err := provider.Rerank(ctx, providers.RerankParams{
			Query:     "test",
			Documents: []string{"doc1"},
		})
		require.Error(t, err)
		require.ErrorIs(t, err, errors.ErrInvalidRequest)
		require.Contains(t, err.Error(), "model is required")
	})

	t.Run("empty query returns error", func(t *testing.T) {
		t.Parallel()

		_, err := provider.Rerank(ctx, providers.RerankParams{
			Model:     "cohere:rerank-v3.5",
			Documents: []string{"doc1"},
		})
		require.Error(t, err)
		require.ErrorIs(t, err, errors.ErrInvalidRequest)
		require.Contains(t, err.Error(), "query is required")
	})

	t.Run("empty documents returns error", func(t *testing.T) {
		t.Parallel()

		_, err := provider.Rerank(ctx, providers.RerankParams{
			Model: "cohere:rerank-v3.5",
			Query: "test",
		})
		require.Error(t, err)
		require.ErrorIs(t, err, errors.ErrInvalidRequest)
		require.Contains(t, err.Error(), "at least one document is required")
	})
}

func TestRerankSendsPlatformBearerAuth(t *testing.T) {
	t.Parallel()

	var (
		mu              sync.Mutex
		capturedHeaders http.Header
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/rerank" {
			mu.Lock()
			capturedHeaders = r.Header.Clone()
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"r-1","results":[]}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data": [], "object": "list"}`))
	}))
	defer server.Close()

	provider, err := New(
		config.WithBaseURL(server.URL),
		config.WithAPIKey("tk_platform_rerank"),
		WithPlatformMode(),
	)
	require.NoError(t, err)

	_, err = provider.Rerank(context.Background(), providers.RerankParams{
		Model:     "cohere:rerank-v3.5",
		Query:     "test",
		Documents: []string{"doc1"},
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()

	// Platform mode: Bearer auth sent via standard Authorization header.
	require.Equal(t, bearerPrefix+"tk_platform_rerank", capturedHeaders.Get("Authorization"))
	// Platform mode should not send gateway key header.
	require.Empty(t, capturedHeaders.Get(apiKeyHeaderName))
}

// --- Integration tests ---

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

	gatewayURL, token := gatewayCredentials()
	if gatewayURL == "" || token == "" {
		t.Skip("GATEWAY_API_BASE and GATEWAY_PLATFORM_TOKEN not set")
	}

	provider, err := New(
		config.WithBaseURL(gatewayURL),
		config.WithAPIKey(token),
		WithPlatformMode(),
	)
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.Completion(ctx, providers.CompletionParams{
		Model: "openai:gpt-4o-mini",
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Say 'hello' and nothing else."},
		},
	})
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Equal(t, objectChatCompletion, resp.Object)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.ContentString())
	require.Contains(t, strings.ToLower(resp.Choices[0].Message.ContentString()), "hello")

	t.Logf("Response: %s", resp.Choices[0].Message.ContentString())
	if resp.Usage != nil {
		t.Logf("Tokens used: %d", resp.Usage.TotalTokens)
	}
}

func TestIntegrationCompletionStream(t *testing.T) {
	t.Parallel()

	gatewayURL, token := gatewayCredentials()
	if gatewayURL == "" || token == "" {
		t.Skip("GATEWAY_API_BASE and GATEWAY_PLATFORM_TOKEN not set")
	}

	provider, err := New(
		config.WithBaseURL(gatewayURL),
		config.WithAPIKey(token),
		WithPlatformMode(),
	)
	require.NoError(t, err)

	ctx := context.Background()
	chunks, errs := provider.CompletionStream(ctx, providers.CompletionParams{
		Model: "openai:gpt-4o-mini",
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Count from 1 to 3, one number per line."},
		},
		Stream: true,
	})

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	require.Greater(t, chunkCount, 0, "should have received chunks")
	require.NotEmpty(t, content.String(), "should have received content")

	t.Logf("Received %d chunks", chunkCount)
	t.Logf("Content: %s", content.String())
}

// Moderation tests.

func TestModeration(t *testing.T) {
	t.Parallel()

	var (
		mu          sync.Mutex
		capturedReq map[string]any
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/v1/moderations", r.URL.Path)

		raw, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		mu.Lock()
		_ = json.Unmarshal(raw, &capturedReq)
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "modr-abc",
			"model": "omni-moderation-latest",
			"results": [{
				"flagged": true,
				"categories": {"violence": true},
				"category_scores": {"violence": 0.93}
			}]
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	resp, err := provider.Moderation(context.Background(), providers.ModerationParams{
		Model: "openai:omni-moderation-latest",
		Input: "hurt someone",
	})
	require.NoError(t, err)
	require.Equal(t, "modr-abc", resp.ID)
	require.Len(t, resp.Results, 1)
	require.True(t, resp.Results[0].Flagged)
	require.True(t, resp.Results[0].Categories["violence"])
	require.InDelta(t, 0.93, resp.Results[0].CategoryScores["violence"], 0.001)

	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, "openai:omni-moderation-latest", capturedReq["model"])
	require.Equal(t, "hurt someone", capturedReq["input"])
	// IncludeRaw must not leak into the JSON body; it is a query param only.
	_, hasIncludeRaw := capturedReq["IncludeRaw"]
	require.False(t, hasIncludeRaw)
	_, hasIncludeRawSnake := capturedReq["include_raw"]
	require.False(t, hasIncludeRawSnake)
}

func TestModerationIncludeRawAddsQuery(t *testing.T) {
	t.Parallel()

	var (
		mu          sync.Mutex
		capturedURL string
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		capturedURL = r.URL.String()
		mu.Unlock()

		require.Equal(t, "true", r.URL.Query().Get("include_raw"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","model":"x","results":[]}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	_, err = provider.Moderation(context.Background(), providers.ModerationParams{
		Model:      "openai:omni-moderation-latest",
		Input:      "x",
		IncludeRaw: true,
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Contains(t, capturedURL, "include_raw=true")
}

func TestModerationUnsupportedProvider(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"detail":"Provider anthropic does not support moderation"}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	_, err = provider.Moderation(context.Background(), providers.ModerationParams{
		Model: "anthropic:claude-3-haiku",
		Input: "x",
	})
	require.Error(t, err)

	var unsup *errors.UnsupportedOperationError
	require.True(t, stderrors.As(err, &unsup), "expected *UnsupportedOperationError, got %T", err)
	require.Equal(t, "anthropic", unsup.Provider)
	require.Equal(t, "moderation", unsup.Operation)
	require.True(t, stderrors.Is(err, errors.ErrUnsupported))
}

func TestModerationUnsupportedMultimodal(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write(
			[]byte(`{"detail":"Provider mistral does not support multimodal moderation input"}`),
		)
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)

	_, err = provider.Moderation(context.Background(), providers.ModerationParams{
		Model: "mistral:mistral-moderation-latest",
		Input: []providers.ContentPart{
			{Type: "text", Text: "hi"},
			{Type: "image_url", ImageURL: &providers.ImageURL{URL: "https://example.com/a.png"}},
		},
	})
	require.Error(t, err)

	var unsup *errors.UnsupportedOperationError
	require.True(t, stderrors.As(err, &unsup), "expected *UnsupportedOperationError, got %T", err)
	require.Equal(t, "mistral", unsup.Provider)
	require.Equal(t, "multimodal_moderation", unsup.Operation)
	require.True(t, stderrors.Is(err, errors.ErrUnsupported))
}

func TestModerationErrorMapping(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		body       string
		wantErr    error
	}{
		{
			name:       "401 auth",
			statusCode: http.StatusUnauthorized,
			body:       `{"detail":"unauthorized"}`,
			wantErr:    errors.ErrAuthentication,
		},
		{
			name:       "402 funds",
			statusCode: http.StatusPaymentRequired,
			body:       `{"detail":"over budget"}`,
			wantErr:    errors.ErrInsufficientFunds,
		},
		{
			name:       "429 rate",
			statusCode: http.StatusTooManyRequests,
			body:       `{"detail":"slow down"}`,
			wantErr:    errors.ErrRateLimit,
		},
		{
			name:       "502 upstream",
			statusCode: http.StatusBadGateway,
			body:       `{"detail":"bad gateway"}`,
			wantErr:    ErrUpstreamProvider,
		},
		{
			name:       "504 timeout",
			statusCode: http.StatusGatewayTimeout,
			body:       `{"detail":"timeout"}`,
			wantErr:    ErrTimeout,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.statusCode)
				_, _ = w.Write([]byte(tc.body))
			}))
			t.Cleanup(srv.Close)

			provider, err := New(config.WithBaseURL(srv.URL))
			require.NoError(t, err)

			_, err = provider.Moderation(context.Background(), providers.ModerationParams{
				Model: "openai:omni-moderation-latest",
				Input: "x",
			})
			require.Error(t, err)
			require.ErrorIs(t, err, tc.wantErr)
		})
	}
}

// TestModerationAuthPlatformMode verifies platform-mode Bearer auth is
// attached to the direct HTTP call we make from Moderation (the embedded
// openai-go SDK is not involved here).
func TestModerationAuthPlatformMode(t *testing.T) {
	t.Parallel()

	var (
		mu              sync.Mutex
		capturedHeaders http.Header
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		capturedHeaders = r.Header.Clone()
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","model":"x","results":[]}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		config.WithAPIKey("tk_platform_token"),
		WithPlatformMode(),
	)
	require.NoError(t, err)

	_, err = provider.Moderation(context.Background(), providers.ModerationParams{
		Model: "openai:omni-moderation-latest",
		Input: "x",
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, bearerPrefix+"tk_platform_token", capturedHeaders.Get("Authorization"))
}

// TestModerationAuthNonPlatformMode verifies the gateway key header is
// attached (via headerTransport) on the direct HTTP call.
func TestModerationAuthNonPlatformMode(t *testing.T) {
	t.Parallel()

	var (
		mu              sync.Mutex
		capturedHeaders http.Header
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		capturedHeaders = r.Header.Clone()
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","model":"x","results":[]}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("gw_test_key"),
	)
	require.NoError(t, err)

	_, err = provider.Moderation(context.Background(), providers.ModerationParams{
		Model: "openai:omni-moderation-latest",
		Input: "x",
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Equal(t, bearerPrefix+"gw_test_key", capturedHeaders.Get(apiKeyHeaderName))
}

func TestIntegrationModeration(t *testing.T) {
	t.Parallel()

	gatewayURL, token := gatewayCredentials()
	if gatewayURL == "" || token == "" {
		t.Skip("GATEWAY_API_BASE and GATEWAY_PLATFORM_TOKEN not set")
	}

	provider, err := New(
		config.WithBaseURL(gatewayURL),
		config.WithAPIKey(token),
		WithPlatformMode(),
	)
	require.NoError(t, err)

	resp, err := provider.Moderation(context.Background(), providers.ModerationParams{
		Model: "openai:omni-moderation-latest",
		Input: "I want to hurt someone",
	})
	require.NoError(t, err)
	require.NotEmpty(t, resp.Results)

	t.Logf("Moderation response: flagged=%v", resp.Results[0].Flagged)
}

// Test helpers.

// mockRoundTripper records whether it was called and delegates to a base transport.
type mockRoundTripper struct {
	base   http.RoundTripper
	called bool
	mu     sync.Mutex
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	m.mu.Lock()
	m.called = true
	m.mu.Unlock()
	return m.base.RoundTrip(req)
}

// mockCompletionParams returns standard completion params for tests.
func mockCompletionParams() providers.CompletionParams {
	return providers.CompletionParams{
		Model:    "openai:gpt-4o-mini",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	}
}

// mockCompletionResponse returns a minimal valid JSON completion response.
func mockCompletionResponse(content string) string {
	return fmt.Sprintf(`{
		"id": "chatcmpl-test",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "test-model",
		"choices": [{
			"index": 0,
			"message": {"role": "assistant", "content": %q},
			"finish_reason": "stop"
		}],
		"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
	}`, content)
}

// gatewayCredentials returns the gateway URL and platform token from
// environment variables. Returns empty strings if not set.
func gatewayCredentials() (gatewayURL string, token string) {
	return os.Getenv(envAPIBase), os.Getenv(envPlatformToken)
}

// --- Integration tests (gated by GATEWAY_API_KEY) ---

func TestIntegrationCreateAndRetrieveBatch(t *testing.T) {
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GATEWAY_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	batch, err := provider.CreateBatch(ctx, providers.CreateBatchParams{
		Model: "openai:gpt-4o-mini",
		Requests: []providers.BatchRequestItem{
			{
				CustomID: "integration-req-1",
				Body: map[string]any{
					"messages":   []any{map[string]any{"role": "user", "content": "Say hello"}},
					"max_tokens": 10,
				},
			},
		},
		CompletionWindow: "24h",
	})
	require.NoError(t, err)
	require.NotEmpty(t, batch.ID)
	require.Equal(t, "batch", batch.Object)
	require.NotEmpty(t, batch.Provider)

	// Retrieve the batch we just created.
	retrieved, err := provider.RetrieveBatch(ctx, batch.ID, batch.Provider)
	require.NoError(t, err)
	require.Equal(t, batch.ID, retrieved.ID)
}

func TestIntegrationBatchNotComplete(t *testing.T) {
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GATEWAY_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	batch, err := provider.CreateBatch(ctx, providers.CreateBatchParams{
		Model: "openai:gpt-4o-mini",
		Requests: []providers.BatchRequestItem{
			{
				CustomID: "integration-req-1",
				Body: map[string]any{
					"messages":   []any{map[string]any{"role": "user", "content": "Say hello"}},
					"max_tokens": 10,
				},
			},
		},
		CompletionWindow: "24h",
	})
	require.NoError(t, err)

	// Immediately requesting results should fail since the batch is not yet complete.
	_, err = provider.RetrieveBatchResults(ctx, batch.ID, batch.Provider)
	require.Error(t, err)
	require.True(t, stderrors.Is(err, ErrBatchNotComplete),
		"expected ErrBatchNotComplete, got %v", err)
}
