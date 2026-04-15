package gateway

import (
	"context"
	"encoding/json"
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
		wantAPIKeyHeader  string // expected X-AnyLLM-Key value; empty means the header must not be sent.
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

// Integration tests - only run if gateway is available.

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
