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

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Object type constants for test assertions.
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectList                = "list"
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

func TestResolvePlatformMode(t *testing.T) {
	// Note: Not using t.Parallel() here because child tests use t.Setenv.

	t.Run("returns false when no platform indicators", func(t *testing.T) {
		t.Setenv(envPlatformToken, "")

		cfg, err := config.New()
		require.NoError(t, err)

		mode, token := resolvePlatformMode(cfg)
		require.False(t, mode)
		require.Empty(t, token)
	})

	t.Run("returns true with explicit WithPlatformMode and API key", func(t *testing.T) {
		t.Setenv(envPlatformToken, "")

		cfg, err := config.New(config.WithAPIKey("tk_test"), WithPlatformMode())
		require.NoError(t, err)

		mode, token := resolvePlatformMode(cfg)
		require.True(t, mode)
		require.Equal(t, "tk_test", token)
	})

	t.Run("returns true with explicit WithPlatformMode and env token", func(t *testing.T) {
		t.Setenv(envPlatformToken, "tk_from_env")

		cfg, err := config.New(WithPlatformMode())
		require.NoError(t, err)

		mode, token := resolvePlatformMode(cfg)
		require.True(t, mode)
		require.Equal(t, "tk_from_env", token)
	})

	t.Run("auto-detects when GATEWAY_PLATFORM_TOKEN set and no API key", func(t *testing.T) {
		t.Setenv(envPlatformToken, "tk_auto")

		cfg, err := config.New()
		require.NoError(t, err)

		mode, token := resolvePlatformMode(cfg)
		require.True(t, mode)
		require.Equal(t, "tk_auto", token)
	})

	t.Run("does not auto-detect when API key is explicitly set", func(t *testing.T) {
		t.Setenv(envPlatformToken, "tk_auto")

		cfg, err := config.New(config.WithAPIKey("explicit_key"))
		require.NoError(t, err)

		mode, _ := resolvePlatformMode(cfg)
		require.False(t, mode)
	})
}

func TestResolveGatewayKey(t *testing.T) {
	// Note: Not using t.Parallel() here because child tests use t.Setenv.

	t.Run("returns key from WithGatewayKey", func(t *testing.T) {
		t.Setenv(envAPIKey, "")

		cfg, err := config.New(WithGatewayKey("gw_explicit"))
		require.NoError(t, err)

		key := resolveGatewayKey(cfg)
		require.Equal(t, "gw_explicit", key)
	})

	t.Run("falls back to GATEWAY_API_KEY env var", func(t *testing.T) {
		t.Setenv(envAPIKey, "gw_from_env")

		cfg, err := config.New()
		require.NoError(t, err)

		key := resolveGatewayKey(cfg)
		require.Equal(t, "gw_from_env", key)
	})

	t.Run("returns empty when no key available", func(t *testing.T) {
		t.Setenv(envAPIKey, "")

		cfg, err := config.New()
		require.NoError(t, err)

		key := resolveGatewayKey(cfg)
		require.Empty(t, key)
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
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-test",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "test-model",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "hello"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		WithGatewayKey("gw_test_key_123"),
	)
	require.NoError(t, err)

	ctx := context.Background()
	_, err = provider.Completion(ctx, providers.CompletionParams{
		Model:    "openai:gpt-4o-mini",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()

	require.Equal(t, "Bearer gw_test_key_123", capturedHeaders.Get(gatewayHeader))
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
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-test",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "test-model",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "hello"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
		}`))
	}))
	t.Cleanup(srv.Close)

	provider, err := New(
		config.WithBaseURL(srv.URL),
		config.WithAPIKey("tk_platform_token"),
		WithPlatformMode(),
	)
	require.NoError(t, err)

	ctx := context.Background()
	_, err = provider.Completion(ctx, providers.CompletionParams{
		Model:    "openai:gpt-4o-mini",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	})
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()

	require.Equal(t, "Bearer tk_platform_token", capturedHeaders.Get("Authorization"))
	require.Empty(t, capturedHeaders.Get(gatewayHeader), "platform mode should not send X-AnyLLM-Key header")
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

func TestPlatformModeErrorConversion(t *testing.T) {
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
			wantErr:    errors.ErrUpstreamProvider,
		},
		{
			name:       "504 returns GatewayTimeoutError",
			statusCode: http.StatusGatewayTimeout,
			body:       `{"error": {"message": "gateway timeout", "type": "timeout", "code": "timeout"}}`,
			wantErr:    errors.ErrGatewayTimeout,
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
			_, err = provider.Completion(ctx, providers.CompletionParams{
				Model:    "openai:gpt-4o-mini",
				Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
			})

			require.Error(t, err)
			require.ErrorIs(t, err, tc.wantErr)
		})
	}
}

func TestNonPlatformModePassesThroughErrors(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write(
			[]byte(`{"error": {"message": "upstream error", "type": "upstream_error", "code": "upstream_error"}}`),
		)
	}))
	t.Cleanup(srv.Close)

	provider, err := New(config.WithBaseURL(srv.URL))
	require.NoError(t, err)
	require.False(t, provider.platformMode)

	ctx := context.Background()
	_, err = provider.Completion(ctx, providers.CompletionParams{
		Model:    "openai:gpt-4o-mini",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	})

	// Non-platform mode uses base error conversion, which maps 502 to ProviderError.
	require.Error(t, err)
	require.ErrorIs(t, err, errors.ErrProvider)
}

func TestStreamingPlatformModeErrorConversion(t *testing.T) {
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
	chunks, errs := provider.CompletionStream(ctx, providers.CompletionParams{
		Model:    "openai:gpt-4o-mini",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	})

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
		_, _ = w.Write([]byte(`{
			"id": "chatcmpl-test",
			"object": "chat.completion",
			"created": 1700000000,
			"model": "test-model",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "ok"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
		}`))
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

// gatewayCredentials returns the gateway URL and platform token from
// environment variables. Returns empty strings if not set.
func gatewayCredentials() (gatewayURL string, token string) {
	return os.Getenv(envAPIBase), os.Getenv(envPlatformToken)
}
