// Package gateway provides a gateway provider implementation for any-llm.
// It connects to an any-llm gateway server, which proxies requests to
// underlying LLM providers.
//
// The gateway supports two authentication modes:
//   - Platform mode: uses a platform token as standard Bearer auth
//   - Non-platform mode: sends a gateway API key via the gatewayHeader header
package gateway

import (
	"context"
	stderrors "errors"
	"fmt"
	"net/http"

	"github.com/openai/openai-go"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
	openaiProvider "github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	bearerPrefix          = "Bearer "
	defaultNonPlatformKey = "gateway-no-key"
	envAPIBase            = "GATEWAY_API_BASE"
	envAPIKey             = "GATEWAY_API_KEY"
	envPlatformToken      = "GATEWAY_PLATFORM_TOKEN"
	extraKeyGatewayKey    = "gateway_key"
	extraKeyPlatformMode  = "platform_mode"
	gatewayHeader         = "X-AnyLLM-Key"
	providerName          = "gateway"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for the any-llm gateway.
// It embeds openai.CompatibleProvider since the gateway exposes an
// OpenAI-compatible API.
type Provider struct {
	*openaiProvider.CompatibleProvider
	platformMode bool
}

// headerTransport wraps an http.RoundTripper to inject custom headers into
// every request. Used in non-platform mode to add the gateway key header.
type headerTransport struct {
	base   http.RoundTripper
	header string
	value  string
}

// RoundTrip implements http.RoundTripper by cloning the request and injecting
// the configured header before delegating to the base transport.
func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.Header.Set(t.header, t.value)
	return t.base.RoundTrip(clone)
}

// New creates a new gateway provider.
//
// The gateway base URL is required and can be set via config.WithBaseURL() or
// the GATEWAY_API_BASE environment variable.
//
// Authentication mode is determined as follows:
//   - If WithPlatformMode() is passed, platform mode is used. The token is
//     resolved from config.WithAPIKey() or GATEWAY_PLATFORM_TOKEN.
//   - If GATEWAY_PLATFORM_TOKEN is set and no explicit API key is provided,
//     platform mode is auto-detected.
//   - Otherwise, non-platform mode is used with the key from WithGatewayKey()
//     or GATEWAY_API_KEY, sent via the gatewayHeader header.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	baseURL, err := cfg.ResolveBaseURL(envAPIBase, "")
	if err != nil {
		return nil, err
	}
	if baseURL == "" {
		return nil, fmt.Errorf(
			"gateway base URL is required (set via WithBaseURL option or %s env var)", envAPIBase,
		)
	}

	platformMode, platformToken := resolvePlatformMode(cfg)
	if platformMode && platformToken == "" {
		return nil, fmt.Errorf(
			"platform mode requires a token (pass WithAPIKey option or set %s env var)",
			envPlatformToken,
		)
	}

	// Build options for the underlying OpenAI-compatible provider.
	compatOpts := []config.Option{config.WithBaseURL(baseURL)}
	if cfg.Timeout > 0 {
		compatOpts = append(compatOpts, config.WithTimeout(cfg.Timeout))
	}

	httpClient := cfg.HTTPClient()
	if platformMode {
		compatOpts = append(compatOpts, config.WithAPIKey(platformToken))
	} else {
		gatewayKey := resolveGatewayKey(cfg)
		if gatewayKey != "" {
			httpClient = newHeaderClient(httpClient, bearerPrefix+gatewayKey)
		}
	}
	compatOpts = append(compatOpts, config.WithHTTPClient(httpClient))

	base, err := openaiProvider.NewCompatible(openaiProvider.CompatibleConfig{
		APIKeyEnvVar:   "", // Gateway uses its own key resolution.
		BaseURLEnvVar:  "", // Already resolved above.
		Capabilities:   capabilities(),
		DefaultAPIKey:  defaultNonPlatformKey, // Placeholder; non-platform doesn't need real auth.
		DefaultBaseURL: "",                    // No default; base URL is required.
		Name:           providerName,
		RequireAPIKey:  false, // Gateway handles auth separately.
	}, compatOpts...)
	if err != nil {
		return nil, err
	}

	return &Provider{
		CompatibleProvider: base,
		platformMode:       platformMode,
	}, nil
}

// WithGatewayKey sets the gateway API key for non-platform mode authentication.
// The key is sent as a Bearer-formatted value in the gatewayHeader header
// (i.e., "X-AnyLLM-Key: Bearer <key>").
func WithGatewayKey(key string) config.Option {
	return config.WithExtra(extraKeyGatewayKey, key)
}

// WithPlatformMode explicitly enables platform mode authentication.
// In platform mode, the token (from config.WithAPIKey or GATEWAY_PLATFORM_TOKEN)
// is used as standard Bearer authentication.
func WithPlatformMode() config.Option {
	return config.WithExtra(extraKeyPlatformMode, true)
}

// Completion performs a chat completion request.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	resp, err := p.CompatibleProvider.Completion(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return resp, nil
}

// CompletionStream performs a streaming chat completion request.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	upstreamChunks, upstreamErrs := p.CompatibleProvider.CompletionStream(ctx, params)

	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		for chunk := range upstreamChunks {
			select {
			case chunks <- chunk:
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			}
		}

		if err := <-upstreamErrs; err != nil {
			errs <- p.ConvertError(err)
		}
	}()

	return chunks, errs
}

// ConvertError converts errors to gateway-specific typed errors where
// applicable, falling back to the base OpenAI error conversion for raw
// errors. Already-converted errors (containing a BaseError) are not
// double-wrapped.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	// Check for gateway-specific HTTP status codes.
	var apiErr *openai.Error
	if stderrors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case http.StatusPaymentRequired:
			return errors.NewInsufficientFundsError(providerName, apiErr)
		case http.StatusBadGateway:
			return errors.NewUpstreamProviderError(providerName, apiErr)
		case http.StatusGatewayTimeout:
			return errors.NewGatewayTimeoutError(providerName, apiErr)
		}
	}

	// If the error was already converted (has a BaseError), return as-is
	// to avoid double-wrapping.
	var baseErr *errors.BaseError
	if stderrors.As(err, &baseErr) {
		return err
	}

	// Raw error: delegate to base OpenAI error conversion.
	return p.CompatibleProvider.ConvertError(err)
}

// Embedding generates embeddings for the given input.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	resp, err := p.CompatibleProvider.Embedding(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return resp, nil
}

// ListModels returns a list of available models from the gateway.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.CompatibleProvider.ListModels(ctx)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return resp, nil
}

// capabilities returns the full set of capabilities for the gateway provider.
// Since the gateway proxies to any backend provider, all features are
// optimistically marked as supported. Actual support depends on the
// underlying provider behind the gateway; consumers should handle
// unsupported-operation errors at call time.
func capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       true,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           true,
		ListModels:          true,
	}
}

// newHeaderClient wraps the given HTTP client's transport to inject the
// gatewayHeader header into every request, preserving the base client's
// timeout and transport settings.
func newHeaderClient(base *http.Client, headerValue string) *http.Client {
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	return &http.Client{
		Timeout: base.Timeout,
		Transport: &headerTransport{
			base:   transport,
			header: gatewayHeader,
			value:  headerValue,
		},
	}
}

// resolveGatewayKey resolves the gateway API key from config extras or
// the GATEWAY_API_KEY environment variable.
func resolveGatewayKey(cfg *config.Config) string {
	if v, ok := cfg.ExtraValue(extraKeyGatewayKey); ok {
		if key, ok := v.(string); ok && key != "" {
			return key
		}
	}

	return cfg.ResolveEnv(envAPIKey)
}

// resolvePlatformMode determines whether platform mode should be used and
// returns the platform token if applicable.
func resolvePlatformMode(cfg *config.Config) (platformMode bool, token string) {
	// Explicit opt-in via WithPlatformMode().
	if v, ok := cfg.ExtraValue(extraKeyPlatformMode); ok {
		if enabled, ok := v.(bool); ok && enabled {
			token = cfg.APIKey
			if token == "" {
				token = cfg.ResolveEnv(envPlatformToken)
			}
			return true, token
		}
	}

	// Auto-detect: GATEWAY_PLATFORM_TOKEN set and no explicit API key or
	// gateway key. If a gateway key is configured, the caller intends
	// non-platform mode even when a platform token is present in the env.
	platformToken := cfg.ResolveEnv(envPlatformToken)
	gatewayKey := resolveGatewayKey(cfg)
	if platformToken != "" && cfg.APIKey == "" && gatewayKey == "" {
		return true, platformToken
	}

	return false, ""
}
