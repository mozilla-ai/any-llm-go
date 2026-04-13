// Package gateway provides a gateway provider implementation for any-llm.
// It connects to an any-llm gateway server, which proxies requests to
// underlying LLM providers.
//
// The gateway supports two authentication modes:
//   - Platform mode: uses a platform token as standard Bearer auth
//   - Non-platform mode: sends a gateway API key via the X-AnyLLM-Key header
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
	envAPIBase       = "GATEWAY_API_BASE"
	envAPIKey        = "GATEWAY_API_KEY"
	envPlatformToken = "GATEWAY_PLATFORM_TOKEN"
	gatewayHeader    = "X-AnyLLM-Key"
	providerName     = "gateway"
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
//     or GATEWAY_API_KEY, sent via the X-AnyLLM-Key header.
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

	if platformMode {
		return newPlatformProvider(cfg, baseURL, platformToken)
	}

	return newNonPlatformProvider(cfg, baseURL)
}

// WithGatewayKey sets the gateway API key for non-platform mode authentication.
// The key is sent via the X-AnyLLM-Key header.
func WithGatewayKey(key string) config.Option {
	return config.WithExtra("gateway_key", key)
}

// WithPlatformMode explicitly enables platform mode authentication.
// In platform mode, the token (from config.WithAPIKey or GATEWAY_PLATFORM_TOKEN)
// is used as standard Bearer authentication.
func WithPlatformMode() config.Option {
	return config.WithExtra("platform_mode", true)
}

// ConvertError extends the base OpenAI error conversion with gateway-specific
// HTTP status codes (402, 502, 504).
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *openai.Error
	if stderrors.As(err, &apiErr) {
		if converted := convertGatewayError(apiErr, err); converted != nil {
			return converted
		}
	}

	return p.CompatibleProvider.ConvertError(err)
}

// Completion performs a chat completion request.
// In platform mode, gateway-specific errors are converted to typed errors.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	resp, err := p.CompatibleProvider.Completion(ctx, params)
	if err != nil && p.platformMode {
		return nil, p.ConvertError(err)
	}

	return resp, err
}

// CompletionStream performs a streaming chat completion request.
// In platform mode, gateway-specific errors are converted to typed errors.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	if !p.platformMode {
		return p.CompatibleProvider.CompletionStream(ctx, params)
	}

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
				return
			}
		}

		if err := <-upstreamErrs; err != nil {
			errs <- p.ConvertError(err)
		}
	}()

	return chunks, errs
}

// Embedding generates embeddings for the given input.
// In platform mode, gateway-specific errors are converted to typed errors.
func (p *Provider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	resp, err := p.CompatibleProvider.Embedding(ctx, params)
	if err != nil && p.platformMode {
		return nil, p.ConvertError(err)
	}

	return resp, err
}

// ListModels returns a list of available models from the gateway.
// In platform mode, gateway-specific errors are converted to typed errors.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.CompatibleProvider.ListModels(ctx)
	if err != nil && p.platformMode {
		return nil, p.ConvertError(err)
	}

	return resp, err
}

// headerTransport wraps an http.RoundTripper to inject custom headers into
// every request. Used in non-platform mode to add the X-AnyLLM-Key header.
type headerTransport struct {
	base   http.RoundTripper
	header string
	value  string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.Header.Set(t.header, t.value)
	return t.base.RoundTrip(clone)
}

// convertGatewayError converts gateway-specific HTTP status codes to typed
// errors. Returns nil if the status code is not gateway-specific, allowing
// the caller to fall through to the base error converter.
func convertGatewayError(apiErr *openai.Error, originalErr error) error {
	switch apiErr.StatusCode {
	case http.StatusPaymentRequired:
		return errors.NewInsufficientFundsError(providerName, originalErr)
	case http.StatusBadGateway:
		return errors.NewUpstreamProviderError(providerName, originalErr)
	case http.StatusGatewayTimeout:
		return errors.NewGatewayTimeoutError(providerName, originalErr)
	default:
		return nil
	}
}

// newNonPlatformProvider creates a gateway provider in non-platform mode.
// The gateway API key is sent via the X-AnyLLM-Key header.
func newNonPlatformProvider(cfg *config.Config, baseURL string) (*Provider, error) {
	gatewayKey := resolveGatewayKey(cfg)

	var compatOpts []config.Option
	compatOpts = append(compatOpts, config.WithBaseURL(baseURL))

	if gatewayKey != "" {
		httpClient := &http.Client{
			Timeout: cfg.Timeout,
			Transport: &headerTransport{
				base:   http.DefaultTransport,
				header: gatewayHeader,
				value:  "Bearer " + gatewayKey,
			},
		}
		compatOpts = append(compatOpts, config.WithHTTPClient(httpClient))
	}

	base, err := openaiProvider.NewCompatible(openaiProvider.CompatibleConfig{
		APIKeyEnvVar:   "",
		BaseURLEnvVar:  "",
		Capabilities:   capabilities(),
		DefaultAPIKey:  "gateway-no-key",
		DefaultBaseURL: "",
		Name:           providerName,
		RequireAPIKey:  false,
	}, compatOpts...)
	if err != nil {
		return nil, err
	}

	return &Provider{
		CompatibleProvider: base,
		platformMode:       false,
	}, nil
}

// newPlatformProvider creates a gateway provider in platform mode.
// The platform token is used as standard Bearer authentication via the
// OpenAI SDK's api_key mechanism.
func newPlatformProvider(cfg *config.Config, baseURL, token string) (*Provider, error) {
	if token == "" {
		return nil, fmt.Errorf(
			"platform mode requires a token (pass WithAPIKey option or set %s env var)",
			envPlatformToken,
		)
	}

	base, err := openaiProvider.NewCompatible(openaiProvider.CompatibleConfig{
		APIKeyEnvVar:   "",
		BaseURLEnvVar:  "",
		Capabilities:   capabilities(),
		DefaultAPIKey:  "",
		DefaultBaseURL: "",
		Name:           providerName,
		RequireAPIKey:  false,
	}, config.WithAPIKey(token), config.WithBaseURL(baseURL))
	if err != nil {
		return nil, err
	}

	return &Provider{
		CompatibleProvider: base,
		platformMode:       true,
	}, nil
}

// resolveGatewayKey resolves the gateway API key from config extras or
// environment variable.
func resolveGatewayKey(cfg *config.Config) string {
	if v, ok := cfg.ExtraValue("gateway_key"); ok {
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
	if v, ok := cfg.ExtraValue("platform_mode"); ok {
		if enabled, ok := v.(bool); ok && enabled {
			token = cfg.APIKey
			if token == "" {
				token = cfg.ResolveEnv(envPlatformToken)
			}
			return true, token
		}
	}

	// Auto-detect: GATEWAY_PLATFORM_TOKEN set and no explicit API key.
	platformToken := cfg.ResolveEnv(envPlatformToken)
	if platformToken != "" && cfg.APIKey == "" {
		return true, platformToken
	}

	return false, ""
}

// capabilities returns the full set of capabilities for the gateway provider.
// The gateway can proxy to any provider, so all features are marked as
// supported. Actual support depends on the underlying provider being called.
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
