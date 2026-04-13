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
// The key is sent as a Bearer-formatted value in the X-AnyLLM-Key header
// (i.e., "X-AnyLLM-Key: Bearer <key>").
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
// HTTP status codes (402, 502, 504). This method handles raw (unconverted)
// errors and implements the providers.ErrorConverter interface.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *openai.Error
	if stderrors.As(err, &apiErr) {
		if converted := convertGatewayError(apiErr); converted != nil {
			return converted
		}
	}

	return p.CompatibleProvider.ConvertError(err)
}

// Completion performs a chat completion request.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	resp, err := p.CompatibleProvider.Completion(ctx, params)
	if err != nil {
		return nil, p.reclassifyError(err)
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
				return
			}
		}

		if err := <-upstreamErrs; err != nil {
			errs <- p.reclassifyError(err)
		}
	}()

	return chunks, errs
}

// Embedding generates embeddings for the given input.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	resp, err := p.CompatibleProvider.Embedding(ctx, params)
	if err != nil {
		return nil, p.reclassifyError(err)
	}

	return resp, nil
}

// ListModels returns a list of available models from the gateway.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.CompatibleProvider.ListModels(ctx)
	if err != nil {
		return nil, p.reclassifyError(err)
	}

	return resp, nil
}

// reclassifyError checks if an already-converted error originated from a
// gateway-specific HTTP status code and re-classifies it. Non-gateway errors
// pass through unchanged, avoiding double-wrapping of errors that were
// already converted by CompatibleProvider.ConvertError.
func (p *Provider) reclassifyError(err error) error {
	var apiErr *openai.Error
	if stderrors.As(err, &apiErr) {
		if converted := convertGatewayError(apiErr); converted != nil {
			return converted
		}
	}

	return err
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
// errors. The apiErr is wrapped directly to avoid double-wrapping when the
// error has already been converted by the base provider. Returns nil if the
// status code is not gateway-specific, allowing the caller to fall through.
func convertGatewayError(apiErr *openai.Error) error {
	switch apiErr.StatusCode {
	case http.StatusPaymentRequired:
		return errors.NewInsufficientFundsError(providerName, apiErr)
	case http.StatusBadGateway:
		return errors.NewUpstreamProviderError(providerName, apiErr)
	case http.StatusGatewayTimeout:
		return errors.NewGatewayTimeoutError(providerName, apiErr)
	default:
		return nil
	}
}

// newNonPlatformProvider creates a gateway provider in non-platform mode.
// The gateway API key is sent via the X-AnyLLM-Key header.
func newNonPlatformProvider(cfg *config.Config, baseURL string) (*Provider, error) {
	gatewayKey := resolveGatewayKey(cfg)

	compatOpts := forwardConfigOptions(cfg, baseURL)

	if gatewayKey != "" {
		baseClient := cfg.HTTPClient()
		baseTransport := baseClient.Transport
		if baseTransport == nil {
			baseTransport = http.DefaultTransport
		}

		httpClient := &http.Client{
			Timeout: baseClient.Timeout,
			Transport: &headerTransport{
				base:   baseTransport,
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

	compatOpts := forwardConfigOptions(cfg, baseURL)
	compatOpts = append(compatOpts, config.WithAPIKey(token))

	base, err := openaiProvider.NewCompatible(openaiProvider.CompatibleConfig{
		APIKeyEnvVar:   "",
		BaseURLEnvVar:  "",
		Capabilities:   capabilities(),
		DefaultAPIKey:  "",
		DefaultBaseURL: "",
		Name:           providerName,
		RequireAPIKey:  false,
	}, compatOpts...)
	if err != nil {
		return nil, err
	}

	return &Provider{
		CompatibleProvider: base,
		platformMode:       true,
	}, nil
}

// forwardConfigOptions builds a slice of config options that forwards
// user-supplied settings (base URL, timeout, HTTP client) to the underlying
// CompatibleProvider. This ensures settings like WithTimeout and
// WithHTTPClient are not silently dropped.
func forwardConfigOptions(cfg *config.Config, baseURL string) []config.Option {
	opts := []config.Option{config.WithBaseURL(baseURL)}

	if cfg.Timeout > 0 {
		opts = append(opts, config.WithTimeout(cfg.Timeout))
	}

	return opts
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
