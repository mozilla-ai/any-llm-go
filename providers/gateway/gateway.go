// Package gateway provides a gateway provider implementation for any-llm.
// It connects to an any-llm gateway server, which proxies requests to
// underlying LLM providers.
//
// This package supersedes providers/platform. Where the platform provider
// decrypts provider keys client-side and fans out to individual provider
// SDKs, the gateway delegates everything to a single server-side endpoint.
//
// The gateway supports two authentication modes:
//   - Platform mode: uses a platform token as standard Bearer auth
//   - Non-platform mode: sends a gateway API key via a custom authentication
//     header (X-AnyLLM-Key)
package gateway

import (
	"context"
	stderrors "errors"
	"fmt"
	"net/http"
	"slices"

	"github.com/openai/openai-go"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
	openaiProvider "github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	// apiKeyHeaderName is the HTTP header that carries the gateway API key in
	// non-platform mode.
	apiKeyHeaderName = "X-AnyLLM-Key"

	// bearerPrefix is the Authorization scheme prefix applied to the gateway
	// key before it is placed in the gateway header (e.g. "Bearer <key>").
	bearerPrefix = "Bearer "

	// placeholderAPIKey satisfies the OpenAI SDK's requirement that an API key
	// be set. The SDK still sends it in the Authorization header, but in
	// non-platform mode real auth is carried by the gateway header, so this
	// is a non-secret placeholder that the gateway ignores.
	placeholderAPIKey = "gateway-no-key"

	// envAPIBase is the environment variable read for the gateway base URL
	// when WithBaseURL is not passed to New.
	envAPIBase = "GATEWAY_API_BASE"

	// envAPIKey is the environment variable read for the gateway API key used
	// in non-platform mode when WithGatewayKey is not passed to New.
	envAPIKey = "GATEWAY_API_KEY"

	// envPlatformToken is the environment variable read for the platform
	// token used as Bearer auth in platform mode.
	envPlatformToken = "GATEWAY_PLATFORM_TOKEN"

	// extraKeyGatewayKey is the config.Extra key used to coordinate
	// WithGatewayKey (writer) with the resolver logic in New (reader).
	extraKeyGatewayKey = "gateway_key"

	// extraKeyPlatformMode is the config.Extra key used to coordinate
	// WithPlatformMode (writer) with the resolver logic in New (reader).
	extraKeyPlatformMode = "platform_mode"

	// providerName is the value returned by Provider.Name and embedded in
	// errors produced by this package.
	providerName = "gateway"
)

// Gateway-specific error codes.
const (
	// errCodeTimeout is the BaseError.Code set on gateway timeout errors
	// (HTTP 504).
	errCodeTimeout = "gateway_timeout"

	// errCodeUpstreamProvider is the BaseError.Code set on upstream provider
	// errors (HTTP 502).
	errCodeUpstreamProvider = "upstream_provider"
)

// Gateway-specific sentinel errors for type checking with errors.Is().
var (
	// ErrTimeout is matched by errors.Is on gateway timeout errors (HTTP 504).
	ErrTimeout = stderrors.New("gateway timeout")

	// ErrUpstreamProvider is matched by errors.Is on upstream provider errors
	// (HTTP 502).
	ErrUpstreamProvider = stderrors.New("upstream provider error")
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// TimeoutError is returned when the gateway times out (HTTP 504).
type TimeoutError struct {
	errors.BaseError
}

// UpstreamProviderError is returned when the upstream provider is
// unreachable (HTTP 502).
type UpstreamProviderError struct {
	errors.BaseError
}

// Provider implements the providers.Provider interface for the any-llm gateway.
// It embeds openai.CompatibleProvider since the gateway exposes an
// OpenAI-compatible API.
type Provider struct {
	*openaiProvider.CompatibleProvider

	// platformMode is used to indicate whether the provider is operating in
	// platform mode (using platform token for Bearer auth) or non-platform mode
	// (using gateway key in custom header).
	// This affects how authentication is handled and how errors are converted.
	platformMode bool
}

// headerTransport wraps an http.RoundTripper to inject custom headers into
// every request. Used in non-platform mode to add the gateway authentication
// header.
type headerTransport struct {
	base   http.RoundTripper
	header string
	value  string
}

// New creates a new gateway provider.
//
// The gateway base URL is required and can be set via config.WithBaseURL() or
// the GATEWAY_API_BASE environment variable.
//
// Authentication mode is determined as follows:
//   - If WithPlatformMode() is passed, platform mode is used. The token is
//     resolved from config.WithAPIKey() or GATEWAY_PLATFORM_TOKEN.
//   - If GATEWAY_PLATFORM_TOKEN is set and no explicit API key or gateway key
//     is provided, platform mode is auto-detected.
//   - Otherwise, non-platform mode is used with the key from WithGatewayKey()
//     or GATEWAY_API_KEY, sent via the gateway authentication header.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	// Resolve gateway key from extras or GATEWAY_API_KEY env var.
	var gatewayKey string
	if v, ok := cfg.ExtraValue(extraKeyGatewayKey); ok {
		if key, ok := v.(string); ok && key != "" {
			gatewayKey = key
		}
	}
	if gatewayKey == "" {
		gatewayKey = cfg.ResolveEnv(envAPIKey)
	}

	// Determine authentication mode.
	platformMode := false
	var platformToken string

	// Explicit opt-in via WithPlatformMode().
	if v, ok := cfg.ExtraValue(extraKeyPlatformMode); ok {
		if enabled, ok := v.(bool); ok && enabled {
			platformMode = true
			platformToken = cfg.APIKey
			if platformToken == "" {
				platformToken = cfg.ResolveEnv(envPlatformToken)
			}
		}
	}

	// Auto-detect: GATEWAY_PLATFORM_TOKEN set and no explicit API key or
	// gateway key configured. A gateway key signals non-platform intent.
	if !platformMode {
		envToken := cfg.ResolveEnv(envPlatformToken)
		if envToken != "" && cfg.APIKey == "" && gatewayKey == "" {
			platformMode = true
			platformToken = envToken
		}
	}

	if platformMode && platformToken == "" {
		return nil, fmt.Errorf(
			"platform mode requires a token (pass WithAPIKey option or set %s env var)",
			envPlatformToken,
		)
	}

	// Pass the user's opts straight through and layer auth-specific opts on
	// top (order matters: later options win). Base URL resolution and the
	// required-URL check are delegated to NewCompatible via BaseURLEnvVar
	// and RequireBaseURL.
	compatOpts := slices.Clone(opts)
	if platformMode {
		compatOpts = append(compatOpts, config.WithAPIKey(platformToken))
	} else {
		// Non-platform mode: override any user-supplied API key with the
		// placeholder so secrets can't leak via the Authorization header.
		// Real auth, if any, is carried by the gateway header below.
		compatOpts = append(compatOpts, config.WithAPIKey(placeholderAPIKey))
		if gatewayKey != "" {
			client := newHeaderClient(cfg.HTTPClient(), bearerPrefix+gatewayKey)
			compatOpts = append(compatOpts, config.WithHTTPClient(client))
		}
	}

	base, err := openaiProvider.NewCompatible(openaiProvider.CompatibleConfig{
		APIKeyEnvVar:   "",         // Gateway uses its own key resolution.
		BaseURLEnvVar:  envAPIBase, // Env var for base URL resolution.
		Capabilities:   capabilities(),
		DefaultAPIKey:  placeholderAPIKey, // Placeholder; non-platform doesn't need real auth.
		DefaultBaseURL: "",                // No default; base URL is required.
		Name:           providerName,
		RequireAPIKey:  false, // Gateway handles auth separately.
		RequireBaseURL: true,  // Gateway has no sensible default endpoint.
	}, compatOpts...)
	if err != nil {
		return nil, err
	}

	return &Provider{
		CompatibleProvider: base,
		platformMode:       platformMode,
	}, nil
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

// WithGatewayKey sets the gateway API key for non-platform mode authentication.
// The key is sent as a Bearer-formatted value in the gateway authentication
// header (X-AnyLLM-Key).
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
			return &UpstreamProviderError{
				BaseError: errors.New(errCodeUpstreamProvider, providerName, apiErr, ErrUpstreamProvider),
			}
		case http.StatusGatewayTimeout:
			return &TimeoutError{
				BaseError: errors.New(errCodeTimeout, providerName, apiErr, ErrTimeout),
			}
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

// RoundTrip implements http.RoundTripper by cloning the request and injecting
// the configured header before delegating to the base transport.
func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.Header.Set(t.header, t.value)
	return t.base.RoundTrip(clone)
}

// newHeaderClient wraps the given HTTP client's transport to inject the
// gateway authentication header into every request, preserving the base
// client's timeout and transport settings.
func newHeaderClient(base *http.Client, headerValue string) *http.Client {
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	return &http.Client{
		Timeout: base.Timeout,
		Transport: &headerTransport{
			base:   transport,
			header: apiKeyHeaderName,
			value:  headerValue,
		},
	}
}
