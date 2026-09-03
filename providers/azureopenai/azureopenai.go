// Package azureopenai provides Azure OpenAI through the current v1 endpoint.
package azureopenai

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/openai/openai-go/v3/option"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
	anyopenai "github.com/mozilla-ai/any-llm-go/providers/openai"
)

const (
	envAPIKey    = "AZURE_OPENAI_API_KEY" //nolint:gosec // Environment variable name, not a credential.
	envBaseURL   = "AZURE_OPENAI_ENDPOINT"
	providerName = "azureopenai"
)

var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider keeps Azure-specific endpoint and authentication policy around the
// shared OpenAI-compatible protocol implementation.
type Provider struct {
	*anyopenai.CompatibleProvider
}

// New creates an Azure OpenAI provider using the v1 endpoint.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	if len(cfg.Extra) != 0 {
		return nil, fmt.Errorf("%s provider does not support extra options", providerName)
	}

	apiKey := cfg.ResolveAPIKey(envAPIKey)
	if apiKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	endpoint, err := cfg.ResolveBaseURL(envBaseURL, "")
	if err != nil {
		return nil, fmt.Errorf("resolve endpoint: %w", err)
	}

	if endpoint == "" {
		return nil, fmt.Errorf(
			"%s endpoint is required (set via WithBaseURL option or %q env var)",
			providerName,
			envBaseURL,
		)
	}

	validateErr := validateConfig(cfg, endpoint)
	if validateErr != nil {
		return nil, validateErr
	}

	// Microsoft documents the current Azure API at /openai/v1 without a dated
	// api-version. The SDK's azure.WithEndpoint still selects legacy deployment
	// routes, so this provider configures the documented base URL directly.
	// https://learn.microsoft.com/azure/foundry/openai/api-version-lifecycle
	clientOptions := []option.RequestOption{
		option.WithAPIKey(""),
		option.WithHeaderDel("Authorization"),
		option.WithBaseURL(v1BaseURL(endpoint)),
		// openai-go v3.53.0 added origin checks for native HTTP clients, so
		// redirects cannot carry this credential outside the configured endpoint.
		// https://github.com/openai/openai-go/blob/v3.53.0/option/requestoption.go#L42-L59
		option.WithHTTPClient(cfg.HTTPClient()),
		option.WithHeader("api-key", apiKey),
	}

	base, err := anyopenai.NewCompatible(anyopenai.CompatibleConfig{
		Capabilities:        capabilities(),
		ClientOptions:       clientOptions,
		DefaultAPIKey:       apiKey,
		DefaultBaseURL:      endpoint,
		Name:                providerName,
		OpenAIMessageSchema: true,
	})
	if err != nil {
		return nil, fmt.Errorf("create compatible provider: %w", err)
	}

	return &Provider{CompatibleProvider: base}, nil
}

func capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           true,
		ListModels:          true,
	}
}

func v1BaseURL(endpoint string) string {
	baseURL := strings.TrimRight(endpoint, "/")
	if !strings.HasSuffix(baseURL, "/openai/v1") {
		baseURL += "/openai/v1"
	}

	return baseURL + "/"
}

func validateConfig(cfg *config.Config, endpoint string) error {
	endpointURL, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("parse Azure endpoint: %w", err)
	}

	if endpointURL.Scheme != "https" {
		return fmt.Errorf("azure authenticated endpoint must use HTTPS")
	}

	if endpointURL.User != nil || endpointURL.RawQuery != "" || endpointURL.Fragment != "" {
		return fmt.Errorf("azure endpoint must not contain credentials, query parameters, or a fragment")
	}

	if cfg.Headers.Get("Api-Key") != "" || cfg.Headers.Get("Authorization") != "" {
		return fmt.Errorf("azure authentication headers must be configured through the provider")
	}

	return nil
}
