// Package azureopenai provides an Azure OpenAI provider implementation for any-llm.
package azureopenai

import (
	"fmt"

	"github.com/openai/openai-go/azure"
	"github.com/openai/openai-go/option"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
	anyopenai "github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	defaultAPIVersion = "preview"
	envAPIKey         = "AZURE_OPENAI_API_KEY"
	envAPIVersion     = "AZURE_OPENAI_API_VERSION"
	envBaseURL        = "AZURE_OPENAI_ENDPOINT"
	extraAPIVersion   = "api_version"
	providerName      = "azureopenai"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
	_ providers.ResponsesProvider  = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Azure OpenAI.
type Provider struct {
	*anyopenai.CompatibleProvider
}

// New creates a new Azure OpenAI provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	apiKey := cfg.ResolveAPIKey(envAPIKey)
	if apiKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	endpoint, err := cfg.ResolveBaseURL(envBaseURL, "")
	if err != nil {
		return nil, err
	}
	if endpoint == "" {
		return nil, fmt.Errorf(
			"%s endpoint is required (set via WithBaseURL option or %q env var)",
			providerName,
			envBaseURL,
		)
	}

	apiVersion := resolveAPIVersion(cfg)
	clientOpts := []option.RequestOption{
		azure.WithEndpoint(endpoint, apiVersion),
		azure.WithAPIKey(apiKey),
		option.WithHTTPClient(cfg.HTTPClient()),
	}

	base, err := anyopenai.NewCompatible(anyopenai.CompatibleConfig{
		APIKeyEnvVar:   envAPIKey,
		BaseURLEnvVar:  envBaseURL,
		Capabilities:   capabilities(),
		ClientOptions:  clientOpts,
		DefaultAPIKey:  "",
		DefaultBaseURL: "",
		Name:           providerName,
		RequireAPIKey:  true,
		RequireBaseURL: true,
	}, config.WithAPIKey(apiKey), config.WithBaseURL(endpoint), config.WithHTTPClient(cfg.HTTPClient()))
	if err != nil {
		return nil, err
	}

	return &Provider{CompatibleProvider: base}, nil
}

// capabilities returns the capabilities for Azure OpenAI.
func capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       false,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           true,
		ListModels:          true,
		Responses:           true,
	}
}

// resolveAPIVersion returns the Azure OpenAI API version from extra config, then
// AZURE_OPENAI_API_VERSION, then the default.
func resolveAPIVersion(cfg *config.Config) string {
	if v, ok := cfg.ExtraValue(extraAPIVersion); ok {
		if s, ok := v.(string); ok && s != "" {
			return s
		}
	}
	if v := cfg.ResolveEnv(envAPIVersion); v != "" {
		return v
	}
	return defaultAPIVersion
}
