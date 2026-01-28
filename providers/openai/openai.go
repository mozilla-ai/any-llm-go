package openai

import (
	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	defaultBaseURL = "https://api.openai.com/v1"
	envAPIKey      = "OPENAI_API_KEY"
	providerName   = "openai"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for OpenAI.
// It embeds BaseProvider which handles the OpenAI SDK integration.
type Provider struct {
	*BaseProvider
}

// New creates a new OpenAI provider.
func New(opts ...config.Option) (*Provider, error) {
	base, err := NewBase(BaseConfig{
		APIKeyEnvVar:   envAPIKey,
		Capabilities:   openAICapabilities(),
		DefaultBaseURL: defaultBaseURL,
		Name:           providerName,
		RequireAPIKey:  true,
	}, opts...)
	if err != nil {
		return nil, err
	}

	return &Provider{BaseProvider: base}, nil
}

// openAICapabilities returns the capabilities for the OpenAI provider.
func openAICapabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       false,
		CompletionReasoning: true,
		CompletionStreaming: true,
		Embedding:           true,
		ListModels:          true,
	}
}
