// Package groq provides a Groq provider implementation for any-llm.
// Groq exposes an OpenAI-compatible API optimized for fast inference.
package groq

import (
	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	defaultBaseURL = "https://api.groq.com/openai/v1"
	envAPIKey      = "GROQ_API_KEY"
	providerName   = "groq"
)

// Object type constants for API responses.
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectList                = "list"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Groq.
// It embeds openai.CompatibleProvider since Groq exposes an OpenAI-compatible API.
type Provider struct {
	*openai.CompatibleProvider
}

// New creates a new Groq provider.
func New(opts ...config.Option) (*Provider, error) {
	base, err := openai.NewCompatible(openai.CompatibleConfig{
		APIKeyEnvVar:   envAPIKey,
		BaseURLEnvVar:  "",
		Capabilities:   groqCapabilities(),
		DefaultAPIKey:  "",
		DefaultBaseURL: defaultBaseURL,
		Name:           providerName,
		RequireAPIKey:  true,
	}, opts...)
	if err != nil {
		return nil, err
	}

	return &Provider{CompatibleProvider: base}, nil
}

// groqCapabilities returns the capabilities for the Groq provider.
func groqCapabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     false, // Groq doesn't support image inputs.
		CompletionPDF:       false,
		CompletionReasoning: false, // Groq doesn't support reasoning parameters.
		CompletionStreaming: true,
		Embedding:           false, // Groq doesn't host embedding models.
		ListModels:          true,
	}
}
