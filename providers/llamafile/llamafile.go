// Package llamafile provides a Llamafile provider implementation for any-llm.
// Llamafile is a single-file executable that bundles a model with llama.cpp,
// exposing an OpenAI-compatible API.
package llamafile

import (
	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	defaultAPIKey  = "llamafile" // Dummy key; Llamafile doesn't require auth.
	defaultBaseURL = "http://localhost:8080/v1"
	envBaseURL     = "LLAMAFILE_BASE_URL"
	providerName   = "llamafile"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Llamafile.
// It embeds openai.CompatibleProvider since Llamafile exposes an OpenAI-compatible API.
type Provider struct {
	*openai.CompatibleProvider
}

// New creates a new Llamafile provider.
func New(opts ...config.Option) (*Provider, error) {
	base, err := openai.NewCompatible(openai.CompatibleConfig{
		APIKeyEnvVar:   "", // Llamafile doesn't use an API key env var.
		BaseURLEnvVar:  envBaseURL,
		Capabilities:   llamafileCapabilities(),
		DefaultAPIKey:  defaultAPIKey,
		DefaultBaseURL: defaultBaseURL,
		Name:           providerName,
		RequireAPIKey:  false,
	}, opts...)
	if err != nil {
		return nil, err
	}

	return &Provider{CompatibleProvider: base}, nil
}

// llamafileCapabilities returns the capabilities for the Llamafile provider.
func llamafileCapabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true, // Depends on the model loaded.
		CompletionPDF:       false,
		CompletionReasoning: false, // Llamafile doesn't support reasoning natively.
		CompletionStreaming: true,
		Embedding:           true,
		ListModels:          true,
	}
}
