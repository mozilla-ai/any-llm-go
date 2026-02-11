// Package llamacpp provides a provider that talks to a running llama.cpp server
// using its OpenAI-compatible HTTP API.
//
// This lets you use local llama.cpp inference (llama-server) through the same
// interface as OpenAI, OpenRouter, Groq, etc.
package llamacpp

import (
	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

const (
	// defaultAPIKey is a dummy key for the OpenAI-compatible client.
	defaultAPIKey = "llama-cpp-dummy-key"

	// defaultBaseURL is where most people run llama-server locally.
	defaultBaseURL = "http://127.0.0.1:8080/v1"

	// providerName identifies this provider in error messages and lookups.
	providerName = "llamacpp"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider is a thin wrapper around the generic OpenAI-compatible provider,
// pre-configured with llama.cpp defaults and quirks.
type Provider struct {
	*openai.CompatibleProvider
}

// New returns a Provider that communicates with a llama.cpp server.
func New(opts ...config.Option) (*Provider, error) {
	base, err := openai.NewCompatible(openai.CompatibleConfig{
		APIKeyEnvVar:   "",
		BaseURLEnvVar:  "",
		Capabilities:   capabilities(),
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

// capabilities returns the feature set that a typical recent llama.cpp
// server actually implements reliably through its /v1 endpoint.
func capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionStreaming: true,
		Embedding:           true,
		ListModels:          true,
	}
}
