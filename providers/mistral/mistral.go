// Package mistral provides a Mistral provider implementation for any-llm.
// Mistral exposes an OpenAI-compatible API with some differences in message handling.
package mistral

import (
	"context"
	"slices"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	defaultBaseURL = "https://api.mistral.ai/v1/"
	envAPIKey      = "MISTRAL_API_KEY"
	providerName   = "mistral"
)

// Message patching constants.
const (
	assistantOKMessage = "OK"
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
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Mistral.
// It embeds openai.CompatibleProvider since Mistral exposes an OpenAI-compatible API.
type Provider struct {
	*openai.CompatibleProvider
}

// New creates a new Mistral provider.
func New(opts ...config.Option) (*Provider, error) {
	base, err := openai.NewCompatible(openai.CompatibleConfig{
		APIKeyEnvVar:   envAPIKey,
		BaseURLEnvVar:  "",
		Capabilities:   mistralCapabilities(),
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

// Completion performs a chat completion request.
// It overrides the base implementation to handle Mistral's API quirks.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	params = preprocessParams(params)
	return p.CompatibleProvider.Completion(ctx, params)
}

// CompletionStream performs a streaming chat completion request.
// It overrides the base implementation to handle Mistral's API quirks.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	params = preprocessParams(params)
	return p.CompatibleProvider.CompletionStream(ctx, params)
}

// mistralCapabilities returns the capabilities for the Mistral provider.
func mistralCapabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true, // Pixtral models support vision.
		CompletionPDF:       false,
		CompletionReasoning: true, // Magistral models support reasoning.
		CompletionStreaming: true,
		Embedding:           true, // mistral-embed model.
		ListModels:          true,
	}
}

// patchMessages inserts an assistant "OK" message between tool result and user messages.
// Mistral requires an assistant message between a tool result and the next user message.
func patchMessages(messages []providers.Message) []providers.Message {
	if len(messages) < 2 {
		return messages
	}

	// Count how many insertions we need for pre-allocation.
	insertions := 0
	for i := 0; i < len(messages)-1; i++ {
		if messages[i].Role == providers.RoleTool && messages[i+1].Role == providers.RoleUser {
			insertions++
		}
	}

	if insertions == 0 {
		return messages
	}

	result := make([]providers.Message, 0, len(messages)+insertions)
	for i, msg := range messages {
		result = append(result, msg)
		if i < len(messages)-1 && msg.Role == providers.RoleTool && messages[i+1].Role == providers.RoleUser {
			result = append(result, providers.Message{
				Role:    providers.RoleAssistant,
				Content: assistantOKMessage,
			})
		}
	}

	return result
}

// preprocessParams handles Mistral's API requirements.
// Mistral doesn't accept the "user" or "reasoning_effort" fields and requires
// an assistant message between tool results and user messages.
func preprocessParams(params providers.CompletionParams) providers.CompletionParams {
	params.Messages = patchMessages(slices.Clone(params.Messages))
	params.ReasoningEffort = "" // Mistral doesn't support reasoning_effort; Magistral models reason automatically.
	params.User = ""            // Mistral doesn't support the user field.
	return params
}
