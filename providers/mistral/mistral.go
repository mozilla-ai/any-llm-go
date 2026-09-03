// Package mistral provides a Mistral provider implementation for any-llm.
// Mistral exposes an OpenAI-compatible API with some differences in message handling.
package mistral

import (
	"context"

	oaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"

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
		APIKeyEnvVar:                    envAPIKey,
		BaseURLEnvVar:                   "",
		Capabilities:                    capabilities(),
		ChatCompletionChunkTransform:    transformChunk,
		ChatCompletionRequestTransform:  transformRequest,
		ChatCompletionResponseTransform: transformResponse,
		DefaultAPIKey:                   "",
		DefaultBaseURL:                  defaultBaseURL,
		Name:                            providerName,
		RequireAPIKey:                   true,
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
	params = patchMessageParams(params)
	return p.CompatibleProvider.Completion(ctx, params)
}

// CompletionStream performs a streaming chat completion request.
// It overrides the base implementation to handle Mistral's API quirks.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	params = patchMessageParams(params)
	return p.CompatibleProvider.CompletionStream(ctx, params)
}

// capabilities returns the capabilities for the Mistral provider.
func capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true, // Pixtral models support vision.
		CompletionPDF:       false,
		CompletionReasoning: true, // Current Mistral models support adjustable reasoning.
		CompletionStreaming: true,
		CompletionTools:     true,
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

// patchMessageParams handles Mistral fields before shared request conversion.
func patchMessageParams(params providers.CompletionParams) providers.CompletionParams {
	params.Messages = patchMessages(params.Messages)
	// Mistral calls its highest documented effort xhigh. Map the binding's
	// provider-neutral max value instead of sending an invalid wire enum.
	// https://docs.mistral.ai/api/endpoint/chat
	if params.ReasoningEffort == providers.ReasoningEffortMax {
		params.ReasoningEffort = providers.ReasoningEffortXHigh
	}
	return params
}

// transformRequest maps the shared token limit to Mistral's max_tokens field
// and removes fields outside its Chat request schema.
// https://docs.mistral.ai/api?property=operation-chat_completion_v1_chat_completions_post_request_max_tokens
func transformRequest(req *oaisdk.ChatCompletionNewParams) {
	if req.MaxCompletionTokens.Valid() {
		req.MaxTokens = oaisdk.Int(req.MaxCompletionTokens.Value)
	}
	req.MaxCompletionTokens = param.Opt[int64]{}
	req.User = param.Opt[string]{}
}
