// Package deepseek provides a DeepSeek provider implementation for any-llm.
// DeepSeek exposes an OpenAI-compatible API with some differences in JSON mode handling.
package deepseek

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	defaultBaseURL = "https://api.deepseek.com"
	envAPIKey      = "DEEPSEEK_API_KEY"
	providerName   = "deepseek"
)

// Object type constants for API responses.
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectList                = "list"
)

// Response format types.
const (
	responseFormatJSONObject = "json_object"
	responseFormatJSONSchema = "json_schema"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for DeepSeek.
// It embeds openai.CompatibleProvider since DeepSeek exposes an OpenAI-compatible API.
type Provider struct {
	*openai.CompatibleProvider
}

// New creates a new DeepSeek provider.
func New(opts ...config.Option) (*Provider, error) {
	base, err := openai.NewCompatible(openai.CompatibleConfig{
		APIKeyEnvVar:   envAPIKey,
		BaseURLEnvVar:  "",
		Capabilities:   deepseekCapabilities(),
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
// It overrides the base implementation to handle DeepSeek's JSON mode quirks.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	params = preprocessParams(params)
	return p.CompatibleProvider.Completion(ctx, params)
}

// CompletionStream performs a streaming chat completion request.
// It overrides the base implementation to handle DeepSeek's JSON mode quirks.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	params = preprocessParams(params)
	return p.CompatibleProvider.CompletionStream(ctx, params)
}

// deepseekCapabilities returns the capabilities for the DeepSeek provider.
func deepseekCapabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     false, // DeepSeek doesn't support images.
		CompletionPDF:       false,
		CompletionReasoning: true, // DeepSeek R1 supports reasoning.
		CompletionStreaming: true,
		Embedding:           false, // DeepSeek doesn't host embedding models.
		ListModels:          true,
	}
}

// preprocessParams handles DeepSeek's JSON mode requirements.
// DeepSeek doesn't support json_schema response format directly.
// Instead, it requires:
// 1. response_format = {"type": "json_object"}
// 2. The word "json" in the prompt
// 3. The schema embedded in the user message
//
// See: https://api-docs.deepseek.com/guides/json_mode
func preprocessParams(params providers.CompletionParams) providers.CompletionParams {
	if params.ResponseFormat == nil {
		return params
	}

	if params.ResponseFormat.Type != responseFormatJSONSchema {
		return params
	}

	if params.ResponseFormat.JSONSchema == nil {
		return params
	}

	// Attempt to convert json_schema to json_object with embedded schema in messages.
	modifiedMessages, ok := preprocessMessagesForJSONSchema(
		params.Messages,
		params.ResponseFormat.JSONSchema.Schema,
	)

	// Only convert to json_object if schema injection succeeded.
	// If injection failed (no user message, non-string content, or marshal error),
	// return original params unchanged to avoid invalid DeepSeek requests.
	if !ok {
		return params
	}

	// Return modified params with json_object format.
	return providers.CompletionParams{
		Model:             params.Model,
		Messages:          modifiedMessages,
		Temperature:       params.Temperature,
		TopP:              params.TopP,
		MaxTokens:         params.MaxTokens,
		Stop:              params.Stop,
		Stream:            params.Stream,
		StreamOptions:     params.StreamOptions,
		Tools:             params.Tools,
		ToolChoice:        params.ToolChoice,
		ParallelToolCalls: params.ParallelToolCalls,
		ResponseFormat: &providers.ResponseFormat{
			Type: responseFormatJSONObject,
		},
		ReasoningEffort: params.ReasoningEffort,
		Seed:            params.Seed,
		User:            params.User,
		Extra:           params.Extra,
	}
}

// preprocessMessagesForJSONSchema injects the JSON schema into the last user message.
// Returns the modified messages and true if injection succeeded, or the original messages
// and false if injection failed (no user message, non-string content, or marshal error).
func preprocessMessagesForJSONSchema(messages []providers.Message, schema map[string]any) ([]providers.Message, bool) {
	if len(messages) == 0 {
		return messages, false
	}

	// Find the last user message.
	lastUserIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == providers.RoleUser {
			lastUserIdx = i
			break
		}
	}

	if lastUserIdx == -1 {
		return messages, false
	}

	// Check if content is a simple string. DeepSeek JSON mode doesn't support
	// multimodal content, so we can't inject schema into content parts.
	targetMsg := messages[lastUserIdx]
	if targetMsg.IsMultiModal() {
		return messages, false
	}

	originalContent := targetMsg.ContentString()

	// Format the schema as JSON.
	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return messages, false
	}

	// Create the modified content with JSON instructions.
	modifiedContent := fmt.Sprintf(`Please respond with a JSON object that matches the following schema:

%s

Return the JSON object only, no other text, do not wrap it in `+"```json"+` or `+"```"+`.

%s`, string(schemaJSON), originalContent)

	// Create a copy of messages to avoid mutating the original.
	result := slices.Clone(messages)

	// Update the message, preserving all fields from the original.
	result[lastUserIdx] = providers.Message{
		Content:    modifiedContent,
		Name:       targetMsg.Name,
		Reasoning:  targetMsg.Reasoning,
		Role:       targetMsg.Role,
		ToolCallID: targetMsg.ToolCallID,
		ToolCalls:  targetMsg.ToolCalls,
	}

	return result, true
}
