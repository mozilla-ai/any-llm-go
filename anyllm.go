// Package anyllm provides a unified interface for interacting with multiple LLM providers.
//
// It normalizes all responses to OpenAI's format, allowing you to switch between
// providers without changing your code.
//
// Basic usage:
//
//	// Using the convenience function
//	response, err := anyllm.Completion(ctx, "openai:gpt-4", messages)
//
//	// Using a specific provider
//	provider, err := anyllm.NewProvider("anthropic", anyllm.WithAPIKey("sk-..."))
//	response, err := provider.Completion(ctx, params)
//
//	// Streaming
//	chunks, errs := provider.CompletionStream(ctx, params)
//	for chunk := range chunks {
//	    fmt.Print(chunk.Choices[0].Delta.Content)
//	}
package anyllm

import (
	"context"
	"fmt"
)

// Completion performs a chat completion request using the specified model.
// The model can be specified as "provider:model" (e.g., "openai:gpt-4") or
// just "model" if the provider option is also specified.
//
// Example:
//
//	response, err := anyllm.Completion(ctx, "openai:gpt-4o", []anyllm.Message{
//	    {Role: "user", Content: "Hello!"},
//	})
func Completion(ctx context.Context, model string, messages []Message, opts ...Option) (*ChatCompletion, error) {
	providerName, modelName := ParseModelString(model)
	if providerName == "" {
		return nil, fmt.Errorf("provider must be specified in model string (e.g., 'openai:gpt-4') or use NewProvider directly")
	}

	provider, err := NewProvider(providerName, opts...)
	if err != nil {
		return nil, err
	}

	return provider.Completion(ctx, CompletionParams{
		Model:    modelName,
		Messages: messages,
	})
}

// CompletionWithParams performs a chat completion with full parameter control.
// The provider is extracted from the model string.
//
// Example:
//
//	response, err := anyllm.CompletionWithParams(ctx, "anthropic:claude-3-opus", anyllm.CompletionParams{
//	    Messages:    messages,
//	    Temperature: ptr(0.7),
//	    MaxTokens:   ptr(1000),
//	})
func CompletionWithParams(ctx context.Context, model string, params CompletionParams, opts ...Option) (*ChatCompletion, error) {
	providerName, modelName := ParseModelString(model)
	if providerName == "" {
		return nil, fmt.Errorf("provider must be specified in model string (e.g., 'openai:gpt-4')")
	}

	provider, err := NewProvider(providerName, opts...)
	if err != nil {
		return nil, err
	}

	params.Model = modelName
	return provider.Completion(ctx, params)
}

// CompletionStream performs a streaming chat completion request.
// Returns two channels: one for chunks and one for errors.
// Both channels are closed when the stream ends.
//
// Example:
//
//	chunks, errs := anyllm.CompletionStream(ctx, "openai:gpt-4", messages)
//	for chunk := range chunks {
//	    fmt.Print(chunk.Choices[0].Delta.Content)
//	}
//	if err := <-errs; err != nil {
//	    log.Fatal(err)
//	}
func CompletionStream(ctx context.Context, model string, messages []Message, opts ...Option) (<-chan ChatCompletionChunk, <-chan error) {
	errChan := make(chan error, 1)

	providerName, modelName := ParseModelString(model)
	if providerName == "" {
		errChan <- fmt.Errorf("provider must be specified in model string (e.g., 'openai:gpt-4')")
		close(errChan)
		chunks := make(chan ChatCompletionChunk)
		close(chunks)
		return chunks, errChan
	}

	provider, err := NewProvider(providerName, opts...)
	if err != nil {
		errChan <- err
		close(errChan)
		chunks := make(chan ChatCompletionChunk)
		close(chunks)
		return chunks, errChan
	}

	return provider.CompletionStream(ctx, CompletionParams{
		Model:    modelName,
		Messages: messages,
		Stream:   true,
	})
}

// CompletionStreamWithParams performs a streaming completion with full parameter control.
func CompletionStreamWithParams(ctx context.Context, model string, params CompletionParams, opts ...Option) (<-chan ChatCompletionChunk, <-chan error) {
	errChan := make(chan error, 1)

	providerName, modelName := ParseModelString(model)
	if providerName == "" {
		errChan <- fmt.Errorf("provider must be specified in model string (e.g., 'openai:gpt-4')")
		close(errChan)
		chunks := make(chan ChatCompletionChunk)
		close(chunks)
		return chunks, errChan
	}

	provider, err := NewProvider(providerName, opts...)
	if err != nil {
		errChan <- err
		close(errChan)
		chunks := make(chan ChatCompletionChunk)
		close(chunks)
		return chunks, errChan
	}

	params.Model = modelName
	params.Stream = true
	return provider.CompletionStream(ctx, params)
}

// Embedding performs an embedding request using the specified model.
//
// Example:
//
//	response, err := anyllm.Embedding(ctx, "openai:text-embedding-3-small", "Hello, world!")
func Embedding(ctx context.Context, model string, input any, opts ...Option) (*EmbeddingResponse, error) {
	providerName, modelName := ParseModelString(model)
	if providerName == "" {
		return nil, fmt.Errorf("provider must be specified in model string (e.g., 'openai:text-embedding-3-small')")
	}

	provider, err := NewProvider(providerName, opts...)
	if err != nil {
		return nil, err
	}

	embeddingProvider, ok := provider.(EmbeddingProvider)
	if !ok {
		return nil, fmt.Errorf("provider %s does not support embeddings", providerName)
	}

	return embeddingProvider.Embedding(ctx, EmbeddingParams{
		Model: modelName,
		Input: input,
	})
}

// ListModels lists available models for the specified provider.
//
// Example:
//
//	models, err := anyllm.ListModels(ctx, "openai")
func ListModels(ctx context.Context, providerName string, opts ...Option) (*ModelsResponse, error) {
	provider, err := NewProvider(providerName, opts...)
	if err != nil {
		return nil, err
	}

	lister, ok := provider.(ModelLister)
	if !ok {
		return nil, fmt.Errorf("provider %s does not support listing models", providerName)
	}

	return lister.ListModels(ctx)
}

// Helper function to create a pointer to a value.
// Useful for optional parameters.
func Ptr[T any](v T) *T {
	return &v
}
