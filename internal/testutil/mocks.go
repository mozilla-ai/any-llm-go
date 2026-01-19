package testutil

import (
	"context"

	llm "github.com/mozilla-ai/any-llm-go"
)

// MockProvider is a mock implementation of the Provider interface for testing.
type MockProvider struct {
	NameFunc             func() string
	CompletionFunc       func(ctx context.Context, params llm.CompletionParams) (*llm.ChatCompletion, error)
	CompletionStreamFunc func(ctx context.Context, params llm.CompletionParams) (<-chan llm.ChatCompletionChunk, <-chan error)
	EmbeddingFunc        func(ctx context.Context, params llm.EmbeddingParams) (*llm.EmbeddingResponse, error)
	ListModelsFunc       func(ctx context.Context) (*llm.ModelsResponse, error)
	CapabilitiesFunc     func() llm.ProviderCapabilities

	// Track calls for assertions
	CompletionCalls       []llm.CompletionParams
	CompletionStreamCalls []llm.CompletionParams
	EmbeddingCalls        []llm.EmbeddingParams
	ListModelsCalls       int
}

// Ensure MockProvider implements all interfaces.
var (
	_ llm.Provider           = (*MockProvider)(nil)
	_ llm.EmbeddingProvider  = (*MockProvider)(nil)
	_ llm.ModelLister        = (*MockProvider)(nil)
	_ llm.CapabilityProvider = (*MockProvider)(nil)
)

// NewMockProvider creates a new MockProvider with default implementations.
func NewMockProvider() *MockProvider {
	return &MockProvider{
		NameFunc: func() string { return "mock" },
		CompletionFunc: func(ctx context.Context, params llm.CompletionParams) (*llm.ChatCompletion, error) {
			return &llm.ChatCompletion{
				ID:     "mock-completion-id",
				Object: "chat.completion",
				Model:  params.Model,
				Choices: []llm.Choice{
					{
						Index: 0,
						Message: llm.Message{
							Role:    llm.RoleAssistant,
							Content: "Hello World",
						},
						FinishReason: llm.FinishReasonStop,
					},
				},
				Usage: &llm.Usage{
					PromptTokens:     10,
					CompletionTokens: 5,
					TotalTokens:      15,
				},
			}, nil
		},
		CompletionStreamFunc: func(ctx context.Context, params llm.CompletionParams) (<-chan llm.ChatCompletionChunk, <-chan error) {
			chunks := make(chan llm.ChatCompletionChunk, 3)
			errs := make(chan error, 1)

			go func() {
				defer close(chunks)
				defer close(errs)

				chunks <- llm.ChatCompletionChunk{
					ID:     "mock-chunk-id",
					Object: "chat.completion.chunk",
					Model:  params.Model,
					Choices: []llm.ChunkChoice{
						{Index: 0, Delta: llm.ChunkDelta{Role: llm.RoleAssistant}},
					},
				}
				chunks <- llm.ChatCompletionChunk{
					ID:     "mock-chunk-id",
					Object: "chat.completion.chunk",
					Model:  params.Model,
					Choices: []llm.ChunkChoice{
						{Index: 0, Delta: llm.ChunkDelta{Content: "Hello World"}},
					},
				}
				chunks <- llm.ChatCompletionChunk{
					ID:     "mock-chunk-id",
					Object: "chat.completion.chunk",
					Model:  params.Model,
					Choices: []llm.ChunkChoice{
						{Index: 0, FinishReason: llm.FinishReasonStop},
					},
				}
			}()

			return chunks, errs
		},
		EmbeddingFunc: func(ctx context.Context, params llm.EmbeddingParams) (*llm.EmbeddingResponse, error) {
			return &llm.EmbeddingResponse{
				Object: "list",
				Model:  params.Model,
				Data: []llm.EmbeddingData{
					{
						Object:    "embedding",
						Embedding: []float64{0.1, 0.2, 0.3},
						Index:     0,
					},
				},
				Usage: &llm.EmbeddingUsage{
					PromptTokens: 5,
					TotalTokens:  5,
				},
			}, nil
		},
		ListModelsFunc: func(ctx context.Context) (*llm.ModelsResponse, error) {
			return &llm.ModelsResponse{
				Object: "list",
				Data: []llm.Model{
					{ID: "model-1", Object: "model", OwnedBy: "mock"},
					{ID: "model-2", Object: "model", OwnedBy: "mock"},
				},
			}, nil
		},
		CapabilitiesFunc: func() llm.ProviderCapabilities {
			return llm.ProviderCapabilities{
				Completion:          true,
				CompletionStreaming: true,
				Embedding:           true,
				ListModels:          true,
			}
		},
	}
}

func (m *MockProvider) Name() string {
	return m.NameFunc()
}

func (m *MockProvider) Completion(ctx context.Context, params llm.CompletionParams) (*llm.ChatCompletion, error) {
	m.CompletionCalls = append(m.CompletionCalls, params)
	return m.CompletionFunc(ctx, params)
}

func (m *MockProvider) CompletionStream(ctx context.Context, params llm.CompletionParams) (<-chan llm.ChatCompletionChunk, <-chan error) {
	m.CompletionStreamCalls = append(m.CompletionStreamCalls, params)
	return m.CompletionStreamFunc(ctx, params)
}

func (m *MockProvider) Embedding(ctx context.Context, params llm.EmbeddingParams) (*llm.EmbeddingResponse, error) {
	m.EmbeddingCalls = append(m.EmbeddingCalls, params)
	return m.EmbeddingFunc(ctx, params)
}

func (m *MockProvider) ListModels(ctx context.Context) (*llm.ModelsResponse, error) {
	m.ListModelsCalls++
	return m.ListModelsFunc(ctx)
}

func (m *MockProvider) Capabilities() llm.ProviderCapabilities {
	return m.CapabilitiesFunc()
}

// MockChatCompletion creates a mock ChatCompletion response.
func MockChatCompletion(content string) *llm.ChatCompletion {
	return &llm.ChatCompletion{
		ID:     "mock-id",
		Object: "chat.completion",
		Model:  "mock-model",
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: content,
				},
				FinishReason: llm.FinishReasonStop,
			},
		},
		Usage: &llm.Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
}

// MockChatCompletionWithToolCalls creates a mock ChatCompletion with tool calls.
func MockChatCompletionWithToolCalls(toolCalls []llm.ToolCall) *llm.ChatCompletion {
	return &llm.ChatCompletion{
		ID:     "mock-id",
		Object: "chat.completion",
		Model:  "mock-model",
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:      llm.RoleAssistant,
					Content:   "",
					ToolCalls: toolCalls,
				},
				FinishReason: llm.FinishReasonToolCalls,
			},
		},
		Usage: &llm.Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}
}

// MockChatCompletionWithReasoning creates a mock ChatCompletion with reasoning.
func MockChatCompletionWithReasoning(content, reasoning string) *llm.ChatCompletion {
	return &llm.ChatCompletion{
		ID:     "mock-id",
		Object: "chat.completion",
		Model:  "mock-model",
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: content,
					Reasoning: &llm.Reasoning{
						Content: reasoning,
					},
				},
				FinishReason: llm.FinishReasonStop,
			},
		},
		Usage: &llm.Usage{
			PromptTokens:     10,
			CompletionTokens: 50,
			TotalTokens:      60,
			ReasoningTokens:  30,
		},
	}
}
