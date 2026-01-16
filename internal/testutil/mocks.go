package testutil

import (
	"context"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

// MockProvider is a mock implementation of the Provider interface for testing.
type MockProvider struct {
	NameFunc             func() string
	CompletionFunc       func(ctx context.Context, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error)
	CompletionStreamFunc func(ctx context.Context, params anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error)
	EmbeddingFunc        func(ctx context.Context, params anyllm.EmbeddingParams) (*anyllm.EmbeddingResponse, error)
	ListModelsFunc       func(ctx context.Context) (*anyllm.ModelsResponse, error)
	CapabilitiesFunc     func() anyllm.ProviderCapabilities

	// Track calls for assertions
	CompletionCalls       []anyllm.CompletionParams
	CompletionStreamCalls []anyllm.CompletionParams
	EmbeddingCalls        []anyllm.EmbeddingParams
	ListModelsCalls       int
}

// Ensure MockProvider implements all interfaces.
var (
	_ anyllm.Provider           = (*MockProvider)(nil)
	_ anyllm.EmbeddingProvider  = (*MockProvider)(nil)
	_ anyllm.ModelLister        = (*MockProvider)(nil)
	_ anyllm.CapabilityProvider = (*MockProvider)(nil)
)

// NewMockProvider creates a new MockProvider with default implementations.
func NewMockProvider() *MockProvider {
	return &MockProvider{
		NameFunc: func() string { return "mock" },
		CompletionFunc: func(ctx context.Context, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
			return &anyllm.ChatCompletion{
				ID:     "mock-completion-id",
				Object: "chat.completion",
				Model:  params.Model,
				Choices: []anyllm.Choice{
					{
						Index: 0,
						Message: anyllm.Message{
							Role:    anyllm.RoleAssistant,
							Content: "Hello World",
						},
						FinishReason: anyllm.FinishReasonStop,
					},
				},
				Usage: &anyllm.Usage{
					PromptTokens:     10,
					CompletionTokens: 5,
					TotalTokens:      15,
				},
			}, nil
		},
		CompletionStreamFunc: func(ctx context.Context, params anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
			chunks := make(chan anyllm.ChatCompletionChunk, 3)
			errs := make(chan error, 1)

			go func() {
				defer close(chunks)
				defer close(errs)

				chunks <- anyllm.ChatCompletionChunk{
					ID:     "mock-chunk-id",
					Object: "chat.completion.chunk",
					Model:  params.Model,
					Choices: []anyllm.ChunkChoice{
						{Index: 0, Delta: anyllm.ChunkDelta{Role: anyllm.RoleAssistant}},
					},
				}
				chunks <- anyllm.ChatCompletionChunk{
					ID:     "mock-chunk-id",
					Object: "chat.completion.chunk",
					Model:  params.Model,
					Choices: []anyllm.ChunkChoice{
						{Index: 0, Delta: anyllm.ChunkDelta{Content: "Hello World"}},
					},
				}
				chunks <- anyllm.ChatCompletionChunk{
					ID:     "mock-chunk-id",
					Object: "chat.completion.chunk",
					Model:  params.Model,
					Choices: []anyllm.ChunkChoice{
						{Index: 0, FinishReason: anyllm.FinishReasonStop},
					},
				}
			}()

			return chunks, errs
		},
		EmbeddingFunc: func(ctx context.Context, params anyllm.EmbeddingParams) (*anyllm.EmbeddingResponse, error) {
			return &anyllm.EmbeddingResponse{
				Object: "list",
				Model:  params.Model,
				Data: []anyllm.EmbeddingData{
					{
						Object:    "embedding",
						Embedding: []float64{0.1, 0.2, 0.3},
						Index:     0,
					},
				},
				Usage: &anyllm.EmbeddingUsage{
					PromptTokens: 5,
					TotalTokens:  5,
				},
			}, nil
		},
		ListModelsFunc: func(ctx context.Context) (*anyllm.ModelsResponse, error) {
			return &anyllm.ModelsResponse{
				Object: "list",
				Data: []anyllm.Model{
					{ID: "model-1", Object: "model", OwnedBy: "mock"},
					{ID: "model-2", Object: "model", OwnedBy: "mock"},
				},
			}, nil
		},
		CapabilitiesFunc: func() anyllm.ProviderCapabilities {
			return anyllm.ProviderCapabilities{
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

func (m *MockProvider) Completion(ctx context.Context, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	m.CompletionCalls = append(m.CompletionCalls, params)
	return m.CompletionFunc(ctx, params)
}

func (m *MockProvider) CompletionStream(ctx context.Context, params anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	m.CompletionStreamCalls = append(m.CompletionStreamCalls, params)
	return m.CompletionStreamFunc(ctx, params)
}

func (m *MockProvider) Embedding(ctx context.Context, params anyllm.EmbeddingParams) (*anyllm.EmbeddingResponse, error) {
	m.EmbeddingCalls = append(m.EmbeddingCalls, params)
	return m.EmbeddingFunc(ctx, params)
}

func (m *MockProvider) ListModels(ctx context.Context) (*anyllm.ModelsResponse, error) {
	m.ListModelsCalls++
	return m.ListModelsFunc(ctx)
}

func (m *MockProvider) Capabilities() anyllm.ProviderCapabilities {
	return m.CapabilitiesFunc()
}

// MockChatCompletion creates a mock ChatCompletion response.
func MockChatCompletion(content string) *anyllm.ChatCompletion {
	return &anyllm.ChatCompletion{
		ID:     "mock-id",
		Object: "chat.completion",
		Model:  "mock-model",
		Choices: []anyllm.Choice{
			{
				Index: 0,
				Message: anyllm.Message{
					Role:    anyllm.RoleAssistant,
					Content: content,
				},
				FinishReason: anyllm.FinishReasonStop,
			},
		},
		Usage: &anyllm.Usage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
}

// MockChatCompletionWithToolCalls creates a mock ChatCompletion with tool calls.
func MockChatCompletionWithToolCalls(toolCalls []anyllm.ToolCall) *anyllm.ChatCompletion {
	return &anyllm.ChatCompletion{
		ID:     "mock-id",
		Object: "chat.completion",
		Model:  "mock-model",
		Choices: []anyllm.Choice{
			{
				Index: 0,
				Message: anyllm.Message{
					Role:      anyllm.RoleAssistant,
					Content:   "",
					ToolCalls: toolCalls,
				},
				FinishReason: anyllm.FinishReasonToolCalls,
			},
		},
		Usage: &anyllm.Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}
}

// MockChatCompletionWithReasoning creates a mock ChatCompletion with reasoning.
func MockChatCompletionWithReasoning(content, reasoning string) *anyllm.ChatCompletion {
	return &anyllm.ChatCompletion{
		ID:     "mock-id",
		Object: "chat.completion",
		Model:  "mock-model",
		Choices: []anyllm.Choice{
			{
				Index: 0,
				Message: anyllm.Message{
					Role:    anyllm.RoleAssistant,
					Content: content,
					Reasoning: &anyllm.Reasoning{
						Content: reasoning,
					},
				},
				FinishReason: anyllm.FinishReasonStop,
			},
		},
		Usage: &anyllm.Usage{
			PromptTokens:     10,
			CompletionTokens: 50,
			TotalTokens:      60,
			ReasoningTokens:  30,
		},
	}
}
