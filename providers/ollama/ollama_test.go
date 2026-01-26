package ollama

import (
	"context"
	stderrors "errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const testOllamaAvailabilityTimeout = 5 * time.Second

func TestNew(t *testing.T) {
	// Note: Not using t.Parallel() here because child test uses t.Setenv.

	t.Run("creates provider with default settings", func(t *testing.T) {
		t.Parallel()

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, providerName, provider.Name())
	})

	t.Run("creates provider with custom base URL", func(t *testing.T) {
		t.Parallel()

		provider, err := New(config.WithBaseURL("http://localhost:11435"))
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("creates provider from OLLAMA_HOST environment variable", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://custom-host:11434")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

func TestCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := New()
	require.NoError(t, err)

	caps := provider.Capabilities()

	require.True(t, caps.Completion)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionImage)
	require.False(t, caps.CompletionPDF)
	require.True(t, caps.Embedding)
	require.True(t, caps.ListModels)
}

func TestConvertMessages(t *testing.T) {
	t.Parallel()

	t.Run("converts system message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "You are a helpful assistant."},
		}

		result := convertMessages(messages)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleSystem, result[0].Role)
		require.Equal(t, "You are a helpful assistant.", result[0].Content)
	})

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result := convertMessages(messages)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleUser, result[0].Role)
		require.Equal(t, "Hello", result[0].Content)
	})

	t.Run("converts assistant message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleAssistant, Content: "Hi there!"},
		}

		result := convertMessages(messages)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleAssistant, result[0].Role)
		require.Equal(t, "Hi there!", result[0].Content)
	})

	t.Run("converts tool message to user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleTool, Content: "sunny, 22°C", ToolCallID: "call_123"},
		}

		result := convertMessages(messages)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleUser, result[0].Role) // Ollama uses user for tool results.
	})

	t.Run("converts assistant message with tool calls", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{
				Role:    providers.RoleAssistant,
				Content: "",
				ToolCalls: []providers.ToolCall{
					{
						ID:   "call_123",
						Type: toolTypeFunction,
						Function: providers.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location": "Paris"}`,
						},
					},
				},
			},
		}

		result := convertMessages(messages)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleAssistant, result[0].Role)
		require.Len(t, result[0].ToolCalls, 1)
		require.Equal(t, "get_weather", result[0].ToolCalls[0].Function.Name)
	})
}

func TestConvertDoneReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		reason   string
		expected string
	}{
		{
			name:     "stop reason",
			reason:   doneReasonStop,
			expected: providers.FinishReasonStop,
		},
		{
			name:     "empty reason",
			reason:   "",
			expected: providers.FinishReasonStop,
		},
		{
			name:     "length reason",
			reason:   doneReasonLength,
			expected: providers.FinishReasonLength,
		},
		{
			name:     "unknown reason",
			reason:   "unknown",
			expected: providers.FinishReasonStop,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := convertDoneReason(tc.reason)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestExtractImages(t *testing.T) {
	t.Parallel()

	t.Run("extracts base64 image", func(t *testing.T) {
		t.Parallel()

		msg := providers.Message{
			Role: providers.RoleUser,
			Content: []providers.ContentPart{
				{Type: "text", Text: "What's in this image?"},
				{
					Type: "image_url",
					ImageURL: &providers.ImageURL{
						URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
					},
				},
			},
		}

		images := extractImages(msg)

		require.Len(t, images, 1)
		require.Equal(t, "/9j/4AAQSkZJRg==", string(images[0]))
	})

	t.Run("ignores non-data URLs", func(t *testing.T) {
		t.Parallel()

		msg := providers.Message{
			Role: providers.RoleUser,
			Content: []providers.ContentPart{
				{
					Type: "image_url",
					ImageURL: &providers.ImageURL{
						URL: "https://example.com/image.png",
					},
				},
			},
		}

		images := extractImages(msg)

		require.Empty(t, images)
	})
}

func TestConvertTools(t *testing.T) {
	t.Parallel()

	tools := []providers.Tool{
		{
			Type: toolTypeFunction,
			Function: providers.Function{
				Name:        "get_weather",
				Description: "Get the current weather",
				Parameters: map[string]any{
					schemaKeyType: schemaTypeObject,
					schemaKeyProperties: map[string]any{
						"location": map[string]any{
							schemaKeyType:        "string",
							schemaKeyDescription: "The city name",
						},
					},
					schemaKeyRequired: []any{"location"},
				},
			},
		},
	}

	result := convertTools(tools)

	require.Len(t, result, 1)
	require.Equal(t, toolTypeFunction, result[0].Type)
	require.Equal(t, "get_weather", result[0].Function.Name)
	require.Equal(t, "Get the current weather", result[0].Function.Description)
	require.Contains(t, result[0].Function.Parameters.Required, "location")
}

func TestConvertToolCalls(t *testing.T) {
	t.Parallel()

	args := api.NewToolCallFunctionArguments()
	args.Set("location", "Paris")

	toolCalls := []api.ToolCall{
		{
			Function: api.ToolCallFunction{
				Name:      "get_weather",
				Arguments: args,
			},
		},
	}

	result := convertToolCalls(toolCalls)

	require.Len(t, result, 1)
	require.Equal(t, "call_0", result[0].ID)
	require.Equal(t, toolTypeFunction, result[0].Type)
	require.Equal(t, "get_weather", result[0].Function.Name)
	require.Contains(t, result[0].Function.Arguments, "Paris")
}

func TestConvertResponseFormat(t *testing.T) {
	t.Parallel()

	t.Run("nil format returns nil", func(t *testing.T) {
		t.Parallel()

		result := convertResponseFormat(nil)
		require.Nil(t, result)
	})

	t.Run("json_object format", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{Type: responseFormatJSON}
		result := convertResponseFormat(format)

		require.NotNil(t, result)
		require.Equal(t, `"json"`, string(result))
	})

	t.Run("json_schema format", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{
			Type: responseFormatSchema,
			JSONSchema: &providers.JSONSchema{
				Name: "test",
				Schema: map[string]any{
					schemaKeyType: schemaTypeObject,
					schemaKeyProperties: map[string]any{
						"name": map[string]any{schemaKeyType: "string"},
					},
				},
			},
		}
		result := convertResponseFormat(format)

		require.NotNil(t, result)
		require.Contains(t, string(result), schemaKeyProperties)
	})
}

func TestConvertMessage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		msg          providers.Message
		expectedRole string
	}{
		{
			name:         "user message",
			msg:          providers.Message{Role: providers.RoleUser, Content: "Hello"},
			expectedRole: providers.RoleUser,
		},
		{
			name:         "assistant message",
			msg:          providers.Message{Role: providers.RoleAssistant, Content: "Hi"},
			expectedRole: providers.RoleAssistant,
		},
		{
			name:         "system message",
			msg:          providers.Message{Role: providers.RoleSystem, Content: "You are helpful"},
			expectedRole: providers.RoleSystem,
		},
		{
			name:         "tool message becomes user",
			msg:          providers.Message{Role: providers.RoleTool, Content: "result", ToolCallID: "123"},
			expectedRole: providers.RoleUser,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := convertMessage(tc.msg)
			require.NotNil(t, result)
			require.Equal(t, tc.expectedRole, result.Role)
		})
	}
}

func TestNewStreamState(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	require.NotNil(t, state)
	require.NotEmpty(t, state.id)
	require.Greater(t, state.created, int64(0))
	require.Empty(t, state.model)
}

func TestStreamStateChunk(t *testing.T) {
	t.Parallel()

	state := &streamState{
		id:      "test-id",
		model:   "test-model",
		created: 12345,
	}

	chunk := state.chunk()

	require.Equal(t, "test-id", chunk.ID)
	require.Equal(t, objectChatCompletionChunk, chunk.Object)
	require.Equal(t, int64(12345), chunk.Created)
	require.Equal(t, "test-model", chunk.Model)
	require.Len(t, chunk.Choices, 1)
	require.Equal(t, 0, chunk.Choices[0].Index)
}

func TestStreamStateHandleChunk(t *testing.T) {
	t.Parallel()

	t.Run("handles content chunk", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		resp := &api.ChatResponse{
			Model: "llama3.2",
			Message: api.Message{
				Content: "Hello ",
			},
		}

		chunk := state.handleChunk(resp)

		require.Equal(t, objectChatCompletionChunk, chunk.Object)
		require.Equal(t, "llama3.2", chunk.Model)
		require.Len(t, chunk.Choices, 1)
		require.Equal(t, "Hello ", chunk.Choices[0].Delta.Content)
		require.Equal(t, "Hello ", state.content.String())
	})

	t.Run("handles thinking chunk", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		resp := &api.ChatResponse{
			Model: "deepseek-r1",
			Message: api.Message{
				Thinking: "Let me think...",
			},
		}

		chunk := state.handleChunk(resp)

		require.NotNil(t, chunk.Choices[0].Delta.Reasoning)
		require.Equal(t, "Let me think...", chunk.Choices[0].Delta.Reasoning.Content)
		require.Equal(t, "Let me think...", state.reasoning.String())
	})

	t.Run("handles done chunk with usage", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		state.model = "llama3.2"
		resp := &api.ChatResponse{
			Model:      "llama3.2",
			Done:       true,
			DoneReason: doneReasonStop,
			Metrics: api.Metrics{
				PromptEvalCount: 10,
				EvalCount:       20,
			},
		}

		chunk := state.handleChunk(resp)

		require.Equal(t, providers.FinishReasonStop, chunk.Choices[0].FinishReason)
		require.NotNil(t, chunk.Usage)
		require.Equal(t, 10, chunk.Usage.PromptTokens)
		require.Equal(t, 20, chunk.Usage.CompletionTokens)
		require.Equal(t, 30, chunk.Usage.TotalTokens)
	})

	t.Run("handles done chunk with tool calls", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		state.model = "llama3.2"

		args := api.NewToolCallFunctionArguments()
		args.Set("location", "Paris")

		resp := &api.ChatResponse{
			Model:      "llama3.2",
			Done:       true,
			DoneReason: doneReasonStop,
			Message: api.Message{
				ToolCalls: []api.ToolCall{
					{Function: api.ToolCallFunction{Name: "get_weather", Arguments: args}},
				},
			},
		}

		chunk := state.handleChunk(resp)

		require.Equal(t, providers.FinishReasonToolCalls, chunk.Choices[0].FinishReason)
	})
}

func TestExtractThinking(t *testing.T) {
	t.Parallel()

	t.Run("returns dedicated thinking content", func(t *testing.T) {
		t.Parallel()

		content, reasoning := extractThinking("Hello", "I'm thinking...")

		require.Equal(t, "Hello", content)
		require.NotNil(t, reasoning)
		require.Equal(t, "I'm thinking...", reasoning.Content)
	})

	t.Run("parses think tags from content", func(t *testing.T) {
		t.Parallel()

		content, reasoning := extractThinking("<think>Let me think</think>Hello world", "")

		require.Equal(t, "Hello world", content)
		require.NotNil(t, reasoning)
		require.Equal(t, "Let me think", reasoning.Content)
	})

	t.Run("returns nil reasoning when no thinking", func(t *testing.T) {
		t.Parallel()

		content, reasoning := extractThinking("Hello world", "")

		require.Equal(t, "Hello world", content)
		require.Nil(t, reasoning)
	})
}

func TestConvertError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          error
		wantSentinel error
		wantNil      bool
	}{
		{
			name:    "nil error returns nil",
			err:     nil,
			wantNil: true,
		},
		{
			name:         "connection refused becomes ProviderError",
			err:          fmt.Errorf("connection refused"),
			wantSentinel: errors.ErrProvider,
		},
		{
			name:         "StatusError 404 becomes ModelNotFoundError",
			err:          api.StatusError{StatusCode: 404, ErrorMessage: "model not found"},
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "StatusError 401 becomes AuthenticationError",
			err:          api.StatusError{StatusCode: 401, ErrorMessage: "unauthorized"},
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "StatusError 429 becomes RateLimitError",
			err:          api.StatusError{StatusCode: 429, ErrorMessage: "rate limited"},
			wantSentinel: errors.ErrRateLimit,
		},
		{
			name:         "StatusError 400 with context becomes ContextLengthError",
			err:          api.StatusError{StatusCode: 400, ErrorMessage: "context length exceeded"},
			wantSentinel: errors.ErrContextLength,
		},
		{
			name:         "StatusError 400 without context becomes InvalidRequestError",
			err:          api.StatusError{StatusCode: 400, ErrorMessage: "bad request"},
			wantSentinel: errors.ErrInvalidRequest,
		},
		{
			name:         "AuthorizationError becomes AuthenticationError",
			err:          api.AuthorizationError{StatusCode: 401},
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "generic error becomes ProviderError",
			err:          fmt.Errorf("some other error"),
			wantSentinel: errors.ErrProvider,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := &Provider{}
			result := p.ConvertError(tc.err)

			if tc.wantNil {
				require.Nil(t, result)
				return
			}

			require.NotNil(t, result)
			require.True(t, stderrors.Is(result, tc.wantSentinel))
		})
	}
}

func TestGenerateID(t *testing.T) {
	t.Parallel()

	id1 := generateID()
	id2 := generateID()

	require.NotEmpty(t, id1)
	require.NotEmpty(t, id2)
	require.True(t, strings.HasPrefix(id1, "chatcmpl-"))
	require.NotEqual(t, id1, id2) // IDs should be unique.
}

// Integration tests - only run if Ollama is available.

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

	model := testutil.TestModel(providerName)
	require.NotEmpty(t, model)

	skipTestIfOllamaUnavailable(t, model)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    model,
		Messages: testutil.SimpleMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Equal(t, objectChatCompletion, resp.Object)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
	require.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
	require.NotNil(t, resp.Usage)
}

func TestIntegrationCompletionWithSystemMessage(t *testing.T) {
	t.Parallel()

	model := testutil.TestModel(providerName)
	require.NotEmpty(t, model)

	skipTestIfOllamaUnavailable(t, model)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    model,
		Messages: testutil.MessagesWithSystem(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
}

func TestIntegrationCompletionStream(t *testing.T) {
	t.Parallel()

	model := testutil.TestModel(providerName)
	require.NotEmpty(t, model)

	skipTestIfOllamaUnavailable(t, model)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    model,
		Messages: testutil.SimpleMessages(),
		Stream:   true,
	}

	chunks, errs := provider.CompletionStream(ctx, params)

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		require.Equal(t, objectChatCompletionChunk, chunk.Object)
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	require.Greater(t, chunkCount, 0)
	require.NotEmpty(t, content.String())
}

func TestIntegrationListModels(t *testing.T) {
	t.Parallel()
	skipTestIfOllamaUnavailable(t, "")

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.ListModels(ctx)
	require.NoError(t, err)

	require.Equal(t, objectList, resp.Object)
	// Note: Models list could be empty if no models are pulled.
}

func TestIntegrationConversation(t *testing.T) {
	t.Parallel()

	model := testutil.TestModel(providerName)
	require.NotEmpty(t, model)

	skipTestIfOllamaUnavailable(t, model)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    model,
		Messages: testutil.ConversationMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model should remember the name "Alice".
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok)
	require.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestIntegrationCompletionWithTools(t *testing.T) {
	t.Parallel()

	model := testutil.TestModel(providerName)
	require.NotEmpty(t, model)

	skipTestIfOllamaUnavailable(t, model)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:      model,
		Messages:   testutil.ToolCallMessages(),
		Tools:      []providers.Tool{testutil.WeatherTool()},
		ToolChoice: "auto",
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model may or may not call the tool depending on the model.
	// Just verify we got a valid response.
	require.NotNil(t, resp.Choices[0].Message)
}

func TestIntegrationAgentLoop(t *testing.T) {
	t.Parallel()

	model := testutil.TestModel(providerName)
	require.NotEmpty(t, model)

	skipTestIfOllamaUnavailable(t, model)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	// Start with the agent loop messages (user asks, assistant calls tool, tool returns).
	messages := testutil.AgentLoopMessages()

	params := providers.CompletionParams{
		Model:    model,
		Messages: messages,
		Tools:    []providers.Tool{testutil.WeatherTool()},
	}

	// The model should respond with the weather information.
	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
	require.NotNil(t, resp.Choices[0].Message)
}

func TestIntegrationEmbedding(t *testing.T) {
	t.Parallel()

	model := testutil.EmbeddingModel(providerName)
	require.NotEmpty(t, model)

	skipTestIfOllamaUnavailable(t, model)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.EmbeddingParams{
		Model: model,
		Input: "Hello, world!",
	}

	resp, err := provider.Embedding(ctx, params)
	require.NoError(t, err)

	require.Equal(t, objectList, resp.Object)
	require.NotEmpty(t, resp.Data)
	require.NotEmpty(t, resp.Data[0].Embedding)
}

// skipTestIfOllamaUnavailable skips the test if Ollama is not running or the model is not available.
// If model is empty, only checks that Ollama is reachable.
func skipTestIfOllamaUnavailable(t *testing.T, model string) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), testOllamaAvailabilityTimeout)
	defer cancel()

	provider, err := New()
	if err != nil {
		t.Skipf("Ollama not available: %v", err)
	}

	models, err := provider.ListModels(ctx)
	if err != nil {
		t.Skipf("Ollama not reachable: %v", err)
	}

	// If no specific model requested, just checking Ollama is reachable is enough.
	if model == "" {
		return
	}

	// Check if the required model is available.
	// Models can be "llama3.2" or "llama3.2:latest", so check for prefix match.
	for _, m := range models.Data {
		if m.ID == model || strings.HasPrefix(m.ID, model+":") {
			return
		}
	}

	t.Skipf("Ollama model %q not available (install with: ollama pull %s)", model, model)
}
