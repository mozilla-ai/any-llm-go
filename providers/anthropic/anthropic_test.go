package anthropic

import (
	"context"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestNew(t *testing.T) {
	t.Run("creates provider with API key", func(t *testing.T) {
		provider, err := New(config.WithAPIKey("test-api-key"))
		require.NoError(t, err)
		assert.NotNil(t, provider)
		assert.Equal(t, "anthropic", provider.Name())
	})

	t.Run("creates provider from environment variable", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "env-api-key")

		provider, err := New()
		require.NoError(t, err)
		assert.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "")

		provider, err := New()
		assert.Nil(t, provider)
		assert.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		assert.ErrorAs(t, err, &missingKeyErr)
		assert.Equal(t, "anthropic", missingKeyErr.Provider)
		assert.Equal(t, "ANTHROPIC_API_KEY", missingKeyErr.EnvVar)
	})
}

func TestCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	caps := provider.Capabilities()

	assert.True(t, caps.Completion)
	assert.True(t, caps.CompletionStreaming)
	assert.True(t, caps.CompletionReasoning)
	assert.True(t, caps.CompletionImage)
	assert.True(t, caps.CompletionPDF)
	assert.False(t, caps.Embedding) // Anthropic doesn't support embeddings.
	assert.False(t, caps.ListModels)
}

func TestConvertMessages(t *testing.T) {
	t.Parallel()

	t.Run("extracts system message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "You are a helpful assistant."},
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system := convertMessages(messages)

		assert.Equal(t, "You are a helpful assistant.", system)
		assert.Len(t, result, 1) // Only user message.
	})

	t.Run("concatenates multiple system messages", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "First part."},
			{Role: providers.RoleSystem, Content: "Second part."},
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system := convertMessages(messages)

		assert.Equal(t, "First part.\nSecond part.", system)
		assert.Len(t, result, 1)
	})

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system := convertMessages(messages)

		assert.Empty(t, system)
		assert.Len(t, result, 1)
	})

	t.Run("converts assistant message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleAssistant, Content: "Hi there!"},
		}

		result, system := convertMessages(messages)

		assert.Empty(t, system)
		assert.Len(t, result, 2)
	})

	t.Run("converts assistant message with tool calls", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "What's the weather?"},
			{
				Role:    providers.RoleAssistant,
				Content: "",
				ToolCalls: []providers.ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: providers.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location": "Paris"}`,
						},
					},
				},
			},
		}

		result, _ := convertMessages(messages)

		assert.Len(t, result, 2)
	})

	t.Run("converts tool result to user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "What's the weather?"},
			{
				Role:    providers.RoleAssistant,
				Content: "",
				ToolCalls: []providers.ToolCall{
					{
						ID:       "call_123",
						Type:     "function",
						Function: providers.FunctionCall{Name: "get_weather", Arguments: `{"location": "Paris"}`},
					},
				},
			},
			{Role: providers.RoleTool, Content: "sunny, 22°C", ToolCallID: "call_123"},
		}

		result, _ := convertMessages(messages)

		assert.Len(t, result, 3)
	})
}

func TestConvertImagePart(t *testing.T) {
	t.Parallel()

	t.Run("converts URL image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "https://example.com/image.png"}
		result := convertImagePart(img)
		assert.NotNil(t, result)
	})

	t.Run("converts base64 image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
		result := convertImagePart(img)
		assert.NotNil(t, result)
	})
}

func TestConvertStopReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "end_turn",
			input:    "end_turn",
			expected: providers.FinishReasonStop,
		},
		{
			name:     "max_tokens",
			input:    "max_tokens",
			expected: providers.FinishReasonLength,
		},
		{
			name:     "tool_use",
			input:    "tool_use",
			expected: providers.FinishReasonToolCalls,
		},
		{
			name:     "stop_sequence",
			input:    "stop_sequence",
			expected: providers.FinishReasonStop,
		},
		{
			name:     "unknown",
			input:    "unknown",
			expected: providers.FinishReasonStop,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := convertStopReason(tc.input)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestNewStreamState(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	assert.NotNil(t, state)
	assert.Equal(t, -1, state.currentToolIdx)
	assert.Empty(t, state.messageID)
	assert.Empty(t, state.model)
	assert.Nil(t, state.toolCalls)
}

func TestStreamStateHandleTextDelta(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleTextDelta("Hello ")
	require.NotNil(t, chunk)
	assert.Equal(t, "msg_123", chunk.ID)
	assert.Equal(t, "claude-3", chunk.Model)
	assert.Equal(t, "chat.completion.chunk", chunk.Object)
	require.Len(t, chunk.Choices, 1)
	assert.Equal(t, "Hello ", chunk.Choices[0].Delta.Content)

	// Verify content is accumulated.
	chunk2 := state.handleTextDelta("world!")
	require.NotNil(t, chunk2)
	assert.Equal(t, "world!", chunk2.Choices[0].Delta.Content)
	assert.Equal(t, "Hello world!", state.content.String())
}

func TestStreamStateHandleThinkingDelta(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleThinkingDelta("Let me think...")
	require.NotNil(t, chunk)
	assert.Equal(t, "msg_123", chunk.ID)
	require.Len(t, chunk.Choices, 1)
	require.NotNil(t, chunk.Choices[0].Delta.Reasoning)
	assert.Equal(t, "Let me think...", chunk.Choices[0].Delta.Reasoning.Content)

	// Verify reasoning is accumulated.
	assert.Equal(t, "Let me think...", state.reasoning.String())
}

func TestStreamStateHandleInputJSONDelta(t *testing.T) {
	t.Parallel()

	t.Run("returns nil when no tool calls", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		chunk := state.handleInputJSONDelta(`{"key":`)
		assert.Nil(t, chunk)
	})

	t.Run("returns nil when tool index out of bounds", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		state.currentToolIdx = 5 // Out of bounds.
		state.toolCalls = []providers.ToolCall{
			{ID: "call_1", Type: "function", Function: providers.FunctionCall{Name: "get_weather", Arguments: ""}},
		}
		chunk := state.handleInputJSONDelta(`{"key":`)
		assert.Nil(t, chunk)
	})

	t.Run("appends to current tool call arguments", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		state.messageID = "msg_123"
		state.model = "claude-3"
		state.currentToolIdx = 0
		state.toolCalls = []providers.ToolCall{
			{ID: "call_1", Type: "function", Function: providers.FunctionCall{Name: "get_weather", Arguments: ""}},
		}

		chunk := state.handleInputJSONDelta(`{"location":`)
		require.NotNil(t, chunk)
		assert.Equal(t, `{"location":`, state.toolCalls[0].Function.Arguments)

		chunk2 := state.handleInputJSONDelta(`"Paris"}`)
		require.NotNil(t, chunk2)
		assert.Equal(t, `{"location":"Paris"}`, state.toolCalls[0].Function.Arguments)
	})
}

func TestApplyThinking(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name              string
		effort            providers.ReasoningEffort
		initialMaxTokens  int64
		expectedMaxTokens int64
		expectThinking    bool
	}{
		{
			name:              "empty effort does nothing",
			effort:            "",
			initialMaxTokens:  1000,
			expectedMaxTokens: 1000,
			expectThinking:    false,
		},
		{
			name:              "ReasoningEffortNone does nothing",
			effort:            providers.ReasoningEffortNone,
			initialMaxTokens:  1000,
			expectedMaxTokens: 1000,
			expectThinking:    false,
		},
		{
			name:              "invalid effort does nothing",
			effort:            "invalid",
			initialMaxTokens:  1000,
			expectedMaxTokens: 1000,
			expectThinking:    false,
		},
		{
			name:              "low effort increases tokens when insufficient",
			effort:            providers.ReasoningEffortLow,
			initialMaxTokens:  1000,
			expectedMaxTokens: 2048, // budget=1024, min=2048
			expectThinking:    true,
		},
		{
			name:              "low effort preserves tokens when sufficient",
			effort:            providers.ReasoningEffortLow,
			initialMaxTokens:  10000,
			expectedMaxTokens: 10000,
			expectThinking:    true,
		},
		{
			name:              "medium effort increases tokens when insufficient",
			effort:            providers.ReasoningEffortMedium,
			initialMaxTokens:  1000,
			expectedMaxTokens: 8192, // budget=4096, min=8192
			expectThinking:    true,
		},
		{
			name:              "high effort increases tokens when insufficient",
			effort:            providers.ReasoningEffortHigh,
			initialMaxTokens:  1000,
			expectedMaxTokens: 32768, // budget=16384, min=32768
			expectThinking:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := &anthropic.MessageNewParams{MaxTokens: tc.initialMaxTokens}
			applyThinking(req, tc.effort, tc.initialMaxTokens)
			assert.Equal(t, tc.expectedMaxTokens, req.MaxTokens)
			if tc.expectThinking {
				assert.NotNil(t, req.Thinking)
			}
		})
	}
}

func TestConvertMessage(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		msg       providers.Message
		expectNil bool
	}{
		{
			name:      "system role returns nil",
			msg:       providers.Message{Role: providers.RoleSystem, Content: "System prompt"},
			expectNil: true,
		},
		{
			name:      "unknown role returns nil",
			msg:       providers.Message{Role: "unknown", Content: "Content"},
			expectNil: true,
		},
		{
			name:      "user role converts",
			msg:       providers.Message{Role: providers.RoleUser, Content: "Hello"},
			expectNil: false,
		},
		{
			name:      "assistant role converts",
			msg:       providers.Message{Role: providers.RoleAssistant, Content: "Hi there!"},
			expectNil: false,
		},
		{
			name:      "tool role converts",
			msg:       providers.Message{Role: providers.RoleTool, Content: "Result", ToolCallID: "call_123"},
			expectNil: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := convertMessage(tc.msg)
			if tc.expectNil {
				assert.Nil(t, result)
			} else {
				assert.NotNil(t, result)
			}
		})
	}
}

func TestConvertToolCall(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		toolCall    providers.ToolCall
		expectInput bool
	}{
		{
			name: "valid JSON arguments",
			toolCall: providers.ToolCall{
				ID:   "call_123",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "get_weather",
					Arguments: `{"location": "Paris"}`,
				},
			},
			expectInput: true,
		},
		{
			name: "invalid JSON arguments results in nil input",
			toolCall: providers.ToolCall{
				ID:   "call_456",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "get_weather",
					Arguments: `{invalid json`,
				},
			},
			expectInput: false,
		},
		{
			name: "empty arguments results in nil input",
			toolCall: providers.ToolCall{
				ID:   "call_789",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "get_weather",
					Arguments: "",
				},
			},
			expectInput: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := convertToolCall(tc.toolCall)
			require.NotNil(t, result.OfToolUse)
			assert.Equal(t, tc.toolCall.ID, result.OfToolUse.ID)
			assert.Equal(t, tc.toolCall.Function.Name, result.OfToolUse.Name)
			assert.Equal(t, "tool_use", string(result.OfToolUse.Type))
			if tc.expectInput {
				assert.NotNil(t, result.OfToolUse.Input)
			} else {
				assert.Nil(t, result.OfToolUse.Input)
			}
		})
	}
}

func TestThinkingBudget(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		effort   providers.ReasoningEffort
		expected int64
		ok       bool
	}{
		{
			name:     "low effort",
			effort:   providers.ReasoningEffortLow,
			expected: 1024,
			ok:       true,
		},
		{
			name:     "medium effort",
			effort:   providers.ReasoningEffortMedium,
			expected: 4096,
			ok:       true,
		},
		{
			name:     "high effort",
			effort:   providers.ReasoningEffortHigh,
			expected: 16384,
			ok:       true,
		},
		{
			name:     "none effort",
			effort:   providers.ReasoningEffortNone,
			expected: 0,
			ok:       false,
		},
		{
			name:     "invalid effort",
			effort:   "invalid",
			expected: 0,
			ok:       false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			budget, ok := thinkingBudget(tc.effort)
			assert.Equal(t, tc.ok, ok)
			assert.Equal(t, tc.expected, budget)
		})
	}
}

// Integration tests - only run if API key is available.

func TestIntegrationCompletion(t *testing.T) {
	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: testutil.SimpleMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.ID)
	assert.Equal(t, "chat.completion", resp.Object)
	assert.Len(t, resp.Choices, 1)
	assert.NotEmpty(t, resp.Choices[0].Message.Content)
	assert.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
	assert.NotNil(t, resp.Usage)
	assert.Greater(t, resp.Usage.TotalTokens, 0)
}

func TestIntegrationCompletionWithSystemMessage(t *testing.T) {
	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: testutil.MessagesWithSystem(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.ID)
	assert.Len(t, resp.Choices, 1)
	assert.NotEmpty(t, resp.Choices[0].Message.Content)
}

func TestIntegrationCompletionStream(t *testing.T) {
	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: testutil.SimpleMessages(),
		Stream:   true,
	}

	chunks, errs := provider.CompletionStream(ctx, params)

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		assert.Equal(t, "chat.completion.chunk", chunk.Object)
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	assert.Greater(t, chunkCount, 0)
	assert.NotEmpty(t, content.String())
}

func TestIntegrationCompletionWithTools(t *testing.T) {
	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:      testutil.TestModel("anthropic"),
		Messages:   testutil.ToolCallMessages(),
		Tools:      []providers.Tool{testutil.WeatherTool()},
		ToolChoice: "auto",
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.ID)
	assert.Len(t, resp.Choices, 1)

	// The model should call the weather tool.
	if len(resp.Choices[0].Message.ToolCalls) > 0 {
		tc := resp.Choices[0].Message.ToolCalls[0]
		assert.Equal(t, "get_weather", tc.Function.Name)
		assert.Contains(t, strings.ToLower(tc.Function.Arguments), "paris")
		assert.Equal(t, providers.FinishReasonToolCalls, resp.Choices[0].FinishReason)
	}
}

func TestIntegrationCompletionWithToolsParallelDisabled(t *testing.T) {
	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	parallel := false
	ctx := context.Background()
	params := providers.CompletionParams{
		Model: testutil.TestModel("anthropic"),
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Get the weather in Paris and London"},
		},
		Tools:             []providers.Tool{testutil.WeatherTool()},
		ToolChoice:        "auto",
		ParallelToolCalls: &parallel,
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.ID)
	assert.Len(t, resp.Choices, 1)
}

func TestIntegrationCompletionConversation(t *testing.T) {
	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: testutil.ConversationMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.ID)
	assert.Len(t, resp.Choices, 1)

	// The model should remember the name "Alice".
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")
	assert.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestIntegrationCompletionReasoning(t *testing.T) {
	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	model := testutil.ReasoningModel("anthropic")
	if model == "" {
		t.Skip("No reasoning model configured for anthropic")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model: model,
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Please say hello! Think very briefly before you respond."},
		},
		ReasoningEffort: providers.ReasoningEffortLow,
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.ID)
	assert.Len(t, resp.Choices, 1)
	assert.NotEmpty(t, resp.Choices[0].Message.Content)

	// With reasoning effort, we should get reasoning content.
	if resp.Choices[0].Message.Reasoning != nil {
		assert.NotEmpty(t, resp.Choices[0].Message.Reasoning.Content)
	}
}

func TestIntegrationAgentLoop(t *testing.T) {
	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	// Start with the agent loop messages (user asks, assistant calls tool, tool returns).
	messages := testutil.AgentLoopMessages()

	params := providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: messages,
		Tools:    []providers.Tool{testutil.WeatherTool()},
	}

	// The model should respond with the weather information.
	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.ID)
	assert.Len(t, resp.Choices, 1)

	// Should have a content response (not another tool call).
	if contentStr, ok := resp.Choices[0].Message.Content.(string); ok && contentStr != "" {
		content := strings.ToLower(contentStr)
		// Should mention the weather or sunny.
		assert.True(
			t,
			strings.Contains(content, "sunny") || strings.Contains(content, "weather") ||
				strings.Contains(content, "salvaterra"),
		)
	}
}

func TestIntegrationAuthenticationError(t *testing.T) {
	provider, err := New(config.WithAPIKey("invalid-api-key"))
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    "claude-3-5-haiku-latest",
		Messages: testutil.SimpleMessages(),
	}

	_, err = provider.Completion(ctx, params)
	assert.Error(t, err)

	// Check that it's converted to an authentication error.
	var authErr *errors.AuthenticationError
	assert.ErrorAs(t, err, &authErr)
}
