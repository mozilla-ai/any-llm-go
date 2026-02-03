package anthropic

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
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
		require.NotNil(t, provider)
		require.Equal(t, "anthropic", provider.Name())
	})

	t.Run("creates provider from environment variable", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "env-api-key")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "")

		provider, err := New()
		require.Nil(t, provider)
		require.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
		require.Equal(t, "anthropic", missingKeyErr.Provider)
		require.Equal(t, "ANTHROPIC_API_KEY", missingKeyErr.EnvVar)
	})
}

func TestCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	caps := provider.Capabilities()

	require.True(t, caps.Completion)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionImage)
	require.True(t, caps.CompletionPDF)
	require.False(t, caps.Embedding) // Anthropic doesn't support embeddings.
	require.False(t, caps.ListModels)
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

		require.Equal(t, "You are a helpful assistant.", system)
		require.Len(t, result, 1) // Only user message.
	})

	t.Run("concatenates multiple system messages", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "First part."},
			{Role: providers.RoleSystem, Content: "Second part."},
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system := convertMessages(messages)

		require.Equal(t, "First part.\nSecond part.", system)
		require.Len(t, result, 1)
	})

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system := convertMessages(messages)

		require.Empty(t, system)
		require.Len(t, result, 1)
	})

	t.Run("converts assistant message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleAssistant, Content: "Hi there!"},
		}

		result, system := convertMessages(messages)

		require.Empty(t, system)
		require.Len(t, result, 2)
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

		require.Len(t, result, 2)
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

		require.Len(t, result, 3)
	})
}

func TestConvertImagePart(t *testing.T) {
	t.Parallel()

	t.Run("converts URL image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "https://example.com/image.png"}
		result := convertImagePart(img)
		require.NotNil(t, result)
	})

	t.Run("converts base64 image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
		result := convertImagePart(img)
		require.NotNil(t, result)
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
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestNewStreamState(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	require.NotNil(t, state)
	require.Equal(t, -1, state.currentToolIdx)
	require.Empty(t, state.messageID)
	require.Empty(t, state.model)
	require.Nil(t, state.toolCalls)
}

func TestStreamStateHandleTextDelta(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleTextDelta("Hello ")
	require.NotNil(t, chunk)
	require.Equal(t, "msg_123", chunk.ID)
	require.Equal(t, "claude-3", chunk.Model)
	require.Equal(t, "chat.completion.chunk", chunk.Object)
	require.Len(t, chunk.Choices, 1)
	require.Equal(t, "Hello ", chunk.Choices[0].Delta.Content)

	// Verify content is accumulated.
	chunk2 := state.handleTextDelta("world!")
	require.NotNil(t, chunk2)
	require.Equal(t, "world!", chunk2.Choices[0].Delta.Content)
	require.Equal(t, "Hello world!", state.content.String())
}

func TestStreamStateHandleThinkingDelta(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleThinkingDelta("Let me think...")
	require.NotNil(t, chunk)
	require.Equal(t, "msg_123", chunk.ID)
	require.Len(t, chunk.Choices, 1)
	require.NotNil(t, chunk.Choices[0].Delta.Reasoning)
	require.Equal(t, "Let me think...", chunk.Choices[0].Delta.Reasoning.Content)

	// Verify reasoning is accumulated.
	require.Equal(t, "Let me think...", state.reasoning.String())
}

func TestStreamStateHandleInputJSONDelta(t *testing.T) {
	t.Parallel()

	t.Run("returns nil when no tool calls", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		chunk := state.handleInputJSONDelta(`{"key":`)
		require.Nil(t, chunk)
	})

	t.Run("returns nil when tool index out of bounds", func(t *testing.T) {
		t.Parallel()

		state := newStreamState()
		state.currentToolIdx = 5 // Out of bounds.
		state.toolCalls = []providers.ToolCall{
			{ID: "call_1", Type: "function", Function: providers.FunctionCall{Name: "get_weather", Arguments: ""}},
		}
		chunk := state.handleInputJSONDelta(`{"key":`)
		require.Nil(t, chunk)
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
		require.Equal(t, `{"location":`, state.toolCalls[0].Function.Arguments)

		chunk2 := state.handleInputJSONDelta(`"Paris"}`)
		require.NotNil(t, chunk2)
		require.Equal(t, `{"location":"Paris"}`, state.toolCalls[0].Function.Arguments)
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
			require.Equal(t, tc.expectedMaxTokens, req.MaxTokens)
			if tc.expectThinking {
				require.NotNil(t, req.Thinking)
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
				require.Nil(t, result)
			} else {
				require.NotNil(t, result)
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
			require.Equal(t, tc.toolCall.ID, result.OfToolUse.ID)
			require.Equal(t, tc.toolCall.Function.Name, result.OfToolUse.Name)
			require.Equal(t, "tool_use", string(result.OfToolUse.Type))
			if tc.expectInput {
				require.NotNil(t, result.OfToolUse.Input)
			} else {
				require.Nil(t, result.OfToolUse.Input)
			}
		})
	}
}

func TestConvertTool(t *testing.T) {
	t.Parallel()

	t.Run("converts tool with properties and required fields", func(t *testing.T) {
		t.Parallel()

		tool := testutil.WeatherTool()
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "get_weather", result.OfTool.Name)
		require.Equal(t, "Get the current weather for a location.", result.OfTool.Description.Value)
		require.Equal(t, "object", string(result.OfTool.InputSchema.Type))

		// Verify properties are preserved.
		props, ok := result.OfTool.InputSchema.Properties.(map[string]any)
		require.True(t, ok, "properties should be a map")
		require.Contains(t, props, "location")

		locationProp, ok := props["location"].(map[string]any)
		require.True(t, ok, "location property should be a map")
		require.Equal(t, "string", locationProp["type"])
		require.Equal(t, "The city name, e.g. 'Paris, France'", locationProp["description"])

		// Verify required fields are preserved.
		require.Contains(t, result.OfTool.InputSchema.Required, "location")
	})

	t.Run("converts tool with multiple parameters", func(t *testing.T) {
		t.Parallel()

		tool := testutil.NewTestCalculatorTool(t)
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "calculate", result.OfTool.Name)
		require.Equal(t, "object", string(result.OfTool.InputSchema.Type))

		// Verify all properties are preserved.
		props, ok := result.OfTool.InputSchema.Properties.(map[string]any)
		require.True(t, ok, "properties should be a map")
		require.Contains(t, props, "a")
		require.Contains(t, props, "b")
		require.Contains(t, props, "operation")

		// Verify property types.
		aProp, ok := props["a"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "number", aProp["type"])

		bProp, ok := props["b"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "number", bProp["type"])

		opProp, ok := props["operation"].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "string", opProp["type"])

		// Verify enum values are preserved.
		enum, ok := opProp["enum"].([]string)
		require.True(t, ok, "enum should be a string slice")
		require.ElementsMatch(t, []string{"add", "subtract", "multiply", "divide"}, enum)

		// Verify all required fields are preserved.
		require.Len(t, result.OfTool.InputSchema.Required, 3)
		require.Contains(t, result.OfTool.InputSchema.Required, "a")
		require.Contains(t, result.OfTool.InputSchema.Required, "b")
		require.Contains(t, result.OfTool.InputSchema.Required, "operation")
	})

	t.Run("converts tool with no required fields", func(t *testing.T) {
		t.Parallel()

		tool := providers.Tool{
			Type: "function",
			Function: providers.Function{
				Name:        "optional_params",
				Description: "A tool with optional parameters.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"optional_field": map[string]any{
							"type":        "string",
							"description": "An optional field",
						},
					},
					// No "required" field.
				},
			},
		}
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "optional_params", result.OfTool.Name)
		require.Empty(t, result.OfTool.InputSchema.Required)
	})

	t.Run("converts tool with empty parameters", func(t *testing.T) {
		t.Parallel()

		tool := testutil.DateTool()
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "get_current_date", result.OfTool.Name)
		require.Equal(t, "object", string(result.OfTool.InputSchema.Type))
	})

	t.Run("returns error for invalid required field type", func(t *testing.T) {
		t.Parallel()

		tool := providers.Tool{
			Type: "function",
			Function: providers.Function{
				Name:        "bad_tool",
				Description: "A tool with invalid required field.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
					"required":   123, // Invalid type.
				},
			},
		}
		_, err := convertTool(tool)

		require.Error(t, err)
		require.Contains(t, err.Error(), "bad_tool")
		require.Contains(t, err.Error(), "invalid required field")
	})

	t.Run("returns error for non-string element in required array", func(t *testing.T) {
		t.Parallel()

		tool := providers.Tool{
			Type: "function",
			Function: providers.Function{
				Name:        "mixed_required",
				Description: "A tool with mixed types in required.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{},
					"required":   []any{"valid", 42, "also_valid"}, // Mixed types.
				},
			},
		}
		_, err := convertTool(tool)

		require.Error(t, err)
		require.Contains(t, err.Error(), "mixed_required")
		require.Contains(t, err.Error(), "element 1")
	})
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
			require.Equal(t, tc.ok, ok)
			require.Equal(t, tc.expected, budget)
		})
	}
}

func TestToStringSlice(t *testing.T) {
	t.Parallel()

	t.Run("returns []string input unchanged", func(t *testing.T) {
		t.Parallel()

		input := []string{"a", "b", "c"}
		result, err := toStringSlice(input)

		require.NoError(t, err)
		require.Equal(t, input, result)
	})

	t.Run("converts []any with all strings", func(t *testing.T) {
		t.Parallel()

		input := []any{"x", "y", "z"}
		result, err := toStringSlice(input)

		require.NoError(t, err)
		require.Equal(t, []string{"x", "y", "z"}, result)
	})

	t.Run("converts empty []any to empty []string", func(t *testing.T) {
		t.Parallel()

		input := []any{}
		result, err := toStringSlice(input)

		require.NoError(t, err)
		require.Empty(t, result)
	})

	t.Run("returns error for []any with non-string element", func(t *testing.T) {
		t.Parallel()

		input := []any{"valid", 42, "another"}
		_, err := toStringSlice(input)

		require.Error(t, err)
		require.Contains(t, err.Error(), "element 1")
		require.Contains(t, err.Error(), "expected string")
		require.Contains(t, err.Error(), "int")
	})

	t.Run("returns error for unexpected type", func(t *testing.T) {
		t.Parallel()

		_, err := toStringSlice(123)

		require.Error(t, err)
		require.Contains(t, err.Error(), "expected []string or []any")
		require.Contains(t, err.Error(), "int")
	})

	t.Run("returns error for nil input", func(t *testing.T) {
		t.Parallel()

		_, err := toStringSlice(nil)

		require.Error(t, err)
		require.Contains(t, err.Error(), "expected []string or []any")
	})
}

// Integration tests - only run if API key is available.

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Equal(t, "chat.completion", resp.Object)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
	require.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
	require.NotNil(t, resp.Usage)
	require.Greater(t, resp.Usage.TotalTokens, 0)
}

func TestIntegrationCompletionWithSystemMessage(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
}

func TestIntegrationCompletionStream(t *testing.T) {
	t.Parallel()

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
		require.Equal(t, "chat.completion.chunk", chunk.Object)
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	require.Greater(t, chunkCount, 0)
	require.NotEmpty(t, content.String())
}

func TestIntegrationCompletionWithTools(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model should call the weather tool.
	if len(resp.Choices[0].Message.ToolCalls) > 0 {
		tc := resp.Choices[0].Message.ToolCalls[0]
		require.Equal(t, "get_weather", tc.Function.Name)
		require.Contains(t, strings.ToLower(tc.Function.Arguments), "paris")
		require.Equal(t, providers.FinishReasonToolCalls, resp.Choices[0].FinishReason)
	}
}

func TestIntegrationCompletionWithToolsParallelDisabled(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
}

func TestIntegrationAgentLoop(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	tools := []providers.Tool{testutil.WeatherTool()}

	// Step 1: Send initial message asking about weather.
	messages := []providers.Message{
		{Role: providers.RoleUser, Content: "What is the weather in Paris? Use the get_weather tool."},
	}

	resp, err := provider.Completion(ctx, providers.CompletionParams{
		Model:      testutil.TestModel("anthropic"),
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
	})
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)

	// Step 2: Verify the model called the tool.
	require.NotEmpty(t, resp.Choices[0].Message.ToolCalls, "expected model to call get_weather tool")
	require.Equal(t, providers.FinishReasonToolCalls, resp.Choices[0].FinishReason)

	tc := resp.Choices[0].Message.ToolCalls[0]
	require.Equal(t, "get_weather", tc.Function.Name)
	require.NotEmpty(t, tc.ID)

	// Step 3: Parse the arguments - this verifies parameters were sent correctly.
	var args struct {
		Location string `json:"location"`
	}
	err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
	require.NoError(t, err, "tool arguments should be valid JSON")
	require.NotEmpty(t, args.Location, "location argument should be present")
	require.Contains(t, strings.ToLower(args.Location), "paris")

	// Step 4: Add assistant message with tool call and tool result.
	messages = append(messages, resp.Choices[0].Message)
	messages = append(messages, providers.Message{
		Role:       providers.RoleTool,
		Content:    testutil.MockWeatherResult(t, args.Location),
		ToolCallID: tc.ID,
	})

	// Step 5: Continue conversation with tool result.
	resp, err = provider.Completion(ctx, providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: messages,
		Tools:    tools,
	})
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)

	// Step 6: Verify the model produced a final response.
	require.Equal(t, providers.FinishReasonStop, resp.Choices[0].FinishReason)
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content in final response")
	require.NotEmpty(t, contentStr)
}

func TestIntegrationAgentLoopMultipleParams(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	tools := []providers.Tool{testutil.NewTestCalculatorTool(t)}

	// Ask the model to use the calculator with specific values.
	messages := []providers.Message{
		{Role: providers.RoleUser, Content: "Use the calculate tool to add 15 and 27 together."},
	}

	resp, err := provider.Completion(ctx, providers.CompletionParams{
		Model:      testutil.TestModel("anthropic"),
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
	})
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)

	// Verify the model called the tool with correct parameters.
	require.NotEmpty(t, resp.Choices[0].Message.ToolCalls, "expected model to call calculate tool")

	tc := resp.Choices[0].Message.ToolCalls[0]
	require.Equal(t, "calculate", tc.Function.Name)

	// Parse and verify all required parameters are present.
	var args struct {
		A         float64 `json:"a"`
		B         float64 `json:"b"`
		Operation string  `json:"operation"`
	}
	err = json.Unmarshal([]byte(tc.Function.Arguments), &args)
	require.NoError(t, err, "tool arguments should be valid JSON")

	// Verify the parameters - this catches "wrong order" bugs.
	require.Equal(t, 15.0, args.A, "first operand should be 15")
	require.Equal(t, 27.0, args.B, "second operand should be 27")
	require.Equal(t, "add", args.Operation, "operation should be 'add'")

	// Complete the agent loop with tool result.
	messages = append(messages, resp.Choices[0].Message)
	messages = append(messages, providers.Message{
		Role:       providers.RoleTool,
		Content:    testutil.MockCalculatorResult(t, args.A, args.B, args.Operation),
		ToolCallID: tc.ID,
	})

	resp, err = provider.Completion(ctx, providers.CompletionParams{
		Model:    testutil.TestModel("anthropic"),
		Messages: messages,
		Tools:    tools,
	})
	require.NoError(t, err)

	// Verify final response mentions the result.
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok)
	require.Contains(t, contentStr, "42")
}

func TestIntegrationCompletionConversation(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model should remember the name "Alice".
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")
	require.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestIntegrationCompletionReasoning(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)

	// With reasoning effort, we should get reasoning content.
	if resp.Choices[0].Message.Reasoning != nil {
		require.NotEmpty(t, resp.Choices[0].Message.Reasoning.Content)
	}
}

func TestIntegrationAgentLoopContinuation(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// Should have a content response (not another tool call).
	if contentStr, ok := resp.Choices[0].Message.Content.(string); ok && contentStr != "" {
		content := strings.ToLower(contentStr)
		// Should mention the weather or sunny.
		require.True(
			t,
			strings.Contains(content, "sunny") || strings.Contains(content, "weather") ||
				strings.Contains(content, "salvaterra"),
		)
	}
}

func TestIntegrationAuthenticationError(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("invalid-api-key"))
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    "claude-3-5-haiku-latest",
		Messages: testutil.SimpleMessages(),
	}

	_, err = provider.Completion(ctx, params)
	require.Error(t, err)

	// Check that it's converted to an authentication error.
	var authErr *errors.AuthenticationError
	require.ErrorAs(t, err, &authErr)
}

func TestConvertError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		err          error
		wantSentinel error
	}{
		{
			name:         "nil error returns nil",
			err:          nil,
			wantSentinel: nil,
		},
		{
			name:         "non-API error becomes ProviderError",
			err:          stderrors.New("network timeout"),
			wantSentinel: errors.ErrProvider,
		},
		{
			name:         "401 status becomes AuthenticationError",
			err:          newTestAPIError(t, 401),
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "429 status becomes RateLimitError",
			err:          newTestAPIError(t, 429),
			wantSentinel: errors.ErrRateLimit,
		},
		{
			name:         "404 status becomes ModelNotFoundError",
			err:          newTestAPIError(t, 404),
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "400 status becomes InvalidRequestError",
			err:          newTestAPIError(t, 400),
			wantSentinel: errors.ErrInvalidRequest,
		},
		{
			name:         "403 status becomes AuthenticationError",
			err:          newTestAPIError(t, 403),
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "500 status becomes ProviderError",
			err:          newTestAPIError(t, 500),
			wantSentinel: errors.ErrProvider,
		},
		{
			name:         "502 status becomes ProviderError",
			err:          newTestAPIError(t, 502),
			wantSentinel: errors.ErrProvider,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := &Provider{}
			result := p.ConvertError(tc.err)

			if tc.wantSentinel == nil {
				require.Nil(t, result)
				return
			}

			require.NotNil(t, result)
			require.True(t, stderrors.Is(result, tc.wantSentinel), "expected error to match %v", tc.wantSentinel)

			// Verify the provider name is set in the error message.
			require.Contains(t, result.Error(), "["+providerName+"]")
		})
	}
}

// newTestAPIError creates an Anthropic API error for testing.
// Note: The raw JSON field is unexported, so we can only test status code based conversion.
func newTestAPIError(t *testing.T, statusCode int) *anthropic.Error {
	t.Helper()

	testURL, _ := url.Parse("https://api.anthropic.com/v1/messages")
	return &anthropic.Error{
		StatusCode: statusCode,
		RequestID:  "req_test123",
		Request:    &http.Request{Method: "POST", URL: testURL},
		Response:   &http.Response{StatusCode: statusCode},
	}
}
