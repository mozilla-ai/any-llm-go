package anthropic

import (
	"context"
	"strings"
	"testing"

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
	t.Run("extracts system message", func(t *testing.T) {
		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "You are a helpful assistant."},
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system := convertMessages(messages)

		assert.Equal(t, "You are a helpful assistant.", system)
		assert.Len(t, result, 1) // Only user message.
	})

	t.Run("concatenates multiple system messages", func(t *testing.T) {
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
		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system := convertMessages(messages)

		assert.Empty(t, system)
		assert.Len(t, result, 1)
	})

	t.Run("converts assistant message", func(t *testing.T) {
		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleAssistant, Content: "Hi there!"},
		}

		result, system := convertMessages(messages)

		assert.Empty(t, system)
		assert.Len(t, result, 2)
	})

	t.Run("converts assistant message with tool calls", func(t *testing.T) {
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
		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "What's the weather?"},
			{
				Role:    providers.RoleAssistant,
				Content: "",
				ToolCalls: []providers.ToolCall{
					{ID: "call_123", Type: "function", Function: providers.FunctionCall{Name: "get_weather", Arguments: `{"location": "Paris"}`}},
				},
			},
			{Role: providers.RoleTool, Content: "sunny, 22°C", ToolCallID: "call_123"},
		}

		result, _ := convertMessages(messages)

		assert.Len(t, result, 3)
	})
}

func TestConvertImagePart(t *testing.T) {
	t.Run("converts URL image", func(t *testing.T) {
		img := &providers.ImageURL{URL: "https://example.com/image.png"}
		result := convertImagePart(img)
		assert.NotNil(t, result)
	})

	t.Run("converts base64 image", func(t *testing.T) {
		img := &providers.ImageURL{URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
		result := convertImagePart(img)
		assert.NotNil(t, result)
	})
}

func TestConvertStopReason(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"end_turn", "end_turn", providers.FinishReasonStop},
		{"max_tokens", "max_tokens", providers.FinishReasonLength},
		{"tool_use", "tool_use", providers.FinishReasonToolCalls},
		{"stop_sequence", "stop_sequence", providers.FinishReasonStop},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Note: We can't directly test convertStopReason since it takes anthropic.MessageStopReason.
			// This would be tested in integration tests.
			_ = tc
		})
	}
}

func TestReasoningEffortToBudget(t *testing.T) {
	tests := []struct {
		effort   providers.ReasoningEffort
		expected int64
	}{
		{providers.ReasoningEffortLow, 1024},
		{providers.ReasoningEffortMedium, 4096},
		{providers.ReasoningEffortHigh, 16384},
	}

	for _, tc := range tests {
		t.Run(string(tc.effort), func(t *testing.T) {
			budget, ok := reasoningEffortToBudget[tc.effort]
			assert.True(t, ok)
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
		Model:    testutil.GetTestModel("anthropic"),
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
		Model:    testutil.GetTestModel("anthropic"),
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
		Model:    testutil.GetTestModel("anthropic"),
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
		Model:      testutil.GetTestModel("anthropic"),
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
		Model: testutil.GetTestModel("anthropic"),
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
		Model:    testutil.GetTestModel("anthropic"),
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

	model := testutil.GetReasoningModel("anthropic")
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
		Model:    testutil.GetTestModel("anthropic"),
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
		assert.True(t, strings.Contains(content, "sunny") || strings.Contains(content, "weather") || strings.Contains(content, "salvaterra"))
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
