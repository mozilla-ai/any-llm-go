package groq

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestNew(t *testing.T) {
	// Note: Not using t.Parallel() here because child test uses t.Setenv.

	t.Run("creates provider with API key", func(t *testing.T) {
		t.Parallel()

		provider, err := New(config.WithAPIKey("test-key"))
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, providerName, provider.Name())
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv(envAPIKey, "")

		provider, err := New()
		require.Nil(t, provider)
		require.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
		require.Equal(t, providerName, missingKeyErr.Provider)
		require.Equal(t, envAPIKey, missingKeyErr.EnvVar)
	})

	t.Run("creates provider with custom base URL", func(t *testing.T) {
		t.Parallel()

		provider, err := New(
			config.WithAPIKey("test-key"),
			config.WithBaseURL("https://custom.groq.com/v1"),
		)
		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

func TestCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	caps := provider.Capabilities()

	require.True(t, caps.Completion)
	require.True(t, caps.CompletionStreaming)
	require.False(t, caps.CompletionReasoning)
	require.False(t, caps.CompletionImage)
	require.False(t, caps.CompletionPDF)
	require.False(t, caps.Embedding)
	require.True(t, caps.ListModels)
}

func TestProviderName(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)
	require.Equal(t, providerName, provider.Name())
}

// skipIfToolUseFailed skips the test if the error is a Groq tool_use_failed error.
// Groq sometimes returns this when the model generates a malformed tool call (e.g., XML
// format instead of JSON). The error confirms the model attempted tool use, so the
// integration is working correctly - Groq just couldn't parse the model's output.
func skipIfToolUseFailed(t *testing.T, err error) {
	t.Helper()

	if err != nil && strings.Contains(err.Error(), "tool_use_failed") {
		t.Skipf("Groq tool_use_failed (model attempted tool call with malformed output): %v", err)
	}
}

// Integration tests - only run if Groq API key is available.

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GROQ_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.SimpleMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Equal(t, objectChatCompletion, resp.Object)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
	require.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
}

func TestIntegrationCompletionWithSystemMessage(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GROQ_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
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

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GROQ_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
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

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GROQ_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.ListModels(ctx)
	require.NoError(t, err)

	require.Equal(t, objectList, resp.Object)
	require.NotEmpty(t, resp.Data)
}

func TestIntegrationCompletionConversation(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GROQ_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.ConversationMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model should remember the name "Alice".
	require.Contains(t, strings.ToLower(resp.Choices[0].Message.ContentString()), "alice")
}

func TestIntegrationCompletionWithTools(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GROQ_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:      testutil.TestModel(providerName),
		Messages:   testutil.ToolCallMessages(),
		Tools:      []providers.Tool{testutil.WeatherTool()},
		ToolChoice: "auto",
	}

	resp, err := provider.Completion(ctx, params)
	skipIfToolUseFailed(t, err)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// Should either have tool calls or content.
	choice := resp.Choices[0]
	hasToolCalls := len(choice.Message.ToolCalls) > 0
	hasContent := choice.Message.ContentString() != ""
	require.True(t, hasToolCalls || hasContent, "Expected tool calls or content")

	if hasToolCalls {
		require.Equal(t, "get_weather", choice.Message.ToolCalls[0].Function.Name)
	}
}

func TestIntegrationAgentLoop(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GROQ_API_KEY not set")
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
		Model:      testutil.TestModel(providerName),
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
	})
	skipIfToolUseFailed(t, err)
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
		Model:    testutil.TestModel(providerName),
		Messages: messages,
		Tools:    tools,
	})
	require.NoError(t, err)
	require.Len(t, resp.Choices, 1)

	// Step 6: Verify the model produced a final response.
	require.Equal(t, providers.FinishReasonStop, resp.Choices[0].FinishReason)
	require.NotEmpty(t, resp.Choices[0].Message.ContentString())
}

func TestIntegrationAgentLoopMultipleParams(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GROQ_API_KEY not set")
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
		Model:      testutil.TestModel(providerName),
		Messages:   messages,
		Tools:      tools,
		ToolChoice: "auto",
	})
	skipIfToolUseFailed(t, err)
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
		Model:    testutil.TestModel(providerName),
		Messages: messages,
		Tools:    tools,
	})
	require.NoError(t, err)

	// Verify final response mentions the result.
	require.Contains(t, resp.Choices[0].Message.ContentString(), "42")
}
