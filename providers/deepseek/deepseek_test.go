package deepseek

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
			config.WithBaseURL("https://custom.deepseek.com"),
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
	require.True(t, caps.CompletionReasoning)
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

func TestPreprocessParams(t *testing.T) {
	t.Parallel()

	t.Run("passes through params without response format", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:    "deepseek-chat",
			Messages: testutil.SimpleMessages(),
		}

		result := preprocessParams(params)

		require.Equal(t, params.Model, result.Model)
		require.Equal(t, params.Messages, result.Messages)
		require.Nil(t, result.ResponseFormat)
	})

	t.Run("passes through json_object format unchanged", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:    "deepseek-chat",
			Messages: testutil.SimpleMessages(),
			ResponseFormat: &providers.ResponseFormat{
				Type: responseFormatJSONObject,
			},
		}

		result := preprocessParams(params)

		require.Equal(t, responseFormatJSONObject, result.ResponseFormat.Type)
		require.Equal(t, params.Messages, result.Messages)
	})

	t.Run("converts json_schema to json_object with embedded schema", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model: "deepseek-chat",
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "What is 2+2?"},
			},
			ResponseFormat: &providers.ResponseFormat{
				Type: responseFormatJSONSchema,
				JSONSchema: &providers.JSONSchema{
					Name: "math_response",
					Schema: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"answer": map[string]any{
								"type": "integer",
							},
						},
					},
				},
			},
		}

		result := preprocessParams(params)

		// Should be converted to json_object.
		require.Equal(t, responseFormatJSONObject, result.ResponseFormat.Type)
		require.Nil(t, result.ResponseFormat.JSONSchema)

		// Message should contain the schema.
		require.Len(t, result.Messages, 1)
		content := result.Messages[0].ContentString()
		require.Contains(t, content, "JSON")
		require.Contains(t, content, "schema")
		require.Contains(t, content, "What is 2+2?")
	})

	t.Run("preserves other params when converting", func(t *testing.T) {
		t.Parallel()

		temp := 0.7
		maxTokens := 100
		params := providers.CompletionParams{
			Model: "deepseek-chat",
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Test"},
			},
			Temperature: &temp,
			MaxTokens:   &maxTokens,
			ResponseFormat: &providers.ResponseFormat{
				Type: responseFormatJSONSchema,
				JSONSchema: &providers.JSONSchema{
					Name:   "test",
					Schema: map[string]any{"type": "object"},
				},
			},
		}

		result := preprocessParams(params)

		require.Equal(t, params.Model, result.Model)
		require.Equal(t, params.Temperature, result.Temperature)
		require.Equal(t, params.MaxTokens, result.MaxTokens)
	})

	t.Run("returns original params when no user message for schema injection", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model: "deepseek-chat",
			Messages: []providers.Message{
				{Role: providers.RoleSystem, Content: "You are helpful."},
			},
			ResponseFormat: &providers.ResponseFormat{
				Type: responseFormatJSONSchema,
				JSONSchema: &providers.JSONSchema{
					Name:   "test",
					Schema: map[string]any{"type": "object"},
				},
			},
		}

		result := preprocessParams(params)

		// Should return original params unchanged since injection failed.
		require.Equal(t, responseFormatJSONSchema, result.ResponseFormat.Type)
		require.NotNil(t, result.ResponseFormat.JSONSchema)
	})

	t.Run("returns original params when user message is multimodal", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model: "deepseek-chat",
			Messages: []providers.Message{
				{
					Role: providers.RoleUser,
					Content: []providers.ContentPart{
						{Type: "text", Text: "What is this?"},
						{Type: "image_url", ImageURL: &providers.ImageURL{URL: "https://example.com/img.png"}},
					},
				},
			},
			ResponseFormat: &providers.ResponseFormat{
				Type: responseFormatJSONSchema,
				JSONSchema: &providers.JSONSchema{
					Name:   "test",
					Schema: map[string]any{"type": "object"},
				},
			},
		}

		result := preprocessParams(params)

		// Should return original params unchanged since multimodal content can't be modified.
		require.Equal(t, responseFormatJSONSchema, result.ResponseFormat.Type)
		require.NotNil(t, result.ResponseFormat.JSONSchema)
	})
}

func TestPreprocessMessagesForJSONSchema(t *testing.T) {
	t.Parallel()

	t.Run("injects schema into last user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "You are helpful."},
			{Role: providers.RoleUser, Content: "What is 2+2?"},
		}
		schema := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"answer": map[string]any{"type": "integer"},
			},
		}

		result, ok := preprocessMessagesForJSONSchema(messages, schema)

		require.True(t, ok)
		require.Len(t, result, 2)
		// System message unchanged.
		require.Equal(t, "You are helpful.", result[0].ContentString())
		// User message modified.
		content := result[1].ContentString()
		require.Contains(t, content, "JSON")
		require.Contains(t, content, "answer")
		require.Contains(t, content, "What is 2+2?")
	})

	t.Run("handles conversation with multiple user messages", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleAssistant, Content: "Hi there!"},
			{Role: providers.RoleUser, Content: "Give me a number."},
		}
		schema := map[string]any{"type": "object"}

		result, ok := preprocessMessagesForJSONSchema(messages, schema)

		require.True(t, ok)
		require.Len(t, result, 3)
		// First user message unchanged.
		require.Equal(t, "Hello", result[0].ContentString())
		// Assistant message unchanged.
		require.Equal(t, "Hi there!", result[1].ContentString())
		// Last user message modified.
		require.Contains(t, result[2].ContentString(), "JSON")
	})

	t.Run("returns false if no user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "System"},
		}
		schema := map[string]any{"type": "object"}

		result, ok := preprocessMessagesForJSONSchema(messages, schema)

		require.False(t, ok)
		require.Equal(t, messages, result)
	})

	t.Run("returns false for multimodal content", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{
				Role: providers.RoleUser,
				Content: []providers.ContentPart{
					{Type: "text", Text: "What is this?"},
					{Type: "image_url", ImageURL: &providers.ImageURL{URL: "https://example.com/img.png"}},
				},
			},
		}
		schema := map[string]any{"type": "object"}

		result, ok := preprocessMessagesForJSONSchema(messages, schema)

		require.False(t, ok)
		require.Equal(t, messages, result)
	})

	t.Run("does not mutate original messages", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Original content"},
		}
		schema := map[string]any{"type": "object"}

		// Return values intentionally ignored; we only verify the original isn't mutated.
		_, _ = preprocessMessagesForJSONSchema(messages, schema)

		// Original should be unchanged.
		require.Equal(t, "Original content", messages[0].ContentString())
	})

	t.Run("preserves Reasoning field", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{
				Role:      providers.RoleUser,
				Content:   "What is 2+2?",
				Reasoning: &providers.Reasoning{Content: "thinking..."},
			},
		}
		schema := map[string]any{"type": "object"}

		result, ok := preprocessMessagesForJSONSchema(messages, schema)

		require.True(t, ok)
		require.NotNil(t, result[0].Reasoning)
		require.Equal(t, "thinking...", result[0].Reasoning.Content)
	})
}

// Integration tests - only run if DeepSeek API key is available.

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("DEEPSEEK_API_KEY not set")
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
		t.Skip("DEEPSEEK_API_KEY not set")
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
		t.Skip("DEEPSEEK_API_KEY not set")
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
		t.Skip("DEEPSEEK_API_KEY not set")
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
		t.Skip("DEEPSEEK_API_KEY not set")
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
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")
	require.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestIntegrationJSONSchema(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("DEEPSEEK_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model: testutil.TestModel(providerName),
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "What is 2+2? Give the answer as an integer."},
		},
		ResponseFormat: &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:        "math_response",
				Description: "A mathematical response",
				Schema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"answer": map[string]any{
							"type":        "integer",
							"description": "The numerical answer",
						},
					},
					"required": []string{"answer"},
				},
			},
		},
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// Response should be valid JSON containing "answer".
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")
	require.Contains(t, contentStr, "answer")
}

func TestIntegrationCompletionWithTools(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("DEEPSEEK_API_KEY not set")
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
		t.Skip("DEEPSEEK_API_KEY not set")
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
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content in final response")
	require.NotEmpty(t, contentStr)
}

func TestIntegrationAgentLoopMultipleParams(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("DEEPSEEK_API_KEY not set")
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
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok)
	require.Contains(t, contentStr, "42")
}
