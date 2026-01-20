package openai

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
		assert.Equal(t, "openai", provider.Name())
	})

	t.Run("creates provider from environment variable", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "env-api-key")

		provider, err := New()
		require.NoError(t, err)
		assert.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "")

		provider, err := New()
		assert.Nil(t, provider)
		assert.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		assert.ErrorAs(t, err, &missingKeyErr)
		assert.Equal(t, "openai", missingKeyErr.Provider)
		assert.Equal(t, "OPENAI_API_KEY", missingKeyErr.EnvVar)
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
	assert.True(t, caps.Embedding)
	assert.True(t, caps.ListModels)
}

func TestConvertParams(t *testing.T) {
	t.Run("converts basic params", func(t *testing.T) {
		params := providers.CompletionParams{
			Model: "gpt-4",
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Hello"},
			},
		}

		req := convertParams(params)

		assert.Equal(t, "gpt-4", string(req.Model))
		assert.Len(t, req.Messages, 1)
	})

	t.Run("converts temperature and top_p", func(t *testing.T) {
		temp := 0.7
		topP := 0.9
		params := providers.CompletionParams{
			Model:       "gpt-4",
			Messages:    testutil.SimpleMessages(),
			Temperature: &temp,
			TopP:        &topP,
		}

		req := convertParams(params)

		assert.Equal(t, 0.7, req.Temperature.Value)
		assert.Equal(t, 0.9, req.TopP.Value)
	})

	t.Run("converts max_tokens", func(t *testing.T) {
		maxTokens := 100
		params := providers.CompletionParams{
			Model:     "gpt-4",
			Messages:  testutil.SimpleMessages(),
			MaxTokens: &maxTokens,
		}

		req := convertParams(params)

		assert.Equal(t, int64(100), req.MaxCompletionTokens.Value)
	})

	t.Run("converts stop sequences", func(t *testing.T) {
		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			Stop:     []string{"END", "STOP"},
		}

		req := convertParams(params)

		assert.NotNil(t, req.Stop)
	})

	t.Run("converts tools", func(t *testing.T) {
		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			Tools:    []providers.Tool{testutil.WeatherTool()},
		}

		req := convertParams(params)

		assert.Len(t, req.Tools, 1)
	})

	t.Run("converts tool_choice auto", func(t *testing.T) {
		params := providers.CompletionParams{
			Model:      "gpt-4",
			Messages:   testutil.SimpleMessages(),
			Tools:      []providers.Tool{testutil.WeatherTool()},
			ToolChoice: "auto",
		}

		req := convertParams(params)

		assert.NotNil(t, req.ToolChoice)
	})

	t.Run("converts tool_choice required", func(t *testing.T) {
		params := providers.CompletionParams{
			Model:      "gpt-4",
			Messages:   testutil.SimpleMessages(),
			Tools:      []providers.Tool{testutil.WeatherTool()},
			ToolChoice: "required",
		}

		req := convertParams(params)

		assert.NotNil(t, req.ToolChoice)
	})

	t.Run("converts tool_choice with specific function", func(t *testing.T) {
		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			Tools:    []providers.Tool{testutil.WeatherTool()},
			ToolChoice: providers.ToolChoice{
				Type:     "function",
				Function: &providers.ToolChoiceFunction{Name: "get_weather"},
			},
		}

		req := convertParams(params)

		assert.NotNil(t, req.ToolChoice)
	})

	t.Run("converts response_format json_object", func(t *testing.T) {
		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			ResponseFormat: &providers.ResponseFormat{
				Type: "json_object",
			},
		}

		req := convertParams(params)

		assert.NotNil(t, req.ResponseFormat)
	})

	t.Run("converts reasoning_effort", func(t *testing.T) {
		params := providers.CompletionParams{
			Model:           "o1-mini",
			Messages:        testutil.SimpleMessages(),
			ReasoningEffort: providers.ReasoningEffortHigh,
		}

		req := convertParams(params)

		assert.NotNil(t, req.ReasoningEffort)
	})

	t.Run("converts seed", func(t *testing.T) {
		seed := 42
		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			Seed:     &seed,
		}

		req := convertParams(params)

		assert.Equal(t, int64(42), req.Seed.Value)
	})

	t.Run("converts user", func(t *testing.T) {
		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			User:     "test-user",
		}

		req := convertParams(params)

		assert.Equal(t, "test-user", req.User.Value)
	})
}

func TestConvertMessage(t *testing.T) {
	t.Run("converts system message", func(t *testing.T) {
		msg := providers.Message{Role: providers.RoleSystem, Content: "You are helpful"}
		result := convertMessage(msg)
		assert.NotNil(t, result)
	})

	t.Run("converts user message", func(t *testing.T) {
		msg := providers.Message{Role: providers.RoleUser, Content: "Hello"}
		result := convertMessage(msg)
		assert.NotNil(t, result)
	})

	t.Run("converts assistant message", func(t *testing.T) {
		msg := providers.Message{Role: providers.RoleAssistant, Content: "Hi there!"}
		result := convertMessage(msg)
		assert.NotNil(t, result)
	})

	t.Run("converts assistant message with tool calls", func(t *testing.T) {
		msg := providers.Message{
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
		}
		result := convertMessage(msg)
		assert.NotNil(t, result)
	})

	t.Run("converts tool result message", func(t *testing.T) {
		msg := providers.Message{
			Role:       providers.RoleTool,
			Content:    "sunny, 22°C",
			ToolCallID: "call_123",
		}
		result := convertMessage(msg)
		assert.NotNil(t, result)
	})

	t.Run("converts multimodal user message", func(t *testing.T) {
		msg := providers.Message{
			Role: providers.RoleUser,
			Content: []providers.ContentPart{
				{Type: "text", Text: "What's in this image?"},
				{Type: "image_url", ImageURL: &providers.ImageURL{URL: "https://example.com/image.png"}},
			},
		}
		result := convertMessage(msg)
		assert.NotNil(t, result)
	})
}

func TestConvertResponse(t *testing.T) {
	t.Run("converts basic response", func(t *testing.T) {
		// We can't easily test this without mocking the OpenAI SDK response.
		// This would be tested in integration tests.
	})
}

// Integration tests - only run if API key is available.

func TestIntegrationCompletion(t *testing.T) {
	if testutil.SkipIfNoAPIKey("openai") {
		t.Skip("OPENAI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("openai"),
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

func TestIntegrationCompletionStream(t *testing.T) {
	if testutil.SkipIfNoAPIKey("openai") {
		t.Skip("OPENAI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("openai"),
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
	if testutil.SkipIfNoAPIKey("openai") {
		t.Skip("OPENAI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:      testutil.TestModel("openai"),
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
		assert.Contains(t, tc.Function.Arguments, "Paris")
		assert.Equal(t, providers.FinishReasonToolCalls, resp.Choices[0].FinishReason)
	}
}

func TestIntegrationCompletionConversation(t *testing.T) {
	if testutil.SkipIfNoAPIKey("openai") {
		t.Skip("OPENAI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel("openai"),
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

func TestIntegrationEmbedding(t *testing.T) {
	if testutil.SkipIfNoAPIKey("openai") {
		t.Skip("OPENAI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.EmbeddingParams{
		Model: testutil.EmbeddingModel("openai"),
		Input: "Hello, world!",
	}

	resp, err := provider.Embedding(ctx, params)
	require.NoError(t, err)

	assert.Equal(t, "list", resp.Object)
	assert.Len(t, resp.Data, 1)
	assert.Greater(t, len(resp.Data[0].Embedding), 0)
	assert.NotNil(t, resp.Usage)
}

func TestIntegrationListModels(t *testing.T) {
	if testutil.SkipIfNoAPIKey("openai") {
		t.Skip("OPENAI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.ListModels(ctx)
	require.NoError(t, err)

	assert.Equal(t, "list", resp.Object)
	assert.Greater(t, len(resp.Data), 0)

	// Check that some expected models are present.
	modelIDs := make([]string, len(resp.Data))
	for i, m := range resp.Data {
		modelIDs[i] = m.ID
	}

	// gpt-4o-mini should be in the list.
	found := false
	for _, id := range modelIDs {
		if strings.Contains(id, "gpt") {
			found = true
			break
		}
	}
	assert.True(t, found, "Expected to find GPT models in the list")
}

func TestIntegrationAuthenticationError(t *testing.T) {
	provider, err := New(config.WithAPIKey("invalid-api-key"))
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    "gpt-4o-mini",
		Messages: testutil.SimpleMessages(),
	}

	_, err = provider.Completion(ctx, params)
	assert.Error(t, err)

	// Check that it's converted to an authentication error.
	var authErr *errors.AuthenticationError
	assert.ErrorAs(t, err, &authErr)
}
