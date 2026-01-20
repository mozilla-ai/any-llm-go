package openai

import (
	"context"
	stderrors "errors"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/openai/openai-go"
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
		require.Equal(t, "openai", provider.Name())
	})

	t.Run("creates provider from environment variable", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "env-api-key")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "")

		provider, err := New()
		require.Nil(t, provider)
		require.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
		require.Equal(t, "openai", missingKeyErr.Provider)
		require.Equal(t, "OPENAI_API_KEY", missingKeyErr.EnvVar)
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
	require.True(t, caps.Embedding)
	require.True(t, caps.ListModels)
}

func TestConvertParams(t *testing.T) {
	t.Parallel()

	t.Run("converts basic params", func(t *testing.T) {
		t.Parallel()
		params := providers.CompletionParams{
			Model: "gpt-4",
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Hello"},
			},
		}

		req := convertParams(params)

		require.Equal(t, "gpt-4", string(req.Model))
		require.Len(t, req.Messages, 1)
	})

	t.Run("converts temperature and top_p", func(t *testing.T) {
		t.Parallel()

		temp := 0.7
		topP := 0.9
		params := providers.CompletionParams{
			Model:       "gpt-4",
			Messages:    testutil.SimpleMessages(),
			Temperature: &temp,
			TopP:        &topP,
		}

		req := convertParams(params)

		require.Equal(t, 0.7, req.Temperature.Value)
		require.Equal(t, 0.9, req.TopP.Value)
	})

	t.Run("converts max_tokens", func(t *testing.T) {
		t.Parallel()

		maxTokens := 100
		params := providers.CompletionParams{
			Model:     "gpt-4",
			Messages:  testutil.SimpleMessages(),
			MaxTokens: &maxTokens,
		}

		req := convertParams(params)

		require.Equal(t, int64(100), req.MaxCompletionTokens.Value)
	})

	t.Run("converts stop sequences", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			Stop:     []string{"END", "STOP"},
		}

		req := convertParams(params)

		require.NotNil(t, req.Stop)
	})

	t.Run("converts tools", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			Tools:    []providers.Tool{testutil.WeatherTool()},
		}

		req := convertParams(params)

		require.Len(t, req.Tools, 1)
	})

	t.Run("converts tool_choice auto", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:      "gpt-4",
			Messages:   testutil.SimpleMessages(),
			Tools:      []providers.Tool{testutil.WeatherTool()},
			ToolChoice: "auto",
		}

		req := convertParams(params)

		require.NotNil(t, req.ToolChoice)
	})

	t.Run("converts tool_choice required", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:      "gpt-4",
			Messages:   testutil.SimpleMessages(),
			Tools:      []providers.Tool{testutil.WeatherTool()},
			ToolChoice: "required",
		}

		req := convertParams(params)

		require.NotNil(t, req.ToolChoice)
	})

	t.Run("converts tool_choice with specific function", func(t *testing.T) {
		t.Parallel()

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

		require.NotNil(t, req.ToolChoice)
	})

	t.Run("converts response_format json_object", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			ResponseFormat: &providers.ResponseFormat{
				Type: "json_object",
			},
		}

		req := convertParams(params)

		require.NotNil(t, req.ResponseFormat)
	})

	t.Run("converts reasoning_effort", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:           "o1-mini",
			Messages:        testutil.SimpleMessages(),
			ReasoningEffort: providers.ReasoningEffortHigh,
		}

		req := convertParams(params)

		require.NotNil(t, req.ReasoningEffort)
	})

	t.Run("converts seed", func(t *testing.T) {
		t.Parallel()

		seed := 42
		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			Seed:     &seed,
		}

		req := convertParams(params)

		require.Equal(t, int64(42), req.Seed.Value)
	})

	t.Run("converts user", func(t *testing.T) {
		t.Parallel()

		params := providers.CompletionParams{
			Model:    "gpt-4",
			Messages: testutil.SimpleMessages(),
			User:     "test-user",
		}

		req := convertParams(params)

		require.Equal(t, "test-user", req.User.Value)
	})
}

func TestConvertMessage(t *testing.T) {
	t.Parallel()

	t.Run("converts system message", func(t *testing.T) {
		t.Parallel()

		msg := providers.Message{Role: providers.RoleSystem, Content: "You are helpful"}
		result := convertMessage(msg)
		require.NotNil(t, result)
	})

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		msg := providers.Message{Role: providers.RoleUser, Content: "Hello"}
		result := convertMessage(msg)
		require.NotNil(t, result)
	})

	t.Run("converts assistant message", func(t *testing.T) {
		t.Parallel()

		msg := providers.Message{Role: providers.RoleAssistant, Content: "Hi there!"}
		result := convertMessage(msg)
		require.NotNil(t, result)
	})

	t.Run("converts assistant message with tool calls", func(t *testing.T) {
		t.Parallel()

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
		require.NotNil(t, result)
	})

	t.Run("converts tool result message", func(t *testing.T) {
		t.Parallel()

		msg := providers.Message{
			Role:       providers.RoleTool,
			Content:    "sunny, 22°C",
			ToolCallID: "call_123",
		}
		result := convertMessage(msg)
		require.NotNil(t, result)
	})

	t.Run("converts multimodal user message", func(t *testing.T) {
		t.Parallel()

		msg := providers.Message{
			Role: providers.RoleUser,
			Content: []providers.ContentPart{
				{Type: "text", Text: "What's in this image?"},
				{Type: "image_url", ImageURL: &providers.ImageURL{URL: "https://example.com/image.png"}},
			},
		}
		result := convertMessage(msg)
		require.NotNil(t, result)
	})
}

func TestConvertResponse(t *testing.T) {
	t.Parallel()

	t.Run("converts basic response", func(t *testing.T) {
		t.Parallel()
		// We can't easily test this without mocking the OpenAI SDK response.
		// This would be tested in integration tests.
	})
}

// Integration tests - only run if API key is available.

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Equal(t, "chat.completion", resp.Object)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
	require.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
	require.NotNil(t, resp.Usage)
	require.Greater(t, resp.Usage.TotalTokens, 0)
}

func TestIntegrationCompletionStream(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model should call the weather tool.
	if len(resp.Choices[0].Message.ToolCalls) > 0 {
		tc := resp.Choices[0].Message.ToolCalls[0]
		require.Equal(t, "get_weather", tc.Function.Name)
		require.Contains(t, tc.Function.Arguments, "Paris")
		require.Equal(t, providers.FinishReasonToolCalls, resp.Choices[0].FinishReason)
	}
}

func TestIntegrationCompletionConversation(t *testing.T) {
	t.Parallel()

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

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model should remember the name "Alice".
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")
	require.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestIntegrationEmbedding(t *testing.T) {
	t.Parallel()

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

	require.Equal(t, "list", resp.Object)
	require.Len(t, resp.Data, 1)
	require.Greater(t, len(resp.Data[0].Embedding), 0)
	require.NotNil(t, resp.Usage)
}

func TestIntegrationListModels(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("openai") {
		t.Skip("OPENAI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.ListModels(ctx)
	require.NoError(t, err)

	require.Equal(t, "list", resp.Object)
	require.Greater(t, len(resp.Data), 0)

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
	require.True(t, found, "Expected to find GPT models in the list")
}

func TestIntegrationAuthenticationError(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("invalid-api-key"))
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    "gpt-4o-mini",
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
			err:          newTestAPIError(t, 401, ""),
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "429 status becomes RateLimitError",
			err:          newTestAPIError(t, 429, ""),
			wantSentinel: errors.ErrRateLimit,
		},
		{
			name:         "404 status becomes ModelNotFoundError",
			err:          newTestAPIError(t, 404, ""),
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "400 with context_length_exceeded becomes ContextLengthError",
			err:          newTestAPIError(t, 400, apiCodeContextLengthExceeded),
			wantSentinel: errors.ErrContextLength,
		},
		{
			name:         "400 with content_filter becomes ContentFilterError",
			err:          newTestAPIError(t, 400, apiCodeContentFilter),
			wantSentinel: errors.ErrContentFilter,
		},
		{
			name:         "400 with content_policy_violation becomes ContentFilterError",
			err:          newTestAPIError(t, 400, apiCodeContentPolicyViolated),
			wantSentinel: errors.ErrContentFilter,
		},
		{
			name:         "400 with unknown code becomes InvalidRequestError",
			err:          newTestAPIError(t, 400, "unknown_error"),
			wantSentinel: errors.ErrInvalidRequest,
		},
		{
			name:         "model_not_found code becomes ModelNotFoundError",
			err:          newTestAPIError(t, 500, apiCodeModelNotFound),
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "invalid_api_key code becomes AuthenticationError",
			err:          newTestAPIError(t, 500, apiCodeInvalidAPIKey),
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "rate_limit_exceeded code becomes RateLimitError",
			err:          newTestAPIError(t, 500, apiCodeRateLimitExceeded),
			wantSentinel: errors.ErrRateLimit,
		},
		{
			name:         "unknown status and code becomes ProviderError",
			err:          newTestAPIError(t, 500, "unknown"),
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

// newTestAPIError creates an OpenAI API error for testing.
func newTestAPIError(t *testing.T, statusCode int, code string) *openai.Error {
	t.Helper()

	testURL, _ := url.Parse("https://api.openai.com/v1/chat/completions")
	return &openai.Error{
		StatusCode: statusCode,
		Code:       code,
		Message:    "test error message",
		Type:       "error",
		Request:    &http.Request{Method: "POST", URL: testURL},
		Response:   &http.Response{StatusCode: statusCode},
	}
}
