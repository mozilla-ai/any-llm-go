package gemini

import (
	"context"
	stderrors "errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

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
		require.Equal(t, providerName, provider.Name())
	})

	t.Run("creates provider from GEMINI_API_KEY", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "env-api-key")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("creates provider from GOOGLE_API_KEY fallback", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "")
		t.Setenv("GOOGLE_API_KEY", "google-api-key")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("GEMINI_API_KEY", "")
		t.Setenv("GOOGLE_API_KEY", "")

		provider, err := New()
		require.Nil(t, provider)
		require.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
		require.Equal(t, providerName, missingKeyErr.Provider)
		require.Equal(t, envAPIKey, missingKeyErr.EnvVar)
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
	require.False(t, caps.CompletionPDF)
	require.True(t, caps.Embedding)
	require.True(t, caps.ListModels)
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

		require.NotNil(t, system)
		require.Len(t, system.Parts, 1)
		require.Equal(t, "You are a helpful assistant.", system.Parts[0].Text)
		require.Len(t, result, 1)
	})

	t.Run("concatenates multiple system messages", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "First part."},
			{Role: providers.RoleSystem, Content: "Second part."},
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system := convertMessages(messages)

		require.NotNil(t, system)
		require.Contains(t, system.Parts[0].Text, "First part.")
		require.Contains(t, system.Parts[0].Text, "Second part.")
		require.Len(t, result, 1)
	})

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system := convertMessages(messages)

		require.Nil(t, system)
		require.Len(t, result, 1)
		require.Equal(t, "user", result[0].Role)
	})

	t.Run("converts assistant message to model role", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleAssistant, Content: "Hi there!"},
		}

		result, system := convertMessages(messages)

		require.Nil(t, system)
		require.Len(t, result, 2)
		require.Equal(t, roleModel, result[1].Role)
		require.Equal(t, "Hi there!", result[1].Parts[0].Text)
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
		require.Equal(t, roleModel, result[1].Role)
		require.NotNil(t, result[1].Parts[0].FunctionCall)
		require.Equal(t, "get_weather", result[1].Parts[0].FunctionCall.Name)
	})

	t.Run("converts tool result message with plain text", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleTool, Content: "sunny, 22°C", Name: "get_weather"},
		}

		result, _ := convertMessages(messages)

		require.Len(t, result, 2)
		require.Equal(t, "user", result[1].Role)
		require.NotNil(t, result[1].Parts[0].FunctionResponse)
		require.Equal(t, "get_weather", result[1].Parts[0].FunctionResponse.Name)
		// Plain text is wrapped as {"result": "sunny, 22°C"}.
		require.Equal(t, "sunny, 22°C", result[1].Parts[0].FunctionResponse.Response["result"])
	})

	t.Run("converts tool result message with JSON content", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleTool, Content: `{"temperature": 22, "condition": "sunny"}`, Name: "get_weather"},
		}

		result, _ := convertMessages(messages)

		require.Len(t, result, 2)
		require.NotNil(t, result[1].Parts[0].FunctionResponse)
		// JSON content is parsed directly.
		require.Equal(t, "sunny", result[1].Parts[0].FunctionResponse.Response["condition"])
	})

	t.Run("converts tool result message with fallback name", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleTool, Content: "result data"},
		}

		result, _ := convertMessages(messages)

		require.Len(t, result, 1)
		require.Equal(t, "function", result[0].Parts[0].FunctionResponse.Name)
	})

	t.Run("no system returns nil instruction", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		_, system := convertMessages(messages)
		require.Nil(t, system)
	})

	t.Run("unknown role returns nil", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: "unknown", Content: "Hello"},
		}

		result, _ := convertMessages(messages)
		require.Empty(t, result)
	})
}

func TestConvertFinishReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    genai.FinishReason
		expected string
	}{
		{
			name:     "STOP",
			input:    genai.FinishReasonStop,
			expected: providers.FinishReasonStop,
		},
		{
			name:     "MAX_TOKENS",
			input:    genai.FinishReasonMaxTokens,
			expected: providers.FinishReasonLength,
		},
		{
			name:     "SAFETY",
			input:    genai.FinishReasonSafety,
			expected: providers.FinishReasonContentFilter,
		},
		{
			name:     "RECITATION",
			input:    genai.FinishReasonRecitation,
			expected: providers.FinishReasonStop,
		},
		{
			name:     "BLOCKLIST",
			input:    genai.FinishReasonBlocklist,
			expected: providers.FinishReasonContentFilter,
		},
		{
			name:     "PROHIBITED_CONTENT",
			input:    genai.FinishReasonProhibitedContent,
			expected: providers.FinishReasonContentFilter,
		},
		{
			name:     "unknown",
			input:    "UNKNOWN",
			expected: providers.FinishReasonStop,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result := convertFinishReason(tc.input)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestConvertTools(t *testing.T) {
	t.Parallel()

	tools := []providers.Tool{
		{
			Type: "function",
			Function: providers.Function{
				Name:        "get_weather",
				Description: "Get the weather for a location.",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city name",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	result := convertTools(tools)

	require.Len(t, result, 1)
	require.Len(t, result[0].FunctionDeclarations, 1)
	require.Equal(t, "get_weather", result[0].FunctionDeclarations[0].Name)
	require.Equal(t, "Get the weather for a location.", result[0].FunctionDeclarations[0].Description)
	require.NotNil(t, result[0].FunctionDeclarations[0].ParametersJsonSchema)
}

func TestConvertToolChoice(t *testing.T) {
	t.Parallel()

	t.Run("auto string", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice("auto")
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeAuto, result.FunctionCallingConfig.Mode)
	})

	t.Run("none string", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice("none")
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeNone, result.FunctionCallingConfig.Mode)
	})

	t.Run("required string", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice("required")
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeAny, result.FunctionCallingConfig.Mode)
	})

	t.Run("specific function", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice(providers.ToolChoice{
			Type:     "function",
			Function: &providers.ToolChoiceFunction{Name: "get_weather"},
		})
		require.NotNil(t, result)
		require.Equal(t, genai.FunctionCallingConfigModeAny, result.FunctionCallingConfig.Mode)
		require.Contains(t, result.FunctionCallingConfig.AllowedFunctionNames, "get_weather")
	})

	t.Run("unknown returns nil", func(t *testing.T) {
		t.Parallel()

		result := convertToolChoice("unknown_value")
		require.Nil(t, result)
	})
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
			err:          &genai.APIError{Code: 401, Message: "unauthorized"},
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "403 status becomes AuthenticationError",
			err:          &genai.APIError{Code: 403, Message: "forbidden"},
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "404 status becomes ModelNotFoundError",
			err:          &genai.APIError{Code: 404, Message: "not found"},
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "429 status becomes RateLimitError",
			err:          &genai.APIError{Code: 429, Message: "rate limited"},
			wantSentinel: errors.ErrRateLimit,
		},
		{
			name:         "400 status becomes InvalidRequestError",
			err:          &genai.APIError{Code: 400, Message: "bad request"},
			wantSentinel: errors.ErrInvalidRequest,
		},
		{
			name:         "400 with context message becomes ContextLengthError",
			err:          &genai.APIError{Code: 400, Message: "context length exceeded"},
			wantSentinel: errors.ErrContextLength,
		},
		{
			name:         "400 with token message becomes ContextLengthError",
			err:          &genai.APIError{Code: 400, Message: "too many tokens in request"},
			wantSentinel: errors.ErrContextLength,
		},
		{
			name:         "400 with safety message becomes ContentFilterError",
			err:          &genai.APIError{Code: 400, Message: "blocked by safety filters"},
			wantSentinel: errors.ErrContentFilter,
		},
		{
			name:         "500 status becomes ProviderError",
			err:          &genai.APIError{Code: 500, Message: "internal error"},
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
			require.True(
				t,
				stderrors.Is(result, tc.wantSentinel),
				"expected error to match %v, got %v",
				tc.wantSentinel,
				result,
			)
			require.Contains(t, result.Error(), "["+providerName+"]")
		})
	}
}

func TestThinkingBudget(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		effort   providers.ReasoningEffort
		expected int32
		ok       bool
	}{
		{
			name:     "low effort",
			effort:   providers.ReasoningEffortLow,
			expected: thinkingBudgetLow,
			ok:       true,
		},
		{
			name:     "medium effort",
			effort:   providers.ReasoningEffortMedium,
			expected: thinkingBudgetMedium,
			ok:       true,
		},
		{
			name:     "high effort",
			effort:   providers.ReasoningEffortHigh,
			expected: thinkingBudgetHigh,
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

func TestApplyThinking(t *testing.T) {
	t.Parallel()

	t.Run("empty effort does nothing", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyThinking(cfg, "")
		require.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("none effort does nothing", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyThinking(cfg, providers.ReasoningEffortNone)
		require.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("low effort sets thinking config", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyThinking(cfg, providers.ReasoningEffortLow)
		require.NotNil(t, cfg.ThinkingConfig)
		require.True(t, cfg.ThinkingConfig.IncludeThoughts)
		require.Equal(t, thinkingBudgetLow, *cfg.ThinkingConfig.ThinkingBudget)
	})

	t.Run("high effort sets thinking config", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyThinking(cfg, providers.ReasoningEffortHigh)
		require.NotNil(t, cfg.ThinkingConfig)
		require.True(t, cfg.ThinkingConfig.IncludeThoughts)
		require.Equal(t, thinkingBudgetHigh, *cfg.ThinkingConfig.ThinkingBudget)
	})
}

func TestConvertImagePart(t *testing.T) {
	t.Parallel()

	t.Run("converts base64 image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
		result := convertImagePart(img)
		require.NotNil(t, result)
		require.NotNil(t, result.InlineData)
		require.Equal(t, "image/jpeg", result.InlineData.MIMEType)
	})

	t.Run("converts URL image", func(t *testing.T) {
		t.Parallel()

		img := &providers.ImageURL{URL: "https://example.com/image.png"}
		result := convertImagePart(img)
		require.NotNil(t, result)
		require.NotNil(t, result.FileData)
		require.Equal(t, "https://example.com/image.png", result.FileData.FileURI)
	})
}

func TestConvertEmbeddingInput(t *testing.T) {
	t.Parallel()

	t.Run("string input", func(t *testing.T) {
		t.Parallel()

		result := convertEmbeddingInput("hello world")
		require.NotNil(t, result)
		require.Len(t, result.Parts, 1)
		require.Equal(t, "hello world", result.Parts[0].Text)
	})

	t.Run("string slice input", func(t *testing.T) {
		t.Parallel()

		result := convertEmbeddingInput([]string{"hello", "world"})
		require.NotNil(t, result)
		require.Len(t, result.Parts, 2)
		require.Equal(t, "hello", result.Parts[0].Text)
		require.Equal(t, "world", result.Parts[1].Text)
	})
}

func TestGenerateID(t *testing.T) {
	t.Parallel()

	t.Run("has correct prefix", func(t *testing.T) {
		t.Parallel()

		id, err := generateID(idPrefixCompletion)
		require.NoError(t, err)
		require.True(t, strings.HasPrefix(id, idPrefixCompletion))
	})

	t.Run("generates unique IDs", func(t *testing.T) {
		t.Parallel()

		id1, err := generateID(idPrefixToolCall)
		require.NoError(t, err)
		id2, err := generateID(idPrefixToolCall)
		require.NoError(t, err)
		require.NotEqual(t, id1, id2)
	})

	t.Run("has expected length", func(t *testing.T) {
		t.Parallel()

		id, err := generateID("test-")
		require.NoError(t, err)
		// prefix (5) + 24 hex chars (12 bytes) = 29.
		require.Len(t, id, 29)
	})
}

func TestNewStreamState(t *testing.T) {
	t.Parallel()

	state, err := newStreamState("gemini-1.5-flash")
	require.NoError(t, err)
	require.NotNil(t, state)
	require.Equal(t, "gemini-1.5-flash", state.model)
	require.True(t, strings.HasPrefix(state.messageID, idPrefixCompletion))
	require.Nil(t, state.toolCalls)
	require.Nil(t, state.usage)
}

func TestStreamStateProcessResponse(t *testing.T) {
	t.Parallel()

	t.Run("processes text content", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hello "}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)
		require.Equal(t, "Hello ", chunks[0].Choices[0].Delta.Content)
		require.Equal(t, "Hello ", state.content.String())
	})

	t.Run("processes thinking content", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Let me think...", Thought: true}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)
		require.NotNil(t, chunks[0].Choices[0].Delta.Reasoning)
		require.Equal(t, "Let me think...", chunks[0].Choices[0].Delta.Reasoning.Content)
	})

	t.Run("processes function call", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
					}},
				},
			}},
		}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Len(t, chunks, 1)
		require.Len(t, chunks[0].Choices[0].Delta.ToolCalls, 1)
		require.Equal(t, "get_weather", chunks[0].Choices[0].Delta.ToolCalls[0].Function.Name)
		require.Contains(t, chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments, "Paris")
	})

	t.Run("tracks usage metadata", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     10,
				CandidatesTokenCount: 5,
			},
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hi"}},
				},
			}},
		}

		_, err = state.processResponse(resp)
		require.NoError(t, err)
		require.NotNil(t, state.usage)
		require.Equal(t, 10, state.usage.PromptTokens)
		require.Equal(t, 5, state.usage.CompletionTokens)
		require.Equal(t, 15, state.usage.TotalTokens)
	})

	t.Run("returns empty slice for empty candidates", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		resp := &genai.GenerateContentResponse{}

		chunks, err := state.processResponse(resp)
		require.NoError(t, err)
		require.Empty(t, chunks)
	})
}

func TestStreamStateFinalChunk(t *testing.T) {
	t.Parallel()

	t.Run("defaults to stop finish reason", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.finishReason = genai.FinishReasonStop

		chunk := state.finalChunk()
		require.Equal(t, providers.FinishReasonStop, chunk.Choices[0].FinishReason)
	})

	t.Run("uses tool_calls when tool calls present", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.finishReason = genai.FinishReasonStop
		state.toolCalls = []providers.ToolCall{
			{ID: "call_1", Type: "function", Function: providers.FunctionCall{Name: "get_weather"}},
		}

		chunk := state.finalChunk()
		require.Equal(t, providers.FinishReasonToolCalls, chunk.Choices[0].FinishReason)
	})

	t.Run("uses max_tokens finish reason", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.finishReason = genai.FinishReasonMaxTokens

		chunk := state.finalChunk()
		require.Equal(t, providers.FinishReasonLength, chunk.Choices[0].FinishReason)
	})

	t.Run("includes usage", func(t *testing.T) {
		t.Parallel()

		state, err := newStreamState("test-model")
		require.NoError(t, err)
		state.usage = &providers.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}

		chunk := state.finalChunk()
		require.NotNil(t, chunk.Usage)
		require.Equal(t, 15, chunk.Usage.TotalTokens)
	})
}

func TestConvertResponse(t *testing.T) {
	t.Parallel()

	t.Run("converts text response", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{Text: "Hello World"}},
				},
				FinishReason: genai.FinishReasonStop,
			}},
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     10,
				CandidatesTokenCount: 5,
			},
		}

		result, err := convertResponse(resp, "gemini-1.5-flash")
		require.NoError(t, err)
		require.Equal(t, objectChatCompletion, result.Object)
		require.Equal(t, "gemini-1.5-flash", result.Model)
		require.Len(t, result.Choices, 1)
		require.Equal(t, "Hello World", result.Choices[0].Message.ContentString())
		require.Equal(t, providers.RoleAssistant, result.Choices[0].Message.Role)
		require.Equal(t, providers.FinishReasonStop, result.Choices[0].FinishReason)
		require.NotNil(t, result.Usage)
		require.Equal(t, 10, result.Usage.PromptTokens)
		require.Equal(t, 5, result.Usage.CompletionTokens)
		require.Equal(t, 15, result.Usage.TotalTokens)
	})

	t.Run("converts function call response", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"location": "Paris"},
						},
					}},
				},
				FinishReason: genai.FinishReasonStop,
			}},
		}

		result, err := convertResponse(resp, "gemini-1.5-flash")
		require.NoError(t, err)
		require.Len(t, result.Choices[0].Message.ToolCalls, 1)
		require.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
		require.Equal(t, providers.FinishReasonToolCalls, result.Choices[0].FinishReason)
	})

	t.Run("converts thinking response", func(t *testing.T) {
		t.Parallel()

		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{Text: "Let me think...", Thought: true},
						{Text: "Hello!"},
					},
				},
				FinishReason: genai.FinishReasonStop,
			}},
		}

		result, err := convertResponse(resp, "gemini-2.0-flash")
		require.NoError(t, err)
		require.Equal(t, "Hello!", result.Choices[0].Message.ContentString())
		require.NotNil(t, result.Choices[0].Message.Reasoning)
		require.Equal(t, "Let me think...", result.Choices[0].Message.Reasoning.Content)
	})
}

func TestApplyResponseFormat(t *testing.T) {
	t.Parallel()

	t.Run("json_object sets mime type", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyResponseFormat(cfg, &providers.ResponseFormat{Type: "json_object"})
		require.Equal(t, "application/json", cfg.ResponseMIMEType)
	})

	t.Run("text does not set mime type", func(t *testing.T) {
		t.Parallel()

		cfg := &genai.GenerateContentConfig{}
		applyResponseFormat(cfg, &providers.ResponseFormat{Type: "text"})
		require.Empty(t, cfg.ResponseMIMEType)
	})
}

// Integration tests - only run if API key is available.

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
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
	require.NotNil(t, resp.Usage)
	require.Greater(t, resp.Usage.TotalTokens, 0)
}

func TestIntegrationCompletionWithSystemMessage(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
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
		t.Skip("GEMINI_API_KEY not set")
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

func TestIntegrationCompletionConversation(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
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

	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")
	require.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestIntegrationEmbedding(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.EmbeddingParams{
		Model: testutil.EmbeddingModel(providerName),
		Input: "Hello world",
	}

	resp, err := provider.Embedding(ctx, params)
	require.NoError(t, err)

	require.Equal(t, objectList, resp.Object)
	require.NotEmpty(t, resp.Data)
	require.NotEmpty(t, resp.Data[0].Embedding)
	require.Equal(t, objectEmbedding, resp.Data[0].Object)
}

func TestIntegrationListModels(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("GEMINI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.ListModels(ctx)
	require.NoError(t, err)

	require.Equal(t, objectList, resp.Object)
	require.NotEmpty(t, resp.Data)

	// Verify model structure.
	for _, model := range resp.Data {
		require.NotEmpty(t, model.ID)
		require.Equal(t, objectModel, model.Object)
		require.Equal(t, "google", model.OwnedBy)
	}
}
