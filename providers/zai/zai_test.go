package zai

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"io"
	"net/http"
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
			config.WithBaseURL("https://custom.z.ai/v1"),
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
	require.False(t, caps.CompletionImage)
	require.False(t, caps.CompletionPDF)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.False(t, caps.Embedding)
	require.True(t, caps.ListModels)
}

func TestProviderName(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)
	require.Equal(t, providerName, provider.Name())
}

func TestHandleErrorResponse(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		statusCode   int
		body         string
		wantSentinel error
		wantContains string
	}{
		{
			name:         "401 becomes AuthenticationError",
			statusCode:   401,
			body:         `{"error":{"message":"invalid api key","code":1001}}`,
			wantSentinel: errors.ErrAuthentication,
			wantContains: "invalid api key",
		},
		{
			name:         "429 becomes RateLimitError",
			statusCode:   429,
			body:         `{"error":{"message":"rate limit exceeded","code":1002}}`,
			wantSentinel: errors.ErrRateLimit,
			wantContains: "rate limit exceeded",
		},
		{
			name:         "404 becomes ModelNotFoundError",
			statusCode:   404,
			body:         `{"error":{"message":"model not found","code":1211}}`,
			wantSentinel: errors.ErrModelNotFound,
			wantContains: "model not found",
		},
		{
			name:         "400 becomes InvalidRequestError",
			statusCode:   400,
			body:         `{"error":{"message":"invalid request","code":1210}}`,
			wantSentinel: errors.ErrInvalidRequest,
			wantContains: "invalid request",
		},
		{
			name:         "500 becomes ProviderError",
			statusCode:   500,
			body:         `{"error":{"message":"internal error","code":1500}}`,
			wantSentinel: errors.ErrProvider,
			wantContains: "internal error",
		},
		{
			name:         "unparseable body falls back to raw message",
			statusCode:   502,
			body:         "bad gateway",
			wantSentinel: errors.ErrProvider,
			wantContains: "bad gateway",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := &Provider{}
			resp := &http.Response{
				StatusCode: tc.statusCode,
				Body:       io.NopCloser(strings.NewReader(tc.body)),
			}

			err := p.handleErrorResponse(resp)
			require.Error(t, err)
			require.True(t, stderrors.Is(err, tc.wantSentinel),
				"expected error to match %v, got %v", tc.wantSentinel, err)
			require.Contains(t, err.Error(), tc.wantContains)
		})
	}
}

func TestCreateRequest(t *testing.T) {
	t.Parallel()

	t.Run("converts basic parameters", func(t *testing.T) {
		t.Parallel()

		p := &Provider{}
		temp := 0.7
		topP := 0.9
		maxTokens := 100

		model := testutil.TestModel(providerName)
		params := providers.CompletionParams{
			Model: model,
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Hello"},
			},
			Temperature: &temp,
			TopP:        &topP,
			MaxTokens:   &maxTokens,
			Stop:        []string{"END"},
			User:        "test-user-123",
		}

		req, err := p.createRequest(params, false)
		require.NoError(t, err)

		require.Equal(t, model, req.Model)
		require.False(t, req.Stream)
		require.Equal(t, &temp, req.Temperature)
		require.Equal(t, &topP, req.TopP)
		require.Equal(t, &maxTokens, req.MaxTokens)
		require.Equal(t, []string{"END"}, req.Stop)
		require.Equal(t, "test-user-123", req.UserID)
		require.Nil(t, req.Thinking)
		require.Len(t, req.Messages, 1)
		require.Equal(t, providers.RoleUser, req.Messages[0].Role)
	})

	t.Run("enables stream", func(t *testing.T) {
		t.Parallel()

		p := &Provider{}
		params := providers.CompletionParams{
			Model: testutil.TestModel(providerName),
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Hello"},
			},
		}

		req, err := p.createRequest(params, true)
		require.NoError(t, err)
		require.True(t, req.Stream)
	})

	t.Run("enables thinking when reasoning effort is set", func(t *testing.T) {
		t.Parallel()

		p := &Provider{}
		params := providers.CompletionParams{
			Model: testutil.ReasoningModel(providerName),
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Think about this"},
			},
			ReasoningEffort: providers.ReasoningEffortHigh,
		}

		req, err := p.createRequest(params, false)
		require.NoError(t, err)
		require.NotNil(t, req.Thinking)
		require.Equal(t, "enabled", req.Thinking.Type)
	})

	t.Run("does not enable thinking for ReasoningEffortNone", func(t *testing.T) {
		t.Parallel()

		p := &Provider{}
		params := providers.CompletionParams{
			Model: testutil.TestModel(providerName),
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Hello"},
			},
			ReasoningEffort: providers.ReasoningEffortNone,
		}

		req, err := p.createRequest(params, false)
		require.NoError(t, err)
		require.Nil(t, req.Thinking)
	})

	t.Run("preserves reasoning_content from history", func(t *testing.T) {
		t.Parallel()

		p := &Provider{}
		params := providers.CompletionParams{
			Model: testutil.ReasoningModel(providerName),
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Think about this"},
				{
					Role:    providers.RoleAssistant,
					Content: "Here is my answer",
					Reasoning: &providers.Reasoning{
						Content: "Let me think step by step...",
					},
				},
				{Role: providers.RoleUser, Content: "Follow up question"},
			},
		}

		req, err := p.createRequest(params, false)
		require.NoError(t, err)
		require.Len(t, req.Messages, 3)

		// The assistant message should carry reasoning_content.
		require.Equal(t, "Let me think step by step...", req.Messages[1].ReasoningContent)

		// User messages should not have reasoning_content.
		require.Empty(t, req.Messages[0].ReasoningContent)
		require.Empty(t, req.Messages[2].ReasoningContent)
	})

	t.Run("converts tool calls and tool results", func(t *testing.T) {
		t.Parallel()

		p := &Provider{}
		params := providers.CompletionParams{
			Model: testutil.TestModel(providerName),
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "What is the weather?"},
				{
					Role:    providers.RoleAssistant,
					Content: "",
					ToolCalls: []providers.ToolCall{
						{
							ID:   "call_abc",
							Type: "function",
							Function: providers.FunctionCall{
								Name:      "get_weather",
								Arguments: `{"location":"Paris"}`,
							},
						},
					},
				},
				{
					Role:       providers.RoleTool,
					Content:    `{"temp":22}`,
					ToolCallID: "call_abc",
				},
			},
			Tools:      []providers.Tool{testutil.WeatherTool()},
			ToolChoice: "auto",
		}

		req, err := p.createRequest(params, false)
		require.NoError(t, err)
		require.Len(t, req.Messages, 3)

		// Assistant message should have tool calls.
		require.Len(t, req.Messages[1].ToolCalls, 1)
		require.Equal(t, "get_weather", req.Messages[1].ToolCalls[0].Function.Name)

		// Tool message should have tool_call_id.
		require.Equal(t, "call_abc", req.Messages[2].ToolCallID)

		// Request-level tools and tool_choice should be set.
		require.Len(t, req.Tools, 1)
		require.Equal(t, "auto", req.ToolChoice)
	})

	t.Run("converts multimodal content parts", func(t *testing.T) {
		t.Parallel()

		p := &Provider{}
		params := providers.CompletionParams{
			Model: testutil.ProviderImageModelMap[providerName],
			Messages: []providers.Message{
				{
					Role: providers.RoleUser,
					Content: []providers.ContentPart{
						{Type: "text", Text: "What is in this image?"},
						{
							Type:     "image_url",
							ImageURL: &providers.ImageURL{URL: "https://example.com/img.jpg"},
						},
					},
				},
			},
		}

		req, err := p.createRequest(params, false)
		require.NoError(t, err)
		require.Len(t, req.Messages, 1)

		// Content should be an array of content parts.
		parts, ok := req.Messages[0].Content.([]contentPart)
		require.True(t, ok, "expected content to be []contentPart")
		require.Len(t, parts, 2)

		require.Equal(t, "text", parts[0].Type)
		require.Equal(t, "What is in this image?", parts[0].Text)

		require.Equal(t, "image_url", parts[1].Type)
		imgURL, ok := parts[1].ImageURL.(map[string]string)
		require.True(t, ok)
		require.Equal(t, "https://example.com/img.jpg", imgURL["url"])
	})

	t.Run("strips data URI prefix from base64 images", func(t *testing.T) {
		t.Parallel()

		p := &Provider{}
		params := providers.CompletionParams{
			Model: testutil.ProviderImageModelMap[providerName],
			Messages: []providers.Message{
				{
					Role: providers.RoleUser,
					Content: []providers.ContentPart{
						{
							Type:     "image_url",
							ImageURL: &providers.ImageURL{URL: "data:image/png;base64,iVBOR..."},
						},
					},
				},
			},
		}

		req, err := p.createRequest(params, false)
		require.NoError(t, err)

		parts, ok := req.Messages[0].Content.([]contentPart)
		require.True(t, ok)
		require.Len(t, parts, 1)

		imgURL, ok := parts[0].ImageURL.(map[string]string)
		require.True(t, ok)
		require.Equal(t, "iVBOR...", imgURL["url"])
	})

	t.Run("serializes user_id correctly in JSON", func(t *testing.T) {
		t.Parallel()

		p := &Provider{}
		params := providers.CompletionParams{
			Model: testutil.TestModel(providerName),
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: "Hello"},
			},
			User: "usr-12345678",
		}

		req, err := p.createRequest(params, false)
		require.NoError(t, err)

		data, err := json.Marshal(req)
		require.NoError(t, err)

		// Verify it uses "user_id" not "user" as the JSON key.
		require.Contains(t, string(data), `"user_id"`)
		require.NotContains(t, string(data), `"user":`)
	})
}

func TestToProviderCompletion(t *testing.T) {
	t.Parallel()

	t.Run("converts basic response", func(t *testing.T) {
		t.Parallel()

		model := testutil.TestModel(providerName)
		zaiResp := &zaiChatCompletion{
			ID:      "chatcmpl-123",
			Object:  objectChatCompletion,
			Created: 1234567890,
			Model:   model,
			Choices: []zaiChoice{
				{
					Index: 0,
					Message: zaiMessage{
						Message: providers.Message{
							Role:    providers.RoleAssistant,
							Content: "Hello World",
						},
					},
					FinishReason: "stop",
				},
			},
			Usage: &providers.Usage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		}

		result := zaiResp.toProviderCompletion()

		require.Equal(t, "chatcmpl-123", result.ID)
		require.Equal(t, objectChatCompletion, result.Object)
		require.Equal(t, int64(1234567890), result.Created)
		require.Equal(t, model, result.Model)
		require.Len(t, result.Choices, 1)
		require.Equal(t, providers.RoleAssistant, result.Choices[0].Message.Role)
		require.Equal(t, "Hello World", result.Choices[0].Message.Content)
		require.Equal(t, "stop", result.Choices[0].FinishReason)
		require.NotNil(t, result.Usage)
		require.Equal(t, 15, result.Usage.TotalTokens)
	})

	t.Run("converts response with reasoning_content", func(t *testing.T) {
		t.Parallel()

		zaiResp := &zaiChatCompletion{
			ID:      "chatcmpl-456",
			Object:  objectChatCompletion,
			Created: 1234567890,
			Model:   testutil.ReasoningModel(providerName),
			Choices: []zaiChoice{
				{
					Index: 0,
					Message: zaiMessage{
						Message: providers.Message{
							Role:    providers.RoleAssistant,
							Content: "The answer is 42.",
						},
						ReasoningContent: "Let me think step by step...",
					},
					FinishReason: "stop",
				},
			},
		}

		result := zaiResp.toProviderCompletion()

		require.Len(t, result.Choices, 1)
		require.Equal(t, "The answer is 42.", result.Choices[0].Message.Content)
		require.NotNil(t, result.Choices[0].Message.Reasoning)
		require.Equal(t, "Let me think step by step...", result.Choices[0].Message.Reasoning.Content)
	})

	t.Run("converts response without reasoning_content", func(t *testing.T) {
		t.Parallel()

		zaiResp := &zaiChatCompletion{
			ID:     "chatcmpl-789",
			Object: objectChatCompletion,
			Model:  testutil.TestModel(providerName),
			Choices: []zaiChoice{
				{
					Index: 0,
					Message: zaiMessage{
						Message: providers.Message{
							Role:    providers.RoleAssistant,
							Content: "Hello",
						},
					},
					FinishReason: "stop",
				},
			},
		}

		result := zaiResp.toProviderCompletion()

		require.Nil(t, result.Choices[0].Message.Reasoning)
	})

	t.Run("converts response with tool_calls", func(t *testing.T) {
		t.Parallel()

		zaiResp := &zaiChatCompletion{
			ID:     "chatcmpl-tool",
			Object: objectChatCompletion,
			Model:  testutil.TestModel(providerName),
			Choices: []zaiChoice{
				{
					Index: 0,
					Message: zaiMessage{
						Message: providers.Message{
							Role: providers.RoleAssistant,
							ToolCalls: []providers.ToolCall{
								{
									ID:   "call_001",
									Type: "function",
									Function: providers.FunctionCall{
										Name:      "get_weather",
										Arguments: `{"location":"Paris"}`,
									},
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
		}

		result := zaiResp.toProviderCompletion()

		require.Len(t, result.Choices, 1)
		require.Len(t, result.Choices[0].Message.ToolCalls, 1)
		require.Equal(t, "get_weather", result.Choices[0].Message.ToolCalls[0].Function.Name)
		require.Equal(t, "tool_calls", result.Choices[0].FinishReason)
	})
}

func TestToProviderChunk(t *testing.T) {
	t.Parallel()

	t.Run("converts basic streaming chunk", func(t *testing.T) {
		t.Parallel()

		model := testutil.TestModel(providerName)
		zaiChunk := &zaiChatCompletionChunk{
			ID:      "chatcmpl-stream-1",
			Object:  objectChatCompletionChunk,
			Created: 1234567890,
			Model:   model,
			Choices: []zaiChunkChoice{
				{
					Index: 0,
					Delta: zaiChunkDelta{
						ChunkDelta: providers.ChunkDelta{
							Role:    providers.RoleAssistant,
							Content: "Hello",
						},
					},
				},
			},
		}

		result := zaiChunk.toProviderChunk()

		require.Equal(t, "chatcmpl-stream-1", result.ID)
		require.Equal(t, objectChatCompletionChunk, result.Object)
		require.Equal(t, model, result.Model)
		require.Len(t, result.Choices, 1)
		require.Equal(t, providers.RoleAssistant, result.Choices[0].Delta.Role)
		require.Equal(t, "Hello", result.Choices[0].Delta.Content)
		require.Nil(t, result.Choices[0].Delta.Reasoning)
	})

	t.Run("converts chunk with reasoning_content", func(t *testing.T) {
		t.Parallel()

		zaiChunk := &zaiChatCompletionChunk{
			ID:     "chatcmpl-stream-2",
			Object: objectChatCompletionChunk,
			Model:  testutil.ReasoningModel(providerName),
			Choices: []zaiChunkChoice{
				{
					Index: 0,
					Delta: zaiChunkDelta{
						ReasoningContent: "Thinking...",
					},
				},
			},
		}

		result := zaiChunk.toProviderChunk()

		require.Len(t, result.Choices, 1)
		require.NotNil(t, result.Choices[0].Delta.Reasoning)
		require.Equal(t, "Thinking...", result.Choices[0].Delta.Reasoning.Content)
		require.Empty(t, result.Choices[0].Delta.Content)
	})

	t.Run("converts final chunk with finish reason and usage", func(t *testing.T) {
		t.Parallel()

		zaiChunk := &zaiChatCompletionChunk{
			ID:     "chatcmpl-stream-3",
			Object: objectChatCompletionChunk,
			Model:  testutil.TestModel(providerName),
			Choices: []zaiChunkChoice{
				{
					Index:        0,
					Delta:        zaiChunkDelta{},
					FinishReason: "stop",
				},
			},
			Usage: &providers.Usage{
				PromptTokens:     10,
				CompletionTokens: 20,
				TotalTokens:      30,
			},
		}

		result := zaiChunk.toProviderChunk()

		require.Equal(t, "stop", result.Choices[0].FinishReason)
		require.NotNil(t, result.Usage)
		require.Equal(t, 30, result.Usage.TotalTokens)
	})

	t.Run("converts chunk with tool calls", func(t *testing.T) {
		t.Parallel()

		zaiChunk := &zaiChatCompletionChunk{
			ID:     "chatcmpl-stream-4",
			Object: objectChatCompletionChunk,
			Model:  testutil.TestModel(providerName),
			Choices: []zaiChunkChoice{
				{
					Index: 0,
					Delta: zaiChunkDelta{
						ChunkDelta: providers.ChunkDelta{
							ToolCalls: []providers.ToolCall{
								{
									ID:   "call_001",
									Type: "function",
									Function: providers.FunctionCall{
										Name:      "get_weather",
										Arguments: `{"loc`,
									},
								},
							},
						},
					},
				},
			},
		}

		result := zaiChunk.toProviderChunk()

		require.Len(t, result.Choices[0].Delta.ToolCalls, 1)
		require.Equal(t, "get_weather", result.Choices[0].Delta.ToolCalls[0].Function.Name)
	})
}

func TestConvertError(t *testing.T) {
	t.Parallel()

	p := &Provider{}

	t.Run("wraps error as ProviderError", func(t *testing.T) {
		t.Parallel()

		err := p.ConvertError(stderrors.New("network timeout"))
		require.Error(t, err)
		require.True(t, stderrors.Is(err, errors.ErrProvider))
		require.Contains(t, err.Error(), "["+providerName+"]")
	})
}

// Integration tests - only run if z.ai API key is available.

func TestIntegrationCompletion(t *testing.T) {
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
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
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
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
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
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
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
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
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
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
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
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
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
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

	// Step 3: Parse the arguments.
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
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
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

	// Verify the parameters.
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

func TestIntegrationCompletionReasoning(t *testing.T) {
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
	}

	model := testutil.ReasoningModel(providerName)
	if model == "" {
		t.Skip("No reasoning model configured for zai")
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

func TestIntegrationAuthenticationError(t *testing.T) {
	provider, err := New(config.WithAPIKey("invalid-api-key"))
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.SimpleMessages(),
	}

	_, err = provider.Completion(ctx, params)
	require.Error(t, err)

	// Check that it's converted to an authentication error.
	var authErr *errors.AuthenticationError
	require.ErrorAs(t, err, &authErr)
}

func TestIntegrationAgentLoopContinuation(t *testing.T) {
	if testutil.SkipIfNoAPIKey(providerName) {
		t.Skip("ZAI_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	// Start with the agent loop messages (user asks, assistant calls tool, tool returns).
	messages := testutil.AgentLoopMessages()

	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: messages,
		Tools:    []providers.Tool{testutil.WeatherTool()},
	}

	// The model should respond with the weather information.
	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// Should have a content response (not another tool call).
	if contentStr := resp.Choices[0].Message.ContentString(); contentStr != "" {
		content := strings.ToLower(contentStr)
		// Should mention the weather or sunny.
		require.True(
			t,
			strings.Contains(content, "sunny") || strings.Contains(content, "weather") ||
				strings.Contains(content, "salvaterra"),
		)
	}
}
