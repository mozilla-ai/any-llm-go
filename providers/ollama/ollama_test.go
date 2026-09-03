package ollama

import (
	"context"
	"encoding/json"
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

type invalidMessageCase struct {
	name    string
	msg     providers.Message
	wantErr error
}

func runInvalidMessageCases(t *testing.T, tests []invalidMessageCase) {
	t.Helper()

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			converted, err := convertMessage(testCase.msg)

			require.Nil(t, converted)
			require.ErrorIs(t, err, testCase.wantErr)
		})
	}
}

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
	require.True(t, caps.CompletionImage)
	require.False(t, caps.CompletionPDF)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.True(t, caps.Embedding)
	require.True(t, caps.ListModels)
}

func TestConvertParamsPreservesReasoningAndModelDefaults(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		effort   providers.ReasoningEffort
		wantJSON string
		wantErr  error
	}{
		{name: "unset", wantJSON: "null"},
		{name: "auto", effort: providers.ReasoningEffortAuto, wantJSON: "null"},
		{name: "none", effort: providers.ReasoningEffortNone, wantJSON: "false"},
		{name: "low", effort: providers.ReasoningEffortLow, wantJSON: `"low"`},
		{name: "medium", effort: providers.ReasoningEffortMedium, wantJSON: `"medium"`},
		{name: "high", effort: providers.ReasoningEffortHigh, wantJSON: `"high"`},
		{name: "max", effort: providers.ReasoningEffort("max"), wantJSON: `"max"`},
		{name: "unsupported", effort: providers.ReasoningEffort("xhigh"), wantErr: errors.ErrUnsupportedParam},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			request, err := (&Provider{}).convertParams(providers.CompletionParams{
				ReasoningEffort: testCase.effort,
			})
			if testCase.wantErr != nil {
				require.Nil(t, request)
				require.ErrorIs(t, err, testCase.wantErr)
				return
			}

			require.NoError(t, err)
			require.Empty(t, request.Options)
			encoded, err := json.Marshal(request.Think)
			require.NoError(t, err)
			require.Equal(t, testCase.wantJSON, string(encoded))
		})
	}
}

func TestConvertMessagesPreservesToolResultNameAndReasoning(t *testing.T) {
	t.Parallel()

	messages := []providers.Message{
		{Role: providers.RoleSystem, Content: "Use tools."},
		{Role: providers.RoleUser, Content: "Weather?"},
		{
			Role:      providers.RoleAssistant,
			Content:   "",
			Reasoning: &providers.Reasoning{Content: "checking"},
			ToolCalls: []providers.ToolCall{{
				ID:   "call_weather",
				Type: toolTypeFunction,
				Function: providers.FunctionCall{
					Name:      "get_weather",
					Arguments: `{"city":"Paris"}`,
				},
			}},
		},
		{Role: providers.RoleTool, Content: "sunny", ToolCallID: "call_weather"},
	}

	converted, err := convertMessages(messages)
	require.NoError(t, err)

	wire, err := json.Marshal(converted)
	require.NoError(t, err)
	require.JSONEq(t, `[
		{"role":"system","content":"Use tools."},
		{"role":"user","content":"Weather?"},
		{"role":"assistant","content":"","thinking":"checking","tool_calls":[
			{"function":{"index":0,"name":"get_weather","arguments":{"city":"Paris"}}}
		]},
		{"role":"tool","content":"sunny","tool_name":"get_weather"}
	]`, string(wire))
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

func TestConvertMessageImages(t *testing.T) {
	t.Parallel()

	t.Run("decodes a base64 data URL once", func(t *testing.T) {
		t.Parallel()

		msg := providers.Message{
			Role: providers.RoleUser,
			Content: []any{
				map[string]any{"type": "text", "text": "What's in this image?"},
				map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": "data:image/jpeg;base64,aGVsbG8=",
					},
				},
			},
		}

		converted, err := convertMessage(msg)

		require.NoError(t, err)
		require.Equal(t, "What's in this image?", converted.Content)
		require.Len(t, converted.Images, 1)
		require.Equal(t, "hello", string(converted.Images[0]))

		wire, err := json.Marshal(converted)
		require.NoError(t, err)
		require.JSONEq(t, `{
			"role":"user",
			"content":"What's in this image?",
			"images":["aGVsbG8="]
		}`, string(wire))
	})

	t.Run("rejects image URLs the SDK cannot encode", func(t *testing.T) {
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

		converted, err := convertMessage(msg)

		require.Nil(t, converted)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	})
}

func TestConvertTools(t *testing.T) {
	t.Parallel()

	empty, err := convertTools(nil)
	require.NoError(t, err)
	require.Nil(t, empty)

	tools := []providers.Tool{
		{
			Type: toolTypeFunction,
			Function: providers.Function{
				Name:        "get_weather",
				Description: "Get the current weather",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The city name",
						},
					},
					"required": []any{"location"},
				},
			},
		},
	}

	result, err := convertTools(tools)
	require.NoError(t, err)

	wire, err := json.Marshal(result)
	require.NoError(t, err)
	require.JSONEq(t, `[{"type":"function","function":{
		"name":"get_weather",
		"description":"Get the current weather",
		"parameters":{
			"type":"object",
			"required":["location"],
			"properties":{"location":{"type":"string","description":"The city name"}}
		}
	}}]`, string(wire))
}

func TestConvertToolsRejectsInvalidDefinitions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		tool    providers.Tool
		wantErr error
	}{
		{
			name:    "unknown tool type",
			tool:    providers.Tool{Type: "custom"},
			wantErr: errors.ErrUnsupportedParam,
		},
		{
			name: "missing function name",
			tool: providers.Tool{
				Type: toolTypeFunction,
				Function: providers.Function{
					Parameters: map[string]any{"type": "object", "properties": map[string]any{}},
				},
			},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name: "missing parameters schema",
			tool: providers.Tool{
				Type: toolTypeFunction, Function: providers.Function{Name: "get_weather"},
			},
			wantErr: errors.ErrInvalidRequest,
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			converted, err := convertTools([]providers.Tool{testCase.tool})

			require.Nil(t, converted)
			require.ErrorIs(t, err, testCase.wantErr)
		})
	}
}

func TestConvertToolsRejectsInvalidOrLossySchemas(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		params  map[string]any
		wantErr error
	}{
		{
			name:    "unencodable schema",
			params:  map[string]any{"type": make(chan int)},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name:    "invalid schema shape",
			params:  map[string]any{"type": "object", "properties": []string{"not", "a", "mapping"}},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name:    "schema outside SDK subset",
			params:  map[string]any{"type": "object", "properties": map[string]any{}, "additionalProperties": false},
			wantErr: errors.ErrUnsupportedParam,
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			converted, err := convertTools([]providers.Tool{{
				Type: toolTypeFunction,
				Function: providers.Function{
					Name: "get_weather", Parameters: testCase.params,
				},
			}})

			require.Nil(t, converted)
			require.ErrorIs(t, err, testCase.wantErr)
		})
	}
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

		result, err := convertResponseFormat(nil)

		require.NoError(t, err)
		require.Nil(t, result)
	})

	t.Run("json_object format", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{Type: responseFormatJSON}
		result, err := convertResponseFormat(format)

		require.NoError(t, err)
		require.NotNil(t, result)
		require.Equal(t, `"json"`, string(result))
	})

	t.Run("text format uses the model default", func(t *testing.T) {
		t.Parallel()

		result, err := convertResponseFormat(&providers.ResponseFormat{Type: responseFormatText})

		require.NoError(t, err)
		require.Nil(t, result)
	})

	t.Run("json_schema format", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{
			Type: responseFormatSchema,
			JSONSchema: &providers.JSONSchema{
				Name: "test",
				Schema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
					},
				},
			},
		}
		result, err := convertResponseFormat(format)

		require.NoError(t, err)
		require.JSONEq(t, `{"type":"object","properties":{"name":{"type":"string"}}}`, string(result))
	})

	t.Run("rejects missing json schema", func(t *testing.T) {
		t.Parallel()

		result, err := convertResponseFormat(&providers.ResponseFormat{Type: responseFormatSchema})

		require.Nil(t, result)
		require.ErrorIs(t, err, errors.ErrInvalidRequest)
	})

	t.Run("rejects unencodable json schema", func(t *testing.T) {
		t.Parallel()

		result, err := convertResponseFormat(&providers.ResponseFormat{
			Type: responseFormatSchema,
			JSONSchema: &providers.JSONSchema{
				Schema: map[string]any{"invalid": make(chan int)},
			},
		})

		require.Nil(t, result)
		require.ErrorIs(t, err, errors.ErrInvalidRequest)
	})

	t.Run("accepts strict json schema", func(t *testing.T) {
		t.Parallel()

		result, err := convertResponseFormat(&providers.ResponseFormat{
			Type: responseFormatSchema,
			JSONSchema: &providers.JSONSchema{
				Schema: map[string]any{"type": "object"},
				Strict: new(true),
			},
		})

		require.NoError(t, err)
		require.JSONEq(t, `{"type":"object"}`, string(result))
	})

	t.Run("rejects disabled strict mode", func(t *testing.T) {
		t.Parallel()

		result, err := convertResponseFormat(&providers.ResponseFormat{
			Type: responseFormatSchema,
			JSONSchema: &providers.JSONSchema{
				Schema: map[string]any{"type": "object"},
				Strict: new(false),
			},
		})

		require.Nil(t, result)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	})

	t.Run("rejects unknown format", func(t *testing.T) {
		t.Parallel()

		result, err := convertResponseFormat(&providers.ResponseFormat{Type: "yaml"})

		require.Nil(t, result)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	})
}

func TestConvertMessageRejectsInvalidMetadata(t *testing.T) {
	t.Parallel()

	tests := []invalidMessageCase{
		{
			name:    "unknown role",
			msg:     providers.Message{Role: "developer", Content: "Hello"},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name:    "name on user message",
			msg:     providers.Message{Role: providers.RoleUser, Content: "Hello", Name: "alice"},
			wantErr: errors.ErrUnsupportedParam,
		},
		{
			name: "tool call on user message",
			msg: providers.Message{
				Role: providers.RoleUser, Content: "Hello", ToolCalls: []providers.ToolCall{{}},
			},
			wantErr: errors.ErrUnsupportedParam,
		},
		{
			name: "reasoning on user message",
			msg: providers.Message{
				Role: providers.RoleUser, Content: "Hello", Reasoning: &providers.Reasoning{Content: "why"},
			},
			wantErr: errors.ErrUnsupportedParam,
		},
	}

	runInvalidMessageCases(t, tests)
}

func TestConvertMessageRejectsInvalidContent(t *testing.T) {
	t.Parallel()

	tests := []invalidMessageCase{
		{
			name:    "unsupported content representation",
			msg:     providers.Message{Role: providers.RoleUser, Content: 42},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name:    "invalid dynamic content part",
			msg:     providers.Message{Role: providers.RoleUser, Content: []any{"text"}},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name: "unknown dynamic content field",
			msg: providers.Message{
				Role: providers.RoleUser,
				Content: []any{map[string]any{
					"type": "text", "text": "Hello", "future": true,
				}},
			},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name: "unknown content part",
			msg: providers.Message{
				Role: providers.RoleUser, Content: []providers.ContentPart{{Type: "audio"}},
			},
			wantErr: errors.ErrUnsupportedParam,
		},
	}

	runInvalidMessageCases(t, tests)
}

func TestConvertMessageRejectsInvalidImageContent(t *testing.T) {
	t.Parallel()

	tests := []invalidMessageCase{
		{
			name: "text part with image field",
			msg: providers.Message{
				Role: providers.RoleUser,
				Content: []providers.ContentPart{{
					Type: contentTypeText, ImageURL: &providers.ImageURL{URL: "data:image/png;base64,aA=="},
				}},
			},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name: "image part without URL",
			msg: providers.Message{
				Role: providers.RoleUser,
				Content: []providers.ContentPart{{
					Type: contentTypeImageURL,
				}},
			},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name: "invalid image base64",
			msg: providers.Message{
				Role: providers.RoleUser,
				Content: []providers.ContentPart{{
					Type: contentTypeImageURL, ImageURL: &providers.ImageURL{URL: "data:image/png;base64,!"},
				}},
			},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name: "unsupported image detail",
			msg: providers.Message{
				Role: providers.RoleUser,
				Content: []providers.ContentPart{{
					Type: contentTypeImageURL,
					ImageURL: &providers.ImageURL{
						URL: "data:image/png;base64,aA==", Detail: "high",
					},
				}},
			},
			wantErr: errors.ErrUnsupportedParam,
		},
	}

	runInvalidMessageCases(t, tests)
}

func TestConvertMessageRejectsInvalidToolCall(t *testing.T) {
	t.Parallel()

	tests := []invalidMessageCase{
		{
			name: "unknown tool call type",
			msg: providers.Message{
				Role: providers.RoleAssistant,
				ToolCalls: []providers.ToolCall{{
					Type: "custom", Function: providers.FunctionCall{Arguments: `{}`},
				}},
			},
			wantErr: errors.ErrUnsupportedParam,
		},
		{
			name: "missing tool function name",
			msg: providers.Message{
				Role: providers.RoleAssistant,
				ToolCalls: []providers.ToolCall{{
					Type: toolTypeFunction, Function: providers.FunctionCall{Arguments: `{}`},
				}},
			},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name: "non-object tool arguments",
			msg: providers.Message{
				Role: providers.RoleAssistant,
				ToolCalls: []providers.ToolCall{{
					Type:     toolTypeFunction,
					Function: providers.FunctionCall{Name: "get_weather", Arguments: `[]`},
				}},
			},
			wantErr: errors.ErrInvalidRequest,
		},
		{
			name: "null tool arguments",
			msg: providers.Message{
				Role: providers.RoleAssistant,
				ToolCalls: []providers.ToolCall{{
					Type:     toolTypeFunction,
					Function: providers.FunctionCall{Name: "get_weather", Arguments: `null`},
				}},
			},
			wantErr: errors.ErrInvalidRequest,
		},
	}

	runInvalidMessageCases(t, tests)
}

func TestConvertMessagesRejectsUnresolvedToolResult(t *testing.T) {
	t.Parallel()

	converted, err := convertMessages([]providers.Message{{
		Role: providers.RoleTool, Content: "sunny", ToolCallID: "missing",
	}})

	require.Nil(t, converted)
	require.ErrorIs(t, err, errors.ErrInvalidRequest)
}

func TestCompletionEntryPointsRejectInvalidMessages(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	params := providers.CompletionParams{Messages: []providers.Message{{Role: "developer"}}}

	completion, err := provider.Completion(t.Context(), params)
	require.Nil(t, completion)
	require.ErrorIs(t, err, errors.ErrInvalidRequest)

	chunks, streamErrors := provider.CompletionStream(t.Context(), params)
	require.Empty(t, chunks)
	require.ErrorIs(t, <-streamErrors, errors.ErrInvalidRequest)
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
