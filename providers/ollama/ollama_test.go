package ollama

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"net/http"
	"net/http/httptest"
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

const (
	testOllamaAvailabilityTimeout = 5 * time.Second
	// This fixture is independently derived from the current Chat OpenAPI and
	// tool-calling guide, so it does not reuse the converter's assumptions.
	documentedChatRequestJSON = `{
		"model":"model",
		"messages":[
			{"role":"user","content":"look","images":["aGVsbG8="]},
			{
				"role":"assistant",
				"content":"",
				"thinking":"thought",
				"tool_calls":[{
					"function":{
						"index":0,
						"name":"weather",
						"arguments":{"city":"Paris"}
					}
				}]
			},
			{"role":"tool","content":"sunny","tool_name":"weather"}
		],
		"options":{"num_ctx":32000}
	}`
)

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

func TestCompletionSendsEstablishedContextDefault(t *testing.T) {
	t.Parallel()

	requestBody := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		requestBody <- body
		w.Header().Set("Content-Type", "application/json")
		_, err := fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"ok"},"done":true}`)
		require.NoError(t, err)
	}))
	t.Cleanup(server.Close)

	provider, err := New(config.WithBaseURL(server.URL))
	require.NoError(t, err)
	_, err = provider.Completion(t.Context(), providers.CompletionParams{
		Model:    "test",
		Messages: []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
	})
	require.NoError(t, err)

	body := <-requestBody
	options, ok := body["options"].(map[string]any)
	require.True(t, ok)
	require.Equal(t, float64(32000), options["num_ctx"])
}

func TestConvertParamsPreservesDocumentedReasoningControls(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	none, err := provider.convertParams(providers.CompletionParams{ReasoningEffort: providers.ReasoningEffortNone})
	require.NoError(t, err)
	require.Equal(t, false, none.Think.Value)

	// These wire values match Ollama's MIT-licensed api/types_test.go fixtures.
	for _, testCase := range []struct {
		name   string
		effort providers.ReasoningEffort
		want   string
	}{
		{name: "low", effort: providers.ReasoningEffortLow, want: "low"},
		{name: "medium", effort: providers.ReasoningEffortMedium, want: "medium"},
		{name: "high", effort: providers.ReasoningEffortHigh, want: "high"},
		{name: "max", effort: providers.ReasoningEffort("max"), want: "max"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			converted, convertErr := provider.convertParams(providers.CompletionParams{
				ReasoningEffort: testCase.effort,
			})
			require.NoError(t, convertErr)
			require.Equal(t, testCase.want, converted.Think.Value)
		})
	}

	req, err := provider.convertParams(providers.CompletionParams{ReasoningEffort: providers.ReasoningEffortAuto})
	require.NoError(t, err)
	require.Nil(t, req.Think)

	_, err = provider.convertParams(providers.CompletionParams{ReasoningEffort: providers.ReasoningEffort("xhigh")})
	require.ErrorIs(t, err, errors.ErrUnsupportedParam)
}

func TestConvertParamsRejectsUnsupportedFields(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	for _, params := range []providers.CompletionParams{
		{StreamOptions: &providers.StreamOptions{}},
		{ToolChoice: "auto"},
		{ParallelToolCalls: new(false)},
		{User: "user-1"},
		{Extra: map[string]any{"unknown": true}},
	} {
		_, err := provider.convertParams(params)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	}
}

func TestConvertParamsMatchesDocumentedMessageWire(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	req, err := provider.convertParams(providers.CompletionParams{
		Model: "model",
		Messages: []providers.Message{
			{
				Role: providers.RoleUser,
				Content: []providers.ContentPart{
					{Type: contentTypeText, Text: "look"},
					{Type: contentTypeImageURL, ImageURL: &providers.ImageURL{URL: "data:image/png;base64,aGVsbG8="}},
				},
			},
			{
				Role:      providers.RoleAssistant,
				Reasoning: &providers.Reasoning{Content: "thought"},
				ToolCalls: []providers.ToolCall{{
					ID:   "call_1",
					Type: toolTypeFunction,
					Function: providers.FunctionCall{
						Name:      "weather",
						Arguments: `{"city":"Paris"}`,
					},
				}},
			},
			{Role: providers.RoleTool, Content: "sunny", ToolCallID: "call_1"},
		},
	})
	require.NoError(t, err)

	encoded, err := json.Marshal(req)
	require.NoError(t, err)
	require.JSONEq(t, documentedChatRequestJSON, string(encoded))
}

func TestConvertMessages(t *testing.T) {
	t.Parallel()

	t.Run("converts system message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "You are a helpful assistant."},
		}

		result, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleSystem, result[0].Role)
		require.Equal(t, "You are a helpful assistant.", result[0].Content)
	})

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleUser, result[0].Role)
		require.Equal(t, "Hello", result[0].Content)
	})

	t.Run("converts assistant message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleAssistant, Content: "Hi there!"},
		}

		result, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleAssistant, result[0].Role)
		require.Equal(t, "Hi there!", result[0].Content)
	})

	t.Run("converts tool message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleTool, Content: "sunny, 22°C", Name: "get_weather", ToolCallID: "call_123"},
		}

		result, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleTool, result[0].Role)
		require.Equal(t, "get_weather", result[0].ToolName)
		require.Empty(t, result[0].ToolCallID)
	})

	t.Run("converts assistant message with tool calls", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{
				Role:    providers.RoleAssistant,
				Content: "",
				ToolCalls: []providers.ToolCall{
					{
						ID:   "call_123",
						Type: toolTypeFunction,
						Function: providers.FunctionCall{
							Name:      "get_weather",
							Arguments: `{"location": "Paris"}`,
						},
					},
				},
			},
		}

		result, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 1)
		require.Equal(t, providers.RoleAssistant, result[0].Role)
		require.Len(t, result[0].ToolCalls, 1)
		require.Empty(t, result[0].ToolCalls[0].ID)
		require.Equal(t, "get_weather", result[0].ToolCalls[0].Function.Name)
	})

	t.Run("rejects an unidentified tool result", func(t *testing.T) {
		t.Parallel()

		_, err := convertMessages([]providers.Message{{
			Role:       providers.RoleTool,
			Content:    "result",
			ToolCallID: "missing",
		}})
		require.ErrorIs(t, err, errors.ErrInvalidRequest)
	})
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
			expected: "unknown",
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

func TestConvertMessageContent(t *testing.T) {
	t.Parallel()

	t.Run("extracts base64 image", func(t *testing.T) {
		t.Parallel()

		msg := providers.Message{
			Role: providers.RoleUser,
			Content: []providers.ContentPart{
				{Type: "text", Text: "What's in this image?"},
				{
					Type: "image_url",
					ImageURL: &providers.ImageURL{
						URL: "data:image/jpeg;base64,aGVsbG8=",
					},
				},
			},
		}

		content, images, err := convertMessageContent(msg)
		require.NoError(t, err)

		require.Equal(t, "What's in this image?", content)
		require.Len(t, images, 1)
		require.Equal(t, "hello", string(images[0]))
	})

	t.Run("rejects non-data URLs", func(t *testing.T) {
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

		_, images, err := convertMessageContent(msg)

		require.Empty(t, images)
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	})
}

func TestConvertTools(t *testing.T) {
	t.Parallel()

	tools := []providers.Tool{
		{
			Type: toolTypeFunction,
			Function: providers.Function{
				Name:        "get_weather",
				Description: "Get the current weather",
				Parameters: map[string]any{
					schemaKeyType: schemaTypeObject,
					schemaKeyProperties: map[string]any{
						"location": map[string]any{
							schemaKeyType:        "string",
							schemaKeyDescription: "The city name",
						},
					},
					schemaKeyRequired: []any{"location"},
				},
			},
		},
	}

	result, err := convertTools(tools)
	require.NoError(t, err)

	require.Len(t, result, 1)
	require.Equal(t, toolTypeFunction, result[0].Type)
	require.Equal(t, "get_weather", result[0].Function.Name)
	require.Equal(t, "Get the current weather", result[0].Function.Description)
	require.Contains(t, result[0].Function.Parameters.Required, "location")
	require.Equal(t, "The city name", result[0].Function.Parameters.Properties.ToMap()["location"].Description)

	for _, test := range []struct {
		parameters     map[string]any
		wantProperties bool
	}{
		{parameters: nil},
		{parameters: map[string]any{}},
		{parameters: map[string]any{schemaKeyType: schemaTypeObject}},
		{parameters: map[string]any{schemaKeyType: schemaTypeObject, schemaKeyProperties: map[string]any{}, schemaKeyRequired: []any{}}, wantProperties: true},
		{parameters: map[string]any{schemaKeyType: schemaTypeObject, schemaKeyProperties: map[string]any{"city": map[string]any{
			schemaKeyType: "string", schemaKeyDescription: "",
		}}}, wantProperties: true},
	} {
		parameters := test.parameters
		before, marshalErr := json.Marshal(parameters)
		require.NoError(t, marshalErr)

		converted, convertErr := convertToolParameters(parameters)
		require.NoError(t, convertErr)
		require.Equal(t, schemaTypeObject, converted.Type)
		require.Equal(t, test.wantProperties, converted.Properties != nil)

		after, marshalErr := json.Marshal(parameters)
		require.NoError(t, marshalErr)
		require.JSONEq(t, string(before), string(after))
	}

	_, err = convertTools([]providers.Tool{{Type: "custom"}})
	require.ErrorIs(t, err, errors.ErrUnsupportedParam)

	_, err = convertTools([]providers.Tool{{
		Type: toolTypeFunction,
		Function: providers.Function{
			Parameters: map[string]any{
				schemaKeyType:          schemaTypeObject,
				"additionalProperties": false,
			},
		},
	}})
	require.ErrorIs(t, err, errors.ErrUnsupportedParam)

	for _, minimum := range []json.Number{"7", "9007199254740993"} {
		_, err = convertTools([]providers.Tool{{
			Type: toolTypeFunction,
			Function: providers.Function{
				Parameters: map[string]any{
					schemaKeyType: schemaTypeObject,
					schemaKeyProperties: map[string]any{
						"count": map[string]any{schemaKeyType: "integer", "minimum": minimum},
					},
				},
			},
		}})
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	}

	_, err = convertTools([]providers.Tool{{
		Type: toolTypeFunction,
		Function: providers.Function{
			Name:        "",
			Description: "",
			Parameters: map[string]any{
				schemaKeyType: schemaTypeObject,
				"$defs": map[string]any{
					"large": map[string]any{schemaKeyType: "integer", "minimum": json.Number("9007199254740993")},
				},
			},
		},
	}})
	require.ErrorIs(t, err, errors.ErrUnsupportedParam)
}

func TestConvertToolCalls(t *testing.T) {
	t.Parallel()

	args := api.NewToolCallFunctionArguments()
	args.Set("location", "Paris")

	toolCalls := []api.ToolCall{
		{
			ID: "provider-call-id",
			Function: api.ToolCallFunction{
				Name:      "get_weather",
				Arguments: args,
			},
		},
		{Function: api.ToolCallFunction{Name: "no_id", Arguments: args}},
	}

	result := convertToolCalls(toolCalls)

	require.Len(t, result, 2)
	require.Equal(t, "provider-call-id", result[0].ID)
	require.Equal(t, toolTypeFunction, result[0].Type)
	require.Equal(t, "get_weather", result[0].Function.Name)
	require.Contains(t, result[0].Function.Arguments, "Paris")
	require.Equal(t, "call_1", result[1].ID)
}

func TestConvertResponsePreservesThinking(t *testing.T) {
	t.Parallel()

	for _, testCase := range []struct {
		name      string
		content   string
		thinking  string
		want      string
		reasoning *providers.Reasoning
	}{
		{
			name: "native thinking takes precedence", content: "<think>tagged</think>answer", thinking: "native",
			want: "<think>tagged</think>answer", reasoning: &providers.Reasoning{Content: "native"},
		},
		{
			name: "legacy tags", content: "<think>private chain</think>public answer",
			want: "public answer", reasoning: &providers.Reasoning{Content: "private chain"},
		},
		{
			name: "surrounding content", content: "  before<think>reason</think>after  ",
			want: "beforeafter", reasoning: &providers.Reasoning{Content: "reason"},
		},
		{
			name: "first complete block", content: "<think>first</think><think>second</think>answer",
			want: "<think>second</think>answer", reasoning: &providers.Reasoning{Content: "first"},
		},
		{
			name: "empty tagged thinking", content: "<think></think>answer",
			want: "answer", reasoning: &providers.Reasoning{},
		},
		{name: "plain content", content: "  public answer  ", want: "  public answer  "},
		{name: "unclosed tag", content: "<think>unfinished", want: "<think>unfinished"},
		{name: "unopened tag", content: "answer</think>", want: "answer</think>"},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			result := convertResponse(&api.ChatResponse{
				Message: api.Message{Content: testCase.content, Thinking: testCase.thinking},
			})
			require.Equal(t, testCase.want, result.Choices[0].Message.Content)
			require.Equal(t, testCase.reasoning, result.Choices[0].Message.Reasoning)
		})
	}
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

	t.Run("json_schema format", func(t *testing.T) {
		t.Parallel()

		format := &providers.ResponseFormat{
			Type: responseFormatSchema,
			JSONSchema: &providers.JSONSchema{
				Name: "test",
				Schema: map[string]any{
					schemaKeyType: schemaTypeObject,
					schemaKeyProperties: map[string]any{
						"name": map[string]any{schemaKeyType: "string"},
					},
				},
			},
		}
		result, err := convertResponseFormat(format)
		require.NoError(t, err)

		require.NotNil(t, result)
		require.Contains(t, string(result), schemaKeyProperties)
	})

	t.Run("rejects missing json schema", func(t *testing.T) {
		t.Parallel()

		_, err := convertResponseFormat(&providers.ResponseFormat{Type: responseFormatSchema})
		require.ErrorIs(t, err, errors.ErrInvalidRequest)
	})

	t.Run("rejects unencodable json schema", func(t *testing.T) {
		t.Parallel()

		_, err := convertResponseFormat(&providers.ResponseFormat{
			Type: responseFormatSchema,
			JSONSchema: &providers.JSONSchema{
				Schema: map[string]any{"value": make(chan int)},
			},
		})
		require.ErrorIs(t, err, errors.ErrInvalidRequest)
	})

	t.Run("accepts explicit non-strict json schema", func(t *testing.T) {
		t.Parallel()

		result, err := convertResponseFormat(&providers.ResponseFormat{
			Type: responseFormatSchema,
			JSONSchema: &providers.JSONSchema{
				Schema: map[string]any{"type": "object"},
				Strict: new(false),
			},
		})
		require.NoError(t, err)
		require.JSONEq(t, `{"type":"object"}`, string(result))
	})

	t.Run("rejects strict json schema", func(t *testing.T) {
		t.Parallel()

		_, err := convertResponseFormat(&providers.ResponseFormat{
			Type:       responseFormatSchema,
			JSONSchema: &providers.JSONSchema{Schema: map[string]any{}, Strict: new(true)},
		})
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	})

	t.Run("rejects unknown format", func(t *testing.T) {
		t.Parallel()

		_, err := convertResponseFormat(&providers.ResponseFormat{Type: "yaml"})
		require.ErrorIs(t, err, errors.ErrUnsupportedParam)
	})
}

func TestConvertMessageRejectsInvalidInput(t *testing.T) {
	t.Parallel()

	for _, testCase := range []struct {
		name     string
		message  providers.Message
		sentinel error
	}{
		{
			name:     "unknown role",
			message:  providers.Message{Role: "developer", Content: "content"},
			sentinel: errors.ErrInvalidRequest,
		},
		{
			name:     "name on user message",
			message:  providers.Message{Role: providers.RoleUser, Content: "content", Name: "name"},
			sentinel: errors.ErrUnsupportedParam,
		},
		{
			name: "tool call on user message",
			message: providers.Message{
				Role:      providers.RoleUser,
				Content:   "content",
				ToolCalls: []providers.ToolCall{{Type: toolTypeFunction}},
			},
			sentinel: errors.ErrUnsupportedParam,
		},
		{
			name: "reasoning on user message",
			message: providers.Message{
				Role:      providers.RoleUser,
				Content:   "content",
				Reasoning: &providers.Reasoning{Content: "thought"},
			},
			sentinel: errors.ErrUnsupportedParam,
		},
		{
			name: "unsupported tool-call type",
			message: providers.Message{
				Role: providers.RoleAssistant,
				ToolCalls: []providers.ToolCall{{
					Type:     "custom",
					Function: providers.FunctionCall{Name: "tool", Arguments: `{}`},
				}},
			},
			sentinel: errors.ErrUnsupportedParam,
		},
		{
			name: "invalid tool arguments",
			message: providers.Message{
				Role: providers.RoleAssistant,
				ToolCalls: []providers.ToolCall{{
					Type:     toolTypeFunction,
					Function: providers.FunctionCall{Name: "tool", Arguments: "["},
				}},
			},
			sentinel: errors.ErrInvalidRequest,
		},
		{
			name: "non-object tool arguments",
			message: providers.Message{
				Role: providers.RoleAssistant,
				ToolCalls: []providers.ToolCall{{
					Type:     toolTypeFunction,
					Function: providers.FunctionCall{Name: "tool", Arguments: `[]`},
				}},
			},
			sentinel: errors.ErrInvalidRequest,
		},
		{
			name:     "unparseable content part",
			message:  providers.Message{Role: providers.RoleUser, Content: []any{"content"}},
			sentinel: errors.ErrInvalidRequest,
		},
		{
			name:     "unsupported content representation",
			message:  providers.Message{Role: providers.RoleUser, Content: 42},
			sentinel: errors.ErrInvalidRequest,
		},
		{
			name:     "unknown content type",
			message:  providers.Message{Role: providers.RoleUser, Content: []providers.ContentPart{{Type: "audio"}}},
			sentinel: errors.ErrUnsupportedParam,
		},
		{
			name: "text with image field",
			message: providers.Message{
				Role: providers.RoleUser,
				Content: []providers.ContentPart{{
					Type:     contentTypeText,
					Text:     "content",
					ImageURL: &providers.ImageURL{URL: "data:image/png;base64,aGVsbG8="},
				}},
			},
			sentinel: errors.ErrInvalidRequest,
		},
		{
			name:     "missing image URL",
			message:  providers.Message{Role: providers.RoleUser, Content: []providers.ContentPart{{Type: contentTypeImageURL}}},
			sentinel: errors.ErrInvalidRequest,
		},
		{
			name: "invalid image base64",
			message: providers.Message{
				Role: providers.RoleUser,
				Content: []providers.ContentPart{{
					Type:     contentTypeImageURL,
					ImageURL: &providers.ImageURL{URL: "data:image/png;base64,%%%"},
				}},
			},
			sentinel: errors.ErrInvalidRequest,
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			_, err := convertMessage(testCase.message)
			require.ErrorIs(t, err, testCase.sentinel)
		})
	}
}

func TestNewStreamState(t *testing.T) {
	t.Parallel()

	state := newStreamState()
	require.NotNil(t, state)
	require.NotEmpty(t, state.id)
	require.Positive(t, state.created)
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
			err:          stderrors.New("connection refused"),
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
			err:          stderrors.New("some other error"),
			wantSentinel: errors.ErrProvider,
		},
		{
			name:         "StatusError 500 becomes ProviderError",
			err:          api.StatusError{StatusCode: 500, ErrorMessage: "internal error"},
			wantSentinel: errors.ErrProvider,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := &Provider{}
			result := p.ConvertError(tc.err)

			if tc.wantNil {
				require.NoError(t, result)
				return
			}

			require.Error(t, result)
			require.ErrorIs(t, result, tc.wantSentinel)
			require.ErrorContains(t, result, tc.err.Error())
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

func newTestProvider(t *testing.T, handler http.Handler) *Provider {
	t.Helper()

	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	provider, err := New(config.WithBaseURL(server.URL))
	require.NoError(t, err)

	return provider
}

func TestCompletionRequiresTerminalResponse(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"partial"},"done":false}`)
	}))

	_, err := provider.Completion(t.Context(), providers.CompletionParams{Model: "test"})
	require.ErrorIs(t, err, errors.ErrProvider)
}

func TestCompletionAcceptsUnknownResponseFields(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprintln(
			w,
			`{"model":"test","message":{"role":"assistant","content":"done","future":"value"},"done":true,"future":"value"}`,
		)
	}))

	response, err := provider.Completion(t.Context(), providers.CompletionParams{Model: "test"})
	require.NoError(t, err)
	require.Equal(t, "done", response.Choices[0].Message.Content)
}

func TestCompletionPreservesCallerTimeout(t *testing.T) {
	t.Parallel()

	releaseHandler := make(chan struct{})
	defer close(releaseHandler)

	provider := newTestProvider(t, http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		<-releaseHandler
	}))

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()

	_, err := provider.Completion(ctx, providers.CompletionParams{Model: "test"})
	require.ErrorIs(t, err, errors.ErrProvider)
	require.ErrorIs(t, err, context.DeadlineExceeded)
}

func TestCompletionStreamRequiresTerminalResponse(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"partial"},"done":false}`)
	}))

	chunks, errs := provider.CompletionStream(t.Context(), providers.CompletionParams{Model: "test"})

	chunkCount := 0
	for range chunks {
		chunkCount++
	}

	require.Equal(t, 1, chunkCount)
	require.ErrorIs(t, <-errs, errors.ErrProvider)
}

func TestCompletionStreamReturnsMidstreamError(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"partial"},"done":false}`)
		_, _ = fmt.Fprintln(w, `{"error":"generation failed"}`)
	}))

	chunks, errs := provider.CompletionStream(t.Context(), providers.CompletionParams{Model: "test"})

	count := 0
	for range chunks {
		count++
	}

	require.Equal(t, 1, count)
	require.ErrorIs(t, <-errs, errors.ErrProvider)
}

func TestCompletionStreamCancellationUnblocksAnUnreadConsumer(t *testing.T) {
	t.Parallel()

	wroteChunk := make(chan struct{})
	provider := newTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprintln(w, `{"model":"test","message":{"role":"assistant","content":"partial"},"done":false}`)

		flusher, ok := w.(http.Flusher)
		if !ok {
			return
		}

		flusher.Flush()
		close(wroteChunk)
		<-request.Context().Done()
	}))

	ctx, cancel := context.WithCancel(t.Context())
	chunks, errs := provider.CompletionStream(ctx, providers.CompletionParams{Model: "test"})

	<-wroteChunk
	cancel()

	select {
	case err := <-errs:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("stream did not stop after cancellation")
	}

	_, open := <-chunks
	require.False(t, open)
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

	require.Positive(t, chunkCount)
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
