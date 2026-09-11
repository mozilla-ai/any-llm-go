package anthropic

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
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

const (
	imagePartType = "image_url"
	testImageURL  = "https://example.com/image.png"
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
	require.True(t, caps.CompletionImage)
	require.True(t, caps.CompletionPDF)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.False(t, caps.Embedding) // Anthropic doesn't support embeddings.
	require.False(t, caps.ListModels)
}

func TestConvertParamsNormalizesSystemMessages(t *testing.T) {
	t.Parallel()

	request, err := new(Provider).convertParams(providers.CompletionParams{
		Model: "claude-opus-5",
		Messages: []providers.Message{
			{Role: providers.RoleSystem, Content: "global instruction"},
			{Role: providers.RoleUser, Content: "start"},
			{Role: providers.RoleSystem, Content: "instruction from this point"},
		},
	})
	require.NoError(t, err)

	body, err := json.Marshal(request)
	require.NoError(t, err)
	require.JSONEq(t, `{
		"model": "claude-opus-5",
		"max_tokens": 4096,
		"system": [{"type": "text", "text": "global instruction\ninstruction from this point"}],
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "start"}]}
		]
	}`, string(body))
}

func TestCompletionSerializesSupportedImageSources(t *testing.T) {
	t.Parallel()

	request, err := new(Provider).convertParams(providers.CompletionParams{
		Model: "claude-opus-5",
		Messages: []providers.Message{{
			Role: providers.RoleUser,
			Content: []providers.ContentPart{
				{Type: imagePartType, ImageURL: new(providers.ImageURL{URL: testImageURL})},
				{Type: imagePartType, ImageURL: new(providers.ImageURL{URL: "data:image/png;base64,aGVsbG8="})},
				{Type: "text", Text: "Compare the images."},
			},
		}},
	})
	require.NoError(t, err)

	body, err := json.Marshal(request)
	require.NoError(t, err)
	require.JSONEq(t, `{
		"model": "claude-opus-5",
		"max_tokens": 4096,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "image", "source": {"type": "url", "url": "https://example.com/image.png"}},
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
				{"type": "text", "text": "Compare the images."}
			]
		}]
	}`, string(body))
}

func TestCompletionRejectsInvalidImageContent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		image    *providers.ImageURL
		sentinel error
	}{
		{
			name:     "image without source",
			image:    nil,
			sentinel: errors.ErrInvalidRequest,
		},
		{
			name:     "image detail",
			image:    new(providers.ImageURL{URL: testImageURL, Detail: "high"}),
			sentinel: errors.ErrUnsupportedParam,
		},
		{
			name:     "data URL without base64 marker",
			image:    new(providers.ImageURL{URL: "data:image/png,aGVsbG8="}),
			sentinel: errors.ErrInvalidRequest,
		},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			_, err := new(Provider).convertParams(providers.CompletionParams{
				Model: "claude-opus-5",
				Messages: []providers.Message{{
					Role: providers.RoleUser,
					Content: []providers.ContentPart{{
						Type:     imagePartType,
						ImageURL: testCase.image,
					}},
				}},
			})
			require.ErrorIs(t, err, testCase.sentinel)
		})
	}
}

func TestConvertMessages(t *testing.T) {
	t.Parallel()

	t.Run("extracts system message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleSystem, Content: "You are a helpful assistant."},
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

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

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

		require.Equal(t, "First part.\nSecond part.", system)
		require.Len(t, result, 1)
	})

	t.Run("converts user message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
		}

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

		require.Empty(t, system)
		require.Len(t, result, 1)
	})

	t.Run("converts assistant message", func(t *testing.T) {
		t.Parallel()

		messages := []providers.Message{
			{Role: providers.RoleUser, Content: "Hello"},
			{Role: providers.RoleAssistant, Content: "Hi there!"},
		}

		result, system, err := convertMessages(messages)
		require.NoError(t, err)

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

		result, _, err := convertMessages(messages)
		require.NoError(t, err)

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

		result, _, err := convertMessages(messages)
		require.NoError(t, err)

		require.Len(t, result, 3)
	})

	t.Run("preserves tool result errors", func(t *testing.T) {
		t.Parallel()

		message, err := convertToolMessage(providers.Message{
			Role:              providers.RoleTool,
			Content:           "weather service unavailable",
			ToolCallID:        "call_123",
			ToolResultIsError: true,
		})
		require.NoError(t, err)
		body, err := json.Marshal(message)

		require.NoError(t, err)
		require.JSONEq(t, `{
			"role": "user",
			"content": [{
				"type": "tool_result",
				"tool_use_id": "call_123",
				"content": [{"type": "text", "text": "weather service unavailable"}],
				"is_error": true
			}]
		}`, string(body))
	})
}

func TestConvertMessagesRejectsUnrepresentableContent(t *testing.T) {
	t.Parallel()

	for _, message := range []providers.Message{
		{Role: "unknown", Content: "hello"},
		{Role: providers.RoleTool, Content: "result"},
		{Role: providers.RoleUser, Content: []providers.ContentPart{{Type: "audio"}}},
		{Role: providers.RoleUser, Content: []providers.ContentPart{{Type: "image_url"}}},
		{Role: providers.RoleAssistant, ToolCalls: []providers.ToolCall{
			{ID: "call_1", Function: providers.FunctionCall{Name: "lookup", Arguments: "null"}},
		}},
	} {
		_, _, err := convertMessages([]providers.Message{message})
		require.Error(t, err, "%+v", message)
	}
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

func TestStreamStateHandleTextDelta(t *testing.T) {
	t.Parallel()

	var state streamState
	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleTextDelta("Hello ")
	require.NotNil(t, chunk)
	require.Equal(t, "msg_123", chunk.ID)
	require.Equal(t, "claude-3", chunk.Model)
	require.Equal(t, "chat.completion.chunk", chunk.Object)
	require.Len(t, chunk.Choices, 1)
	require.Equal(t, "Hello ", chunk.Choices[0].Delta.Content)
}

func TestStreamStateHandleThinkingDelta(t *testing.T) {
	t.Parallel()

	var state streamState

	state.messageID = "msg_123"
	state.model = "claude-3"

	chunk := state.handleThinkingDelta("Let me think...")
	require.NotNil(t, chunk)
	require.Equal(t, "msg_123", chunk.ID)
	require.Len(t, chunk.Choices, 1)
	require.NotNil(t, chunk.Choices[0].Delta.Reasoning)
	require.Equal(t, "Let me think...", chunk.Choices[0].Delta.Reasoning.Content)
}

func TestStreamStateHandleInputJSONDelta(t *testing.T) {
	t.Parallel()

	t.Run("returns nil when no tool calls", func(t *testing.T) {
		t.Parallel()

		var state streamState

		chunk := state.handleInputJSONDelta("", `{"key":`)
		require.Nil(t, chunk)
	})

	t.Run("emits only the current fragment", func(t *testing.T) {
		t.Parallel()

		var state streamState

		state.messageID = "msg_123"
		state.model = "claude-3"

		chunk := state.handleInputJSONDelta("call_1", `{"location":`)
		require.NotNil(t, chunk)
		require.Equal(t, `{"location":`, chunk.Choices[0].Delta.ToolCalls[0].Function.Arguments)

		chunk2 := state.handleInputJSONDelta("call_1", `"Paris"}`)
		require.NotNil(t, chunk2)
		require.Equal(t, `"Paris"}`, chunk2.Choices[0].Delta.ToolCalls[0].Function.Arguments)
	})
}

func TestApplyThinking(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		model            string
		effort           providers.ReasoningEffort
		maxTokens        int64
		expectedThinking string
		expectedEffort   string
		expectedMax      int64
		wantError        bool
	}{
		{
			name:        "omitted effort remains omitted",
			effort:      "",
			expectedMax: 1000,
		},
		{
			name:             "none disables thinking explicitly",
			effort:           providers.ReasoningEffortNone,
			expectedThinking: `{"type":"disabled"}`,
			expectedMax:      1000,
		},
		{
			name:        "auto preserves the model default",
			effort:      providers.ReasoningEffortAuto,
			expectedMax: 1000,
		},
		{
			name:             "adaptive minimal maps to low effort and enables thinking",
			model:            "claude-sonnet-4-6",
			effort:           "minimal",
			expectedThinking: `{"type":"adaptive"}`,
			expectedEffort:   `{"effort":"low"}`,
			expectedMax:      1000,
		},
		{
			name:             "legacy low enables extended thinking",
			model:            "claude-sonnet-4-5",
			effort:           providers.ReasoningEffortLow,
			expectedThinking: `{"type":"enabled","budget_tokens":1024}`,
			expectedMax:      2048,
		},
		{
			name:             "legacy medium preserves larger caller max",
			model:            "claude-haiku-4-5-20251001",
			effort:           providers.ReasoningEffortMedium,
			maxTokens:        9000,
			expectedThinking: `{"type":"enabled","budget_tokens":4096}`,
			expectedMax:      9000,
		},
		{
			name:             "unknown alias retains legacy high behavior",
			model:            "bedrock-custom-alias",
			effort:           providers.ReasoningEffortHigh,
			expectedThinking: `{"type":"enabled","budget_tokens":16384}`,
			expectedMax:      32768,
		},
		{
			name:             "adaptive xhigh remains xhigh",
			model:            "claude-opus-4-7-20260901",
			effort:           "xhigh",
			expectedThinking: `{"type":"adaptive"}`,
			expectedEffort:   `{"effort":"xhigh"}`,
			expectedMax:      1000,
		},
		{
			name:             "Opus 4.5 uses extended thinking with effort",
			model:            "claude-opus-4-5",
			effort:           providers.ReasoningEffortLow,
			expectedThinking: `{"type":"enabled","budget_tokens":1024}`,
			expectedEffort:   `{"effort":"low"}`,
			expectedMax:      2048,
		},
		{
			name:      "legacy max is rejected",
			model:     "claude-sonnet-4-5",
			effort:    "max",
			wantError: true,
		},
		{
			name:             "similar version is not classified as adaptive",
			model:            "claude-sonnet-4-60",
			effort:           providers.ReasoningEffortLow,
			expectedThinking: `{"type":"enabled","budget_tokens":1024}`,
			expectedMax:      2048,
		},
		{
			name:      "unknown effort is rejected",
			effort:    "invalid",
			wantError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := new(anthropic.MessageNewParams)
			req.MaxTokens = tc.maxTokens
			if req.MaxTokens == 0 {
				req.MaxTokens = 1000
			}
			err := applyThinking(req, tc.effort, tc.model)

			if tc.wantError {
				require.Error(t, err)

				return
			}

			require.NoError(t, err)
			require.Equal(t, tc.expectedMax, req.MaxTokens)

			body, err := json.Marshal(req)
			require.NoError(t, err)

			var wire map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(body, &wire))

			if tc.expectedThinking == "" {
				require.NotContains(t, wire, "thinking")
			} else {
				require.JSONEq(t, tc.expectedThinking, string(wire["thinking"]))
			}

			if tc.expectedEffort == "" {
				require.NotContains(t, wire, "output_config")
			} else {
				require.JSONEq(t, tc.expectedEffort, string(wire["output_config"]))
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

			result, err := convertMessage(tc.msg)
			if tc.expectNil {
				require.Error(t, err)
				require.Nil(t, result)
			} else {
				require.NoError(t, err)
				require.NotNil(t, result)
			}
		})
	}
}

func TestConvertToolCall(t *testing.T) {
	t.Parallel()

	toolCall := providers.ToolCall{
		ID:   "call_123",
		Type: "function",
		Function: providers.FunctionCall{
			Name:      "get_weather",
			Arguments: `{"location": "Paris"}`,
		},
	}

	result, err := convertToolCall(toolCall)
	require.NoError(t, err)
	require.NotNil(t, result.OfToolUse)
	require.Equal(t, toolCall.ID, result.OfToolUse.ID)
	require.Equal(t, toolCall.Function.Name, result.OfToolUse.Name)
	require.Equal(t, "tool_use", string(result.OfToolUse.Type))
	require.NotNil(t, result.OfToolUse.Input)
}

func TestParallelToolsWithoutExplicitChoice(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name     string
		parallel *bool
		want     string
		noTools  bool
	}{
		{name: "omitted", want: ""},
		{name: "disabled", parallel: new(false), want: `{"type":"auto","disable_parallel_tool_use":true}`},
		{name: "enabled", parallel: new(true), want: `{"type":"auto","disable_parallel_tool_use":false}`},
		{name: "no tools", parallel: new(false), noTools: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			tools := []providers.Tool{testutil.DateTool()}
			if tc.noTools {
				tools = nil
			}

			request, err := new(Provider).convertParams(providers.CompletionParams{
				Tools: tools, ParallelToolCalls: tc.parallel,
			})
			require.NoError(t, err)
			encoded, err := json.Marshal(request)
			require.NoError(t, err)

			var body map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(encoded, &body))

			if tc.want == "" {
				require.NotContains(t, body, "tool_choice")

				return
			}

			require.JSONEq(t, tc.want, string(body["tool_choice"]))
		})
	}
}

func TestConvertTool(t *testing.T) {
	t.Parallel()

	t.Run("preserves complete schema on the wire", func(t *testing.T) {
		t.Parallel()

		const schema = `{"type":"object","properties":{},"required":[],"additionalProperties":false,` +
			`"$defs":{"id":{"type":"integer"}},"x-future":9007199254740993}`

		var parameters map[string]any

		decoder := json.NewDecoder(strings.NewReader(schema))
		decoder.UseNumber()
		require.NoError(t, decoder.Decode(&parameters))
		tool, err := convertTool(providers.Tool{
			Type:     "function",
			Function: providers.Function{Name: "lookup", Parameters: parameters},
		})
		require.NoError(t, err)
		encoded, err := json.Marshal(tool)
		require.NoError(t, err)

		var wire struct {
			InputSchema json.RawMessage `json:"input_schema"`
		}
		require.NoError(t, json.Unmarshal(encoded, &wire))
		require.JSONEq(t, schema, string(wire.InputSchema))
		require.Contains(t, string(wire.InputSchema), "9007199254740993")

		unchanged, err := json.Marshal(parameters)
		require.NoError(t, err)
		require.JSONEq(t, schema, string(unchanged))
	})

	t.Run("defaults missing type without mutating parameters", func(t *testing.T) {
		t.Parallel()

		for _, parameters := range []map[string]any{nil, {"properties": map[string]any{}}} {
			tool, err := convertTool(providers.Tool{Function: providers.Function{Parameters: parameters}})
			require.NoError(t, err)
			encoded, err := json.Marshal(tool.OfTool.InputSchema)
			require.NoError(t, err)
			require.Contains(t, string(encoded), `"type":"object"`)
			require.NotContains(t, parameters, "type")
		}
	})

	t.Run("reports non-JSON schema values", func(t *testing.T) {
		t.Parallel()

		_, err := convertTool(providers.Tool{Function: providers.Function{
			Name: "invalid", Parameters: map[string]any{"extension": make(chan int)},
		}})

		var unsupported *json.UnsupportedTypeError
		require.ErrorAs(t, err, &unsupported)
		require.Contains(t, err.Error(), "tool invalid: encode input schema")
	})

	t.Run("converts tool with properties and required fields", func(t *testing.T) {
		t.Parallel()

		tool := testutil.WeatherTool()
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "get_weather", result.OfTool.Name)
		require.Equal(t, "Get the current weather for a location.", result.OfTool.Description.Value)
		encoded, err := json.Marshal(result.OfTool.InputSchema)
		require.NoError(t, err)
		expected, err := json.Marshal(tool.Function.Parameters)
		require.NoError(t, err)
		require.JSONEq(t, string(expected), string(encoded))
	})

	t.Run("converts tool with multiple parameters", func(t *testing.T) {
		t.Parallel()

		tool := testutil.NewTestCalculatorTool(t)
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "calculate", result.OfTool.Name)
		encoded, err := json.Marshal(result.OfTool.InputSchema)
		require.NoError(t, err)
		expected, err := json.Marshal(tool.Function.Parameters)
		require.NoError(t, err)
		require.JSONEq(t, string(expected), string(encoded))
	})

	t.Run("converts tool with empty parameters", func(t *testing.T) {
		t.Parallel()

		tool := testutil.DateTool()
		result, err := convertTool(tool)

		require.NoError(t, err)
		require.NotNil(t, result.OfTool)
		require.Equal(t, "get_current_date", result.OfTool.Name)
		encoded, err := json.Marshal(result.OfTool.InputSchema)
		require.NoError(t, err)
		require.JSONEq(t, `{"type":"object","properties":{}}`, string(encoded))
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
	require.Positive(t, resp.Usage.TotalTokens)
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

	require.Positive(t, chunkCount)
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
	require.InDelta(t, 15.0, args.A, 0, "first operand should be 15")
	require.InDelta(t, 27.0, args.B, 0, "second operand should be 27")
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

func TestIntegrationCompletionWithStructuredOutput(t *testing.T) {
	t.Parallel()

	if testutil.SkipIfNoAPIKey("anthropic") {
		t.Skip("ANTHROPIC_API_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
		"required": []any{"answer"},
	}

	ctx := context.Background()
	result, err := provider.Completion(ctx, providers.CompletionParams{
		Model: testutil.TestModel("anthropic"),
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "What is 2+2? Respond using the provided schema."},
		},
		ResponseFormat: &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:   "answer_schema",
				Schema: schema,
			},
		},
	})
	require.NoError(t, err)
	require.NotEmpty(t, result.ID)
	require.Len(t, result.Choices, 1)
	require.NotEmpty(t, result.Choices[0].Message.Content)

	contentStr, ok := result.Choices[0].Message.Content.(string)
	require.True(t, ok, "expected string content")

	var response map[string]any
	err = json.Unmarshal([]byte(contentStr), &response)
	require.NoError(t, err, "response should be valid JSON")
	require.Contains(t, response, "answer", "response should contain 'answer' key")
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
			name:         "authentication type becomes AuthenticationError",
			err:          newTestAPIError(t, http.StatusUnauthorized, "authentication_error"),
			wantSentinel: errors.ErrAuthentication,
		},
		{
			name:         "rate-limit type becomes RateLimitError",
			err:          newTestAPIError(t, http.StatusTooManyRequests, "rate_limit_error"),
			wantSentinel: errors.ErrRateLimit,
		},
		{
			name:         "billing type becomes InsufficientFundsError",
			err:          newTestAPIError(t, http.StatusPaymentRequired, "billing_error"),
			wantSentinel: errors.ErrInsufficientFunds,
		},
		{
			name:         "invalid-request context message preserves ContextLengthError",
			err:          newTestAPIError(t, http.StatusBadRequest, "invalid_request_error"),
			wantSentinel: errors.ErrContextLength,
		},
		{
			name:         "permission type remains ProviderError",
			err:          newTestAPIError(t, http.StatusForbidden, "permission_error"),
			wantSentinel: errors.ErrProvider,
		},
		{
			name:         "not-found type preserves ModelNotFoundError",
			err:          newTestAPIError(t, http.StatusNotFound, "not_found_error"),
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "unknown 404 body preserves ModelNotFoundError",
			err:          newTestAPIError(t, http.StatusNotFound, "future_error"),
			wantSentinel: errors.ErrModelNotFound,
		},
		{
			name:         "request-too-large status remains ProviderError",
			err:          newTestAPIError(t, http.StatusRequestEntityTooLarge, "request_too_large"),
			wantSentinel: errors.ErrProvider,
		},
		{
			name:         "overloaded type becomes ProviderError",
			err:          newTestAPIError(t, 529, "overloaded_error"),
			wantSentinel: errors.ErrProvider,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			p := &Provider{}
			result := p.ConvertError(tc.err)

			if tc.wantSentinel == nil {
				require.NoError(t, result)
				return
			}

			require.Error(t, result)
			require.ErrorIs(t, result, tc.wantSentinel, "expected error to match %v", tc.wantSentinel)

			// Verify the provider name is set in the error message.
			require.Contains(t, result.Error(), "["+providerName+"]")
		})
	}
}

func TestInvalidRequestClassificationPreservesContextLengthCompatibility(t *testing.T) {
	t.Parallel()

	for _, message := range []string{"context_length exceeded", "prompt is too long: 201 tokens > 200 maximum"} {
		body := fmt.Sprintf(`{"type":"error","error":{"type":"invalid_request_error","message":%q}}`, message)

		var apiErr anthropic.Error
		require.NoError(t, json.Unmarshal([]byte(body), &apiErr))
		apiErr.StatusCode = http.StatusBadRequest
		converted := new(Provider).ConvertError(&apiErr)
		require.ErrorIs(t, converted, errors.ErrContextLength)

		var original *anthropic.Error
		require.ErrorAs(t, converted, &original)
		require.Same(t, &apiErr, original)
	}
}

func TestConvertErrorPreservesRetryAfter(t *testing.T) {
	t.Parallel()

	apiErr := newTestAPIError(t, http.StatusTooManyRequests, "rate_limit_error")
	apiErr.Response.Header.Set("Retry-After", "17")

	converted := new(Provider).ConvertError(apiErr)
	rateLimitErr, ok := stderrors.AsType[*errors.RateLimitError](converted)
	require.True(t, ok)
	require.Equal(t, 17, rateLimitErr.RetryAfter)
}

func TestConvertParams_ResponseFormat(t *testing.T) {
	t.Parallel()

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer": map[string]any{"type": "string"},
		},
		"required": []any{"answer"},
	}

	p, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)

	baseParams := func() providers.CompletionParams {
		return providers.CompletionParams{
			Model: "claude-3-5-haiku-20241022",
			Messages: []providers.Message{
				{Role: providers.RoleUser, Content: []providers.ContentPart{{Type: "text", Text: "hello"}}},
			},
		}
	}

	t.Run("nil ResponseFormat leaves OutputConfig unset", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ResponseFormat = nil

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, anthropic.OutputConfigParam{}, result.OutputConfig)
	})

	t.Run("json_object type is no-op", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ResponseFormat = &providers.ResponseFormat{Type: responseFormatJSONObject}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, anthropic.OutputConfigParam{}, result.OutputConfig)
	})

	t.Run("json_schema with nil JSONSchema is no-op", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ResponseFormat = &providers.ResponseFormat{Type: responseFormatJSONSchema, JSONSchema: nil}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, anthropic.OutputConfigParam{}, result.OutputConfig)
	})

	t.Run("json_schema with valid schema sets OutputConfig", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.ResponseFormat = &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:   "answer_schema",
				Schema: schema,
			},
		}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, schema, result.OutputConfig.Format.Schema)
	})

	t.Run("adaptive effort preserves the structured output format", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.Model = "claude-sonnet-4-6"
		params.ReasoningEffort = providers.ReasoningEffortHigh
		params.ResponseFormat = &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:   "answer_schema",
				Schema: schema,
			},
		}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, schema, result.OutputConfig.Format.Schema)
		require.Equal(t, anthropic.OutputConfigEffortHigh, result.OutputConfig.Effort)
		require.NotNil(t, result.Thinking.OfAdaptive)
	})

	t.Run("unsupported JSONSchema fields are not forwarded", func(t *testing.T) {
		t.Parallel()

		strict := true
		params := baseParams()
		params.ResponseFormat = &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:        "answer_schema",
				Description: "A schema for answers",
				Strict:      &strict,
				Schema:      schema,
			},
		}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, schema, result.OutputConfig.Format.Schema)
		// JSONOutputFormatParam only has Schema and Type fields;
		// Name, Description, and Strict have no destination.
		require.Equal(t, anthropic.JSONOutputFormatParam{Schema: schema}, result.OutputConfig.Format)
	})

	t.Run("streaming path also receives OutputConfig", func(t *testing.T) {
		t.Parallel()

		params := baseParams()
		params.Stream = true
		params.ResponseFormat = &providers.ResponseFormat{
			Type: responseFormatJSONSchema,
			JSONSchema: &providers.JSONSchema{
				Name:   "answer_schema",
				Schema: schema,
			},
		}

		result, err := p.convertParams(params)
		require.NoError(t, err)
		require.Equal(t, schema, result.OutputConfig.Format.Schema)
	})
}

func newTestAPIError(t *testing.T, statusCode int, errorType string) *anthropic.Error {
	t.Helper()

	var apiErr anthropic.Error

	body := fmt.Sprintf(
		`{"type":"error","error":{"type":%q,"message":"token and safety content"}}`,
		errorType,
	)
	require.NoError(t, json.Unmarshal([]byte(body), &apiErr))

	testURL, err := url.Parse("https://api.anthropic.com/v1/messages")
	require.NoError(t, err)

	apiErr.StatusCode = statusCode
	apiErr.RequestID = "req_test123"
	apiErr.Request = &http.Request{Method: http.MethodPost, URL: testURL}
	apiErr.Response = &http.Response{StatusCode: statusCode, Header: make(http.Header)}

	return &apiErr
}
