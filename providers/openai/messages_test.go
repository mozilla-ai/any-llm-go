package openai

import (
	"encoding/json"
	"testing"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/providers"
)

const (
	testContent  = "content"
	testImageURL = "https://example.com/image.png"
	testName     = "participant"
)

func TestConvertOpenAIMessageWireShapes(t *testing.T) {
	t.Parallel()

	textParts := []providers.ContentPart{{Type: contentTypeText, Text: testContent}}
	functionCall := []providers.ToolCall{{
		ID:       "call_123",
		Type:     toolTypeFunction,
		Function: providers.FunctionCall{Name: "lookup", Arguments: `{}`},
	}}
	tests := []struct {
		name    string
		message providers.Message
		want    string
	}{
		{
			name:    "assistant name",
			message: providers.Message{Role: providers.RoleAssistant, Content: testContent, Name: testName},
			want:    `{"role":"assistant","content":"content","name":"participant"}`,
		},
		{
			name:    "developer name",
			message: providers.Message{Role: providers.RoleDeveloper, Content: testContent, Name: testName},
			want:    `{"role":"developer","content":"content","name":"participant"}`,
		},
		{
			name:    "system name",
			message: providers.Message{Role: providers.RoleSystem, Content: testContent, Name: testName},
			want:    `{"role":"system","content":"content","name":"participant"}`,
		},
		{
			name:    "user name",
			message: providers.Message{Role: providers.RoleUser, Content: testContent, Name: testName},
			want:    `{"role":"user","content":"content","name":"participant"}`,
		},
		{
			name:    "assistant text parts",
			message: providers.Message{Role: providers.RoleAssistant, Content: textParts},
			want:    `{"role":"assistant","content":[{"type":"text","text":"content"}]}`,
		},
		{
			name:    "developer text parts",
			message: providers.Message{Role: providers.RoleDeveloper, Content: textParts},
			want:    `{"role":"developer","content":[{"type":"text","text":"content"}]}`,
		},
		{
			name:    "system text parts",
			message: providers.Message{Role: providers.RoleSystem, Content: textParts},
			want:    `{"role":"system","content":[{"type":"text","text":"content"}]}`,
		},
		{
			name: "tool text parts",
			message: providers.Message{
				Role: providers.RoleTool, Content: textParts, ToolCallID: "call_123",
			},
			want: `{"role":"tool","content":[{"type":"text","text":"content"}],"tool_call_id":"call_123"}`,
		},
		{
			name: "image detail",
			message: providers.Message{Role: providers.RoleUser, Content: []providers.ContentPart{{
				Type: contentTypeImageURL,
				ImageURL: &providers.ImageURL{
					URL: testImageURL, Detail: "high",
				},
			}}},
			want: `{
				"role":"user",
				"content":[{"type":"image_url","image_url":{
					"url":"https://example.com/image.png","detail":"high"
				}}]
			}`,
		},
		{
			name:    "empty assistant",
			message: providers.Message{Role: providers.RoleAssistant},
			want:    `{"role":"assistant","content":""}`,
		},
		{
			name:    "assistant tool call without content",
			message: providers.Message{Role: providers.RoleAssistant, ToolCalls: functionCall},
			want: `{
				"role":"assistant",
				"tool_calls":[{"id":"call_123","type":"function",
					"function":{"name":"lookup","arguments":"{}"}}]
			}`,
		},
	}

	// OpenAI's current Chat schema defines these role, name, content, image-detail,
	// and optional assistant-content shapes. Developer replaces system for o1 and
	// newer models.
	// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			requireOpenAIMessageJSON(t, tc.message, tc.want)
		})
	}
}

func requireOpenAIMessageJSON(t *testing.T, message providers.Message, want string) {
	t.Helper()

	converted, err := convertOpenAIMessage(message)
	require.NoError(t, err)
	requireMessageJSON(t, converted, want)
}

func requireMessageJSON(t *testing.T, message openaisdk.ChatCompletionMessageParamUnion, want string) {
	t.Helper()

	wire, err := json.Marshal(message)
	require.NoError(t, err)
	require.JSONEq(t, want, string(wire))
}
