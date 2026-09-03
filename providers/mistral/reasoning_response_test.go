package mistral

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const (
	jsonObjectTypeError    = "cannot unmarshal object"
	mistralReasoningModel  = "mistral-small-latest"
	mistralThinkingContent = `[
	{"type":"thinking","thinking":[
		{"type":"text","text":"step one","reference_ids":[{}]},
		{"type":"reference","reference_ids":[1,1.0,1e2,999999999999999999999999,"ref-1"],"text":{}},
		{"type":"tool_reference","tool":"web_search","title":"source","url":null},
		{"type":"text","text":"step two"}
	],"signature":null,"closed":false},
	{"type":"text","text":"the answer"},
	{"type":"future_chunk","text":{},"future":true}
]`
)

func mistralCompletionServer(t *testing.T, responseContent string) string {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(responseWriter http.ResponseWriter, _ *http.Request) {
		responseWriter.Header().Set("Content-Type", "application/json")
		_, err := fmt.Fprintf(responseWriter, `{
			"id":"chatcmpl-test",
			"object":"chat.completion",
			"created":1700000000,
			"model":"%s",
			"choices":[{
				"index":0,
				"message":{"role":"assistant","content":%s},
				"finish_reason":"stop"
			}]
		}`, mistralReasoningModel, responseContent)
		if err != nil {
			t.Errorf("writing Mistral response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	return server.URL
}

func TestCompletionPreservesThinkingChunks(t *testing.T) {
	t.Parallel()

	// Mistral returns chunk arrays for high reasoning and requires callers to
	// replay the complete array in later turns.
	// https://docs.mistral.ai/studio-api/conversations/reasoning
	serverURL := mistralCompletionServer(t, mistralThinkingContent)

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:    mistralReasoningModel,
		Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)

	message := completion.Choices[0].Message
	require.Equal(t, "the answer", message.Content)
	require.NotNil(t, message.Reasoning)
	require.Equal(t, "step one\nstep two", message.Reasoning.Content)
	require.JSONEq(t, mistralThinkingContent, string(message.Reasoning.ProviderRaw))
}

func TestCompletionPreservesReasoningWithoutAnswer(t *testing.T) {
	t.Parallel()

	const content = `[{"type":"thinking","thinking":[],"closed":false}]`
	serverURL := mistralCompletionServer(t, content)
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(serverURL),
	)
	require.NoError(t, err)

	completion, err := provider.Completion(t.Context(), providers.CompletionParams{
		Model:    mistralReasoningModel,
		Messages: testutil.SimpleMessages(),
	})
	require.NoError(t, err)

	message := completion.Choices[0].Message
	require.Nil(t, message.Content)
	require.NotNil(t, message.Reasoning)
	require.Empty(t, message.Reasoning.Content)
	require.JSONEq(t, content, string(message.Reasoning.ProviderRaw))
}

func TestCompletionRejectsMalformedKnownContentChunks(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		content string
		wantErr string
	}{
		{name: "text field has wrong type", content: `[{"type":"text","text":1}]`, wantErr: "cannot unmarshal number"},
		{name: "text field is missing", content: `[{"type":"text"}]`, wantErr: "text content chunk is missing text"},
		{
			name:    "thinking field has wrong type",
			content: `[{"type":"thinking","thinking":{}}]`,
			wantErr: jsonObjectTypeError,
		},
		{
			name:    "thinking field is missing",
			content: `[{"type":"thinking"}]`,
			wantErr: "thinking content chunk is missing thinking",
		},
		{
			name:    "thinking text field is missing",
			content: `[{"type":"thinking","thinking":[{"type":"text"}]}]`,
			wantErr: "thinking text chunk is missing text",
		},
		{
			name:    "thinking reference IDs are missing",
			content: `[{"type":"thinking","thinking":[{"type":"reference"}]}]`,
			wantErr: "thinking reference chunk is missing reference_ids",
		},
		{
			name:    "thinking reference ID has wrong type",
			content: `[{"type":"thinking","thinking":[{"type":"reference","reference_ids":[{}]}]}]`,
			wantErr: "thinking reference_id must be an integer or string",
		},
		{
			name:    "thinking tool title is missing",
			content: `[{"type":"thinking","thinking":[{"type":"tool_reference","tool":"web_search"}]}]`,
			wantErr: "thinking tool_reference chunk is missing tool or title",
		},
		{
			name: "thinking tool URL has wrong type",
			content: `[
				{"type":"thinking","thinking":[
					{"type":"tool_reference","tool":"web_search","title":"source","url":{}}
				]}
			]`,
			wantErr: jsonObjectTypeError,
		},
		{
			name:    "thinking signature has wrong type",
			content: `[{"type":"thinking","thinking":[],"signature":{}}]`,
			wantErr: jsonObjectTypeError,
		},
		{
			name:    "thinking closed has wrong type",
			content: `[{"type":"thinking","thinking":[],"closed":"false"}]`,
			wantErr: "cannot unmarshal string",
		},
		{
			name:    "thinking union is closed",
			content: `[{"type":"thinking","thinking":[{"type":"future_thinking"}]}]`,
			wantErr: `unsupported thinking chunk type "future_thinking"`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			serverURL := mistralCompletionServer(t, test.content)
			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL(serverURL),
			)
			require.NoError(t, err)

			_, err = provider.Completion(t.Context(), providers.CompletionParams{
				Model:    mistralReasoningModel,
				Messages: testutil.SimpleMessages(),
			})
			require.ErrorContains(t, err, test.wantErr)
		})
	}
}
