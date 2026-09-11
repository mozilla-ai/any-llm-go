package openai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestConvertOpenAIMessageRejectsUnsupportedShapes(t *testing.T) {
	t.Parallel()

	image := &providers.ImageURL{URL: testImageURL}
	tests := []struct {
		name    string
		message providers.Message
		wantErr string
	}{
		{
			name: "custom tool call",
			message: providers.Message{
				Role: providers.RoleAssistant, ToolCalls: []providers.ToolCall{{Type: "custom"}},
			},
			wantErr: "unsupported tool call type",
		},
		{
			name: "assistant image",
			message: providers.Message{
				Role:    providers.RoleAssistant,
				Content: []providers.ContentPart{{Type: contentTypeImageURL, ImageURL: image}},
			},
			wantErr: "unsupported assistant content part type",
		},
		{
			name: "developer image",
			message: providers.Message{
				Role:    providers.RoleDeveloper,
				Content: []providers.ContentPart{{Type: contentTypeImageURL, ImageURL: image}},
			},
			wantErr: "unsupported developer content part type",
		},
		{
			name: "system image",
			message: providers.Message{
				Role:    providers.RoleSystem,
				Content: []providers.ContentPart{{Type: contentTypeImageURL, ImageURL: image}},
			},
			wantErr: "unsupported system content part type",
		},
		{
			name: "tool image",
			message: providers.Message{
				Role:    providers.RoleTool,
				Content: []providers.ContentPart{{Type: contentTypeImageURL, ImageURL: image}},
			},
			wantErr: "unsupported tool content part type",
		},
		{
			name: "text with image payload",
			message: providers.Message{
				Role:    providers.RoleUser,
				Content: []providers.ContentPart{{Type: contentTypeText, Text: "text", ImageURL: image}},
			},
			wantErr: "text content requires only text",
		},
		{
			name: "image without URL",
			message: providers.Message{
				Role:    providers.RoleUser,
				Content: []providers.ContentPart{{Type: contentTypeImageURL}},
			},
			wantErr: "image_url content requires only a non-empty URL",
		},
		{
			name: "unknown user part",
			message: providers.Message{
				Role: providers.RoleUser, Content: []providers.ContentPart{{Type: "unknown"}},
			},
			wantErr: "unsupported user content part type",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, err := convertOpenAIMessage(tc.message)
			require.ErrorContains(t, err, tc.wantErr)
		})
	}
}

func TestCompatibleProviderKeepsOpenAIMessageFieldsProviderSpecific(t *testing.T) {
	t.Parallel()

	requests := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, err.Error(), http.StatusInternalServerError)

			return
		}

		requests <- body

		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{
			"id":"chatcmpl_123","object":"chat.completion","created":1,
			"model":"gpt-test","choices":[]
		}`))
	}))
	t.Cleanup(server.Close)

	params := providers.CompletionParams{
		Model: "gpt-test",
		Messages: []providers.Message{{
			Role: providers.RoleDeveloper, Content: "Follow the schema.", Name: "policy",
		}},
	}
	openAIProvider, err := NewCompatible(CompatibleConfig{
		DefaultBaseURL: server.URL, Name: "openai-test", OpenAIMessageSchema: true,
	}, config.WithAPIKey("test-key"))
	require.NoError(t, err)
	_, err = openAIProvider.Completion(t.Context(), params)
	require.NoError(t, err)
	require.JSONEq(t, `{
		"model":"gpt-test",
		"messages":[{
			"role":"developer","content":"Follow the schema.","name":"policy"
		}]
	}`, string(<-requests))

	compatibleProvider, err := NewCompatible(CompatibleConfig{
		DefaultBaseURL: server.URL, Name: "compatible-test",
	}, config.WithAPIKey("test-key"))
	require.NoError(t, err)
	_, err = compatibleProvider.Completion(t.Context(), params)
	require.ErrorContains(t, err, `unknown message role: "developer"`)

	message, err := convertCompatibleMessage(providers.Message{
		Role: providers.RoleUser,
		Name: "not-verified-for-compatible-providers",
		Content: []providers.ContentPart{{
			Type:     contentTypeImageURL,
			ImageURL: &providers.ImageURL{URL: testImageURL, Detail: "high"},
		}},
	})
	require.NoError(t, err)
	requireMessageJSON(t, message, `{
		"role":"user",
		"content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]
	}`)
}
