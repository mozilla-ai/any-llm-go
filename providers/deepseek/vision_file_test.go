package deepseek

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	llmerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const visionFileCompletionResponse = `{
	"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"deepseek-v4-flash-vision-exp",
	"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"},"logprobs":null}]
}`

func TestCompletionPreservesDeepSeekVisionFileWire(t *testing.T) {
	t.Parallel()

	serverURL, requestBody := deepSeekCompletionServer(t, visionFileCompletionResponse)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(serverURL))
	require.NoError(t, err)

	_, err = provider.Completion(t.Context(), providers.CompletionParams{
		Model: "deepseek-v4-flash-vision-exp",
		Messages: []providers.Message{{
			Role: providers.RoleUser,
			Content: []providers.ContentPart{
				{Type: "text", Text: "Compare these images."},
				{Type: "file", File: &providers.FileContent{FileID: "file-api-example"}},
				{
					Type: "file",
					File: &providers.FileContent{
						FileData: "data:image/png;base64,iVBORw0KGgo=",
						Filename: "chart.png",
					},
				},
				{Type: "file", File: &providers.FileContent{FileData: "data:image/gif;base64,R0lGODlh"}},
				{
					Type: "image_url",
					ImageURL: &providers.ImageURL{
						URL:    "https://example.com/chart.png",
						Detail: "original",
					},
				},
			},
		}},
	})
	require.NoError(t, err)

	body := <-requestBody
	messages, ok := body["messages"].([]any)
	require.True(t, ok)
	user, ok := messages[0].(map[string]any)
	require.True(t, ok)
	require.Equal(t, []any{
		map[string]any{"type": "text", "text": "Compare these images."},
		map[string]any{"type": "file", "file_id": "file-api-example"},
		map[string]any{
			"type":      "file",
			"file_data": "data:image/png;base64,iVBORw0KGgo=",
			"filename":  "chart.png",
		},
		map[string]any{"type": "file", "file_data": "data:image/gif;base64,R0lGODlh"},
		map[string]any{
			"type": "image_url",
			"image_url": map[string]any{
				"url":    "https://example.com/chart.png",
				"detail": "original",
			},
		},
	}, user["content"])
}

func TestCompletionRejectsInvalidDeepSeekVisionFiles(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		part providers.ContentPart
	}{
		{name: "missing file", part: providers.ContentPart{Type: "file"}},
		{name: "empty file", part: providers.ContentPart{Type: "file", File: &providers.FileContent{}}},
		{
			name: "both file sources",
			part: providers.ContentPart{
				Type: "file",
				File: &providers.FileContent{FileID: "file-api-example", FileData: "data:image/png;base64,data"},
			},
		},
		{
			name: "filename with file id",
			part: providers.ContentPart{
				Type: "file",
				File: &providers.FileContent{FileID: "file-api-example", Filename: "chart.png"},
			},
		},
		{
			name: "file with text",
			part: providers.ContentPart{
				Type: "file", Text: "unexpected", File: &providers.FileContent{FileID: "file-api-example"},
			},
		},
		{
			name: "file with image URL",
			part: providers.ContentPart{
				Type: "file", ImageURL: &providers.ImageURL{URL: "https://example.com/chart.png"},
				File: &providers.FileContent{FileID: "file-api-example"},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			serverURL, requests := deepSeekCompletionServer(t, visionFileCompletionResponse)
			provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(serverURL))
			require.NoError(t, err)

			_, err = provider.Completion(t.Context(), providers.CompletionParams{
				Model: "deepseek-v4-flash-vision-exp",
				Messages: []providers.Message{{
					Role:    providers.RoleUser,
					Content: []providers.ContentPart{tc.part},
				}},
			})
			require.ErrorIs(t, err, llmerrors.ErrInvalidRequest)
			select {
			case <-requests:
				t.Fatal("invalid file input reached transport")
			default:
			}
		})
	}
}
