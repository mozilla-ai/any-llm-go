package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestConvertUserMessagePreservesFileContent(t *testing.T) {
	t.Parallel()

	message := convertUserMessage(providers.Message{
		Role: providers.RoleUser,
		Content: []providers.ContentPart{
			{Type: "file", File: &providers.FileContent{FileID: "file-example"}},
			{
				Type: "file",
				File: &providers.FileContent{
					FileData: "data:image/png;base64,iVBORw0KGgo=",
					Filename: "chart.png",
				},
			},
		},
	})
	wire, err := json.Marshal(message)
	require.NoError(t, err)
	require.JSONEq(t, `{
		"role":"user",
		"content":[
			{"type":"file","file":{"file_id":"file-example"}},
			{"type":"file","file":{"file_data":"data:image/png;base64,iVBORw0KGgo=","filename":"chart.png"}}
		]
	}`, string(wire))
}
