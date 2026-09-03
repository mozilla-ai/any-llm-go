package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestConvertParamsPreservesToolInputJSON(t *testing.T) {
	t.Parallel()

	req, err := (&Provider{}).convertParams(providers.CompletionParams{
		Model: "claude-opus-5",
		Messages: []providers.Message{{
			Role: providers.RoleAssistant,
			ToolCalls: []providers.ToolCall{{
				ID: "toolu_1",
				Function: providers.FunctionCall{
					Name:      "weather",
					Arguments: `{"city":"Paris","population":9007199254740993}`,
				},
			}},
		}},
	})
	require.NoError(t, err)

	body, err := json.Marshal(req)
	require.NoError(t, err)

	var wire struct {
		Messages []struct {
			Content []struct {
				Input json.RawMessage `json:"input"`
				Type  string          `json:"type"`
			} `json:"content"`
		} `json:"messages"`
	}
	require.NoError(t, json.Unmarshal(body, &wire))
	require.Len(t, wire.Messages, 1)
	require.Len(t, wire.Messages[0].Content, 1)
	require.Equal(t, "tool_use", wire.Messages[0].Content[0].Type)
	require.JSONEq(
		t,
		`{"city":"Paris","population":9007199254740993}`,
		string(wire.Messages[0].Content[0].Input),
	)
	require.Contains(t, string(wire.Messages[0].Content[0].Input), "9007199254740993")
}

func TestConvertParamsRejectsInvalidToolInput(t *testing.T) {
	t.Parallel()

	for _, input := range []string{"", `{"city":`, `[]`, `null`} {
		_, err := (&Provider{}).convertParams(providers.CompletionParams{
			Model: "claude-opus-5",
			Messages: []providers.Message{{
				Role: providers.RoleAssistant,
				ToolCalls: []providers.ToolCall{{
					ID: "toolu_1",
					Function: providers.FunctionCall{
						Name:      "weather",
						Arguments: input,
					},
				}},
			}},
		})
		require.ErrorIs(t, err, errors.ErrInvalidRequest, input)
	}
}
