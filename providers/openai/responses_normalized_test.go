package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/providers"
)

const normalizedResponseFixture = `{
  "id":"resp_123",
  "object":"response",
  "created_at":1,
  "model":"gpt-5.6-sol",
  "status":"completed",
  "output":[
    {
      "type":"message",
      "id":"msg_1",
      "status":"completed",
      "role":"assistant",
      "content":[
        {"type":"output_text","text":"hello","annotations":[],"logprobs":[]},
        {"type":"refusal","refusal":"cannot"}
      ]
    },
    {
      "type":"function_call",
      "id":"call_1",
      "status":"completed",
      "call_id":"tool_1",
      "name":"lookup",
      "arguments":"{\"id\":1}"
    },
    {
      "type":"reasoning",
      "id":"reason_1",
      "status":"completed",
      "summary":[{"type":"summary_text","text":"checked"}]
    },
    {"type":"future_item","id":"future_1","vendor_field":true}
  ],
  "usage":{
    "input_tokens":10,
    "input_tokens_details":{"cached_tokens":3},
    "output_tokens":5,
    "output_tokens_details":{"reasoning_tokens":2},
    "total_tokens":15
  },
  "future_envelope":"preserved"
}`

func TestResponsesPreservesPortableWireAndStructuredOutput(t *testing.T) {
	t.Parallel()

	var requestBody json.RawMessage
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/v1/responses", r.URL.Path)
		require.NoError(t, json.NewDecoder(r.Body).Decode(&requestBody))
		w.Header().Set("Content-Type", "application/json")
		_, err := io.WriteString(w, normalizedResponseFixture)
		require.NoError(t, err)
	}))
	t.Cleanup(server.Close)

	provider, err := NewCompatible(CompatibleConfig{
		Capabilities:   providers.Capabilities{Responses: true},
		DefaultAPIKey:  "test-key",
		DefaultBaseURL: server.URL + "/v1",
		Name:           "test-provider",
	})
	require.NoError(t, err)

	result, err := provider.Responses(t.Context(), providers.ResponsesParams{
		Input: []providers.ResponsesInputItem{
			{Role: providers.ResponsesInputRoleDeveloper, Content: "developer"},
			{Role: providers.ResponsesInputRoleSystem, Content: "system"},
			{Role: providers.ResponsesInputRoleUser, Content: "user"},
			{Role: providers.ResponsesInputRoleAssistant, Content: "assistant"},
		},
		Instructions:    new(""),
		MaxOutputTokens: new(123),
		Model:           "gpt-5.6-sol",
		ReasoningEffort: providers.ReasoningEffortNone,
	})
	require.NoError(t, err)

	require.JSONEq(t, `{
      "input":[
        {"role":"developer","content":"developer"},
        {"role":"system","content":"system"},
        {"role":"user","content":"user"},
        {"role":"assistant","content":"assistant"}
      ],
      "instructions":"",
      "max_output_tokens":123,
      "model":"gpt-5.6-sol",
      "reasoning":{"effort":"none"}
    }`, string(requestBody))
	require.Equal(t, "resp_123", result.ID)
	require.Equal(t, "gpt-5.6-sol", result.Model)
	require.Equal(t, "completed", result.Status)
	require.Equal(t, "hello", result.OutputText)
	require.Equal(t, &providers.ResponsesUsage{
		InputTokens:     10,
		OutputTokens:    5,
		TotalTokens:     15,
		CachedTokens:    3,
		ReasoningTokens: 2,
	}, result.Usage)
	require.Len(t, result.OutputItems, 4)
	require.Len(t, result.OutputItems[0].Content, 2)
	require.Equal(t, "output_text", result.OutputItems[0].Content[0].Type)
	require.Equal(t, "hello", result.OutputItems[0].Content[0].Text)
	require.JSONEq(t,
		`{"type":"output_text","text":"hello","annotations":[],"logprobs":[]}`,
		string(result.OutputItems[0].Content[0].ProviderRaw),
	)
	require.Equal(t, "refusal", result.OutputItems[0].Content[1].Type)
	require.Equal(t, "cannot", result.OutputItems[0].Content[1].Refusal)
	require.Equal(t, "lookup", result.OutputItems[1].Name)
	require.Equal(t, "tool_1", result.OutputItems[1].CallID)
	require.JSONEq(t, `{"id":1}`, result.OutputItems[1].Arguments)
	require.Equal(t, []string{"checked"}, result.OutputItems[2].Summary)
	require.JSONEq(t,
		`{"type":"future_item","id":"future_1","vendor_field":true}`,
		string(result.OutputItems[3].ProviderRaw),
	)
	require.Contains(t, string(result.ProviderRaw), `"future_envelope":"preserved"`)
}
