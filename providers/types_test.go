package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestRerankTypesJSON(t *testing.T) {
	t.Parallel()

	topN := 3
	params := RerankParams{
		Model:     "cohere:rerank-v3.5",
		Query:     "test query",
		Documents: []string{"doc1", "doc2"},
		TopN:      &topN,
	}

	data, err := json.Marshal(params)
	require.NoError(t, err)

	var decoded RerankParams
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, params.Model, decoded.Model)
	require.Equal(t, params.Query, decoded.Query)
	require.Equal(t, params.Documents, decoded.Documents)
	require.Equal(t, *params.TopN, *decoded.TopN)
}

func TestRerankResponseJSON(t *testing.T) {
	t.Parallel()

	totalTokens := 100
	resp := RerankResponse{
		ID: "rerank-123",
		Results: []RerankResult{
			{Index: 0, RelevanceScore: 0.95},
			{Index: 2, RelevanceScore: 0.80},
		},
		Meta: &RerankMeta{
			BilledUnits: map[string]float64{"search_units": 1.0},
			Tokens:      map[string]int{"input_tokens": 100},
		},
		Usage: &RerankUsage{TotalTokens: &totalTokens},
	}

	data, err := json.Marshal(resp)
	require.NoError(t, err)

	var decoded RerankResponse
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, resp.ID, decoded.ID)
	require.Len(t, decoded.Results, 2)
	require.Equal(t, 0.95, decoded.Results[0].RelevanceScore)
	require.NotNil(t, decoded.Usage)
	require.Equal(t, 100, *decoded.Usage.TotalTokens)
}

func TestToolCallExtraExcludedFromJSON(t *testing.T) {
	t.Parallel()

	tc := ToolCall{
		ID:   "call_123",
		Type: "function",
		Function: FunctionCall{
			Name:      "get_weather",
			Arguments: `{"location": "Paris"}`,
		},
		Extra: map[string]ProviderData{
			"google": {"thought_signature": "abc123"},
		},
	}

	b, err := json.Marshal(tc)
	require.NoError(t, err)

	var decoded map[string]any
	err = json.Unmarshal(b, &decoded)
	require.NoError(t, err)

	// Extra must not appear in JSON output.
	_, hasExtra := decoded["extra"]
	require.False(t, hasExtra, "Extra field must be excluded from JSON serialization")

	// Standard fields must be present.
	require.Equal(t, "call_123", decoded["id"])
	require.Equal(t, "function", decoded["type"])
}

func TestReasoningProviderRawRoundTrips(t *testing.T) {
	t.Parallel()

	reasoning := Reasoning{
		Content: "step one",
		ProviderRaw: json.RawMessage(
			`[{"type":"thinking","thinking":[{"type":"text","text":"step one"}],"signature":"sig-abc"}]`,
		),
	}

	data, err := json.Marshal(reasoning)
	require.NoError(t, err)

	var decoded Reasoning
	require.NoError(t, json.Unmarshal(data, &decoded))
	require.Equal(t, reasoning, decoded)

	var empty Reasoning

	emptyData, err := json.Marshal(empty)
	require.NoError(t, err)
	require.JSONEq(t, `{}`, string(emptyData))
}
