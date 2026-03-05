package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

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
