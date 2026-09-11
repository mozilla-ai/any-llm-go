package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestThinkingRequestContracts(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		model      string
		effort     providers.ReasoningEffort
		maxTokens  int
		stream     bool
		thinking   map[string]any
		effortWire string
		wantMax    float64
	}{
		{
			"legacy nonstream",
			"claude-sonnet-4-5",
			providers.ReasoningEffortLow,
			6000,
			false,
			map[string]any{"type": "enabled", "budget_tokens": float64(1024)},
			"",
			6000,
		},
		{
			"legacy stream at budget boundary",
			"claude-haiku-4-5",
			providers.ReasoningEffortHigh,
			32768,
			true,
			map[string]any{"type": "enabled", "budget_tokens": float64(16384)},
			"",
			32768,
		},
		{
			"current default-off nonstream",
			"claude-sonnet-4-6",
			providers.ReasoningEffortMedium,
			6000,
			false,
			map[string]any{"type": "adaptive"},
			"medium",
			6000,
		},
		{
			"current default-off stream",
			"claude-opus-4-6-20260901",
			providers.ReasoningEffortHigh,
			6000,
			true,
			map[string]any{"type": "adaptive"},
			"high",
			6000,
		},
		{
			"none explicitly disables",
			"claude-sonnet-4-6",
			providers.ReasoningEffortNone,
			6000,
			false,
			map[string]any{"type": "disabled"},
			"",
			6000,
		},
		{"auto leaves defaults", "claude-sonnet-4-6", providers.ReasoningEffortAuto, 6000, true, nil, "", 6000},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			requests := make(chan map[string]any, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body map[string]any
				require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
				requests <- body
				w.Header().Set("Content-Type", "application/json")
				if tc.stream {
					w.Header().Set("Content-Type", "text/event-stream")
					_, _ = fmt.Fprint(
						w,
						anthropicMessageStartSSE+"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
					)
					return
				}
				_, _ = fmt.Fprint(
					w,
					`{"id":"msg","type":"message","role":"assistant","model":"`+tc.model+`","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
				)
			}))
			t.Cleanup(server.Close)

			provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
			require.NoError(t, err)
			params := providers.CompletionParams{
				Model: tc.model, MaxTokens: new(tc.maxTokens), ReasoningEffort: tc.effort,
				Messages: []providers.Message{{Role: providers.RoleUser, Content: "solve"}},
			}
			if tc.stream {
				chunks, errs := provider.CompletionStream(t.Context(), params)
				for range chunks {
				}
				require.NoError(t, <-errs)
			} else {
				_, err = provider.Completion(t.Context(), params)
				require.NoError(t, err)
			}

			body := <-requests
			require.Equal(t, tc.wantMax, body["max_tokens"])
			if tc.thinking == nil {
				require.NotContains(t, body, "thinking")
			} else {
				require.Equal(t, tc.thinking, body["thinking"])
			}
			output, _ := body["output_config"].(map[string]any)
			if tc.effortWire == "" {
				require.Empty(t, output["effort"])
			} else {
				require.Equal(t, tc.effortWire, output["effort"])
			}
		})
	}
}
