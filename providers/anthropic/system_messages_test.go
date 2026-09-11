package anthropic_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/anthropic"
)

func TestCompletionNormalizesSystemMessages(t *testing.T) {
	t.Parallel()

	fixtures := []struct {
		name           string
		model          string
		messages       []providers.Message
		expectedSystem string
		expectedRoles  []string
	}{
		{
			name:  "unsupported model with interleaved system message",
			model: "claude-sonnet-5",
			messages: []providers.Message{
				{Role: providers.RoleUser, Content: "first user"},
				{Role: providers.RoleSystem, Content: "later instruction"},
			},
			expectedSystem: "later instruction",
			expectedRoles:  []string{"user"},
		},
		{
			name:  "current model with leading multiple and interleaved system messages",
			model: "claude-opus-5",
			messages: []providers.Message{
				{Role: providers.RoleSystem, Content: "first instruction"},
				{Role: providers.RoleSystem, Content: "second instruction"},
				{Role: providers.RoleUser, Content: "first user"},
				{Role: providers.RoleSystem, Content: "later instruction"},
				{Role: providers.RoleUser, Content: "second user"},
				{Role: providers.RoleAssistant, Content: "assistant"},
			},
			expectedSystem: "first instruction\nsecond instruction\nlater instruction",
			expectedRoles:  []string{"user", "user", "assistant"},
		},
	}

	for _, fixture := range fixtures {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", fixture.name, streaming), func(t *testing.T) {
				t.Parallel()

				captured := make(chan []byte, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					body, err := io.ReadAll(r.Body)
					if err != nil {
						t.Error(err)
						return
					}
					captured <- body

					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						_, err = io.WriteString(
							w,
							"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"test\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
						)
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, err = io.WriteString(
							w,
							`{"id":"m","type":"message","role":"assistant","model":"test","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
						)
					}
					if err != nil {
						t.Error(err)
					}
				}))
				defer server.Close()

				provider, err := anthropic.New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
				require.NoError(t, err)

				params := providers.CompletionParams{Model: fixture.model, Messages: fixture.messages}
				before, err := json.Marshal(params)
				require.NoError(t, err)

				if streaming {
					chunks, errs := provider.CompletionStream(t.Context(), params)
					for range chunks {
					}
					for streamErr := range errs {
						require.NoError(t, streamErr)
					}
				} else {
					_, err = provider.Completion(t.Context(), params)
					require.NoError(t, err)
				}

				after, err := json.Marshal(params)
				require.NoError(t, err)
				require.JSONEq(t, string(before), string(after), "caller input mutated")

				var wire struct {
					Messages []struct {
						Role string `json:"role"`
					} `json:"messages"`
					System []struct {
						Text string `json:"text"`
					} `json:"system"`
				}
				require.NoError(t, json.Unmarshal(<-captured, &wire))
				require.Len(t, wire.System, 1)
				require.Equal(t, fixture.expectedSystem, wire.System[0].Text)

				roles := make([]string, len(wire.Messages))
				for i, message := range wire.Messages {
					roles[i] = message.Role
				}
				require.Equal(t, fixture.expectedRoles, roles)
			})
		}
	}
}
