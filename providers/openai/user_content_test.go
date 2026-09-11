package openai_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

type countedText struct{ calls *atomic.Int32 }

func (v countedText) MarshalJSON() ([]byte, error) {
	v.calls.Add(1)
	return []byte(`"prefix-17"`), nil
}

func TestUserPartsSingleNormalization(t *testing.T) {
	t.Parallel()

	for _, native := range []bool{false, true} {
		t.Run(map[bool]string{false: "compatible", true: "native"}[native], func(t *testing.T) {
			t.Parallel()

			wire := make(chan string, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
					return
				}
				wire <- string(body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(
					w,
					`{"id":"reply-3","object":"chat.completion","model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"result-59"},"finish_reason":"stop"}]}`,
				)
			}))
			defer server.Close()
			var p providers.Provider
			var err error
			if native {
				p, err = openai.New(config.WithAPIKey("fake-audit"), config.WithBaseURL(server.URL))
			} else {
				p, err = openai.NewCompatible(
					openai.CompatibleConfig{Name: "audit-compatible"},
					config.WithAPIKey("fake-audit"),
					config.WithBaseURL(server.URL),
				)
			}
			require.NoError(t, err)
			var calls atomic.Int32
			params := providers.CompletionParams{
				Model: "test-model",
				Messages: []providers.Message{{Role: providers.RoleUser, Content: []any{
					map[string]any{"type": "text", "text": countedText{&calls}},
					map[string]any{
						"type":      "image_url",
						"image_url": map[string]any{"url": "https://example.invalid/image-29.png", "detail": "high"},
					},
					map[string]any{"type": "text", "text": "suffix-43"},
				}}},
			}
			before, err := json.Marshal(params)
			require.NoError(t, err)
			calls.Store(0)
			result, err := p.Completion(t.Context(), params)
			require.NoError(t, err)
			require.Equal(t, "result-59", result.Choices[0].Message.Content)
			count := calls.Load()
			t.Logf("normalization calls=%d", count)
			if count != 1 {
				t.Errorf("normalized input %d times, want once", count)
			}
			after, err := json.Marshal(params)
			require.NoError(t, err)
			require.Equal(t, string(before), string(after), "caller message changed")
			detail := ""
			if native {
				detail = `,"detail":"high"`
			}
			body := <-wire
			t.Logf("complete request=%s", body)
			require.JSONEq(
				t,
				`{"model":"test-model","messages":[{"role":"user","content":[{"type":"text","text":"prefix-17"},{"type":"image_url","image_url":{"url":"https://example.invalid/image-29.png"`+detail+`}},{"type":"text","text":"suffix-43"}]}]}`,
				body,
			)
		})
	}
}
