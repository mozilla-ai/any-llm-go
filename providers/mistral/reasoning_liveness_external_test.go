package mistral_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/mistral"
)

// The second choice remains active after the first has finished. The server
// intentionally leaves the transport open until both events reach the caller.
func TestStreamDeliversOrdinaryChoicesBeforeTransportCloses(t *testing.T) {
	for _, finish := range []string{"", "stop"} {
		t.Run("first_finish_"+finish, func(t *testing.T) {
			release := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
					return
				}
				t.Logf("request: %s", body)
				w.Header().Set("Content-Type", "text/event-stream")
				for _, event := range []string{
					fmt.Sprintf(`{"id":"audit-73","object":"chat.completion.chunk","created":1710000000,"model":"mistral-small-latest","choices":[{"index":0,"delta":{"role":"assistant","content":"first"},"finish_reason":%q}]}`, finish),
					`{"id":"audit-73","object":"chat.completion.chunk","created":1710000000,"model":"mistral-small-latest","choices":[{"index":1,"delta":{"role":"assistant","content":"still active"},"finish_reason":null}]}`,
				} {
					if _, err := fmt.Fprintf(w, "data: %s\n\n", event); err != nil {
						t.Error(err)
						return
					}
				}
				if err := http.NewResponseController(w).Flush(); err != nil {
					t.Error(err)
					return
				}
				select {
				case <-release:
				case <-r.Context().Done():
					return
				}
				_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
			}))
			defer server.Close()
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			p, err := mistral.New(config.WithAPIKey("audit-dummy"), config.WithBaseURL(server.URL))
			if err != nil {
				t.Fatal(err)
			}
			params := providers.CompletionParams{
				Model:    "mistral-small-latest",
				Messages: []providers.Message{{Role: providers.RoleUser, Content: "Return two distinct alternatives."}},
			}
			before, err := json.Marshal(params)
			if err != nil {
				t.Fatal(err)
			}
			chunks, errs := p.CompletionStream(ctx, params)
			for i, text := range []string{"first", "still active"} {
				select {
				case c, ok := <-chunks:
					if !ok || len(c.Choices) != 1 || c.Choices[0].Index != i || c.Choices[0].Delta.Content != text {
						t.Fatalf("wrong chunk %d: %+v", i, c)
					}
				case <-time.After(750 * time.Millisecond):
					t.Fatalf("chunk %d withheld while transport remains open; first finish=%q", i, finish)
				}
			}
			close(release)
			for range chunks {
			}
			for err := range errs {
				if err != nil {
					t.Error(err)
				}
			}
			after, err := json.Marshal(params)
			if err != nil || !reflect.DeepEqual(before, after) {
				t.Fatalf("caller mutated: %s -> %s (%v)", before, after, err)
			}
		})
	}
}
