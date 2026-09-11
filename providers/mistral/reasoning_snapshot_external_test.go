package mistral_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/mistral"
)

func TestStreamSnapshotRetainsSeparateTextMetadata(t *testing.T) {
	t.Parallel()
	for _, prefix := range []string{"none", "array", "string"} {
		t.Run(prefix, func(t *testing.T) {
			t.Parallel()
			thinking := `{"type":"thinking","thinking":[{"type":"text","text":"reason-7"}],"signature":"sig-13","closed":true}`
			text := `{"type":"text","text":"answer-29","reference_ids":["ref-41"],"future":{"n":9007199254740993}}`
			blocks := []string{thinking, text}
			wantAnswer := "answer-29"
			if prefix != "none" {
				blocks = append([]string{`{"type":"text","text":"prefix-17 "}`}, blocks...)
				wantAnswer = "prefix-17 answer-29"
			}
			wire := make(chan []byte, 2)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				wire <- body
				var request struct {
					Stream bool `json:"stream"`
				}
				_ = json.Unmarshal(body, &request)
				if !request.Stream {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(
						w,
						`{"id":"next","object":"chat.completion","created":1,"model":"local","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
					)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				for i, block := range blocks {
					content := "[" + block + "]"
					if i == 0 && prefix == "string" {
						content = `"prefix-17 "`
					}
					_, _ = fmt.Fprintf(
						w,
						"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"local\",\"choices\":[{\"index\":4,\"delta\":{\"content\":%s}}]}\n\n",
						content,
					)
				}
				_, _ = io.WriteString(
					w,
					"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"local\",\"choices\":[{\"index\":4,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":23,\"total_tokens\":34}}\n\ndata: [DONE]\n\n",
				)
			}))
			defer server.Close()
			p, err := mistral.New(config.WithAPIKey("synthetic-audit-token"), config.WithBaseURL(server.URL))
			if err != nil {
				t.Fatal(err)
			}
			chunks, errs := p.CompletionStream(
				t.Context(),
				providers.CompletionParams{
					Model:    "local",
					Messages: []providers.Message{{Role: providers.RoleUser, Content: "input"}},
				},
			)
			var answer string
			var snapshot *providers.Reasoning
			var usage *providers.Usage
			for chunk := range chunks {
				for _, choice := range chunk.Choices {
					if choice.Index != 4 {
						t.Errorf("wrong choice ownership: %d", choice.Index)
					}
					answer += choice.Delta.Content
					if choice.Delta.Content != "" && choice.FinishReason == "" && choice.Delta.Reasoning != nil {
						t.Error("text-only delta gained reasoning metadata")
					}
					if choice.FinishReason != "" {
						snapshot = choice.Delta.Reasoning
					}
				}
				if chunk.Usage != nil {
					usage = chunk.Usage
				}
			}
			if streamErr := <-errs; streamErr != nil {
				t.Fatal(streamErr)
			}
			<-wire
			if answer != wantAnswer {
				t.Errorf("answer=%q, want %q", answer, wantAnswer)
			}
			if snapshot == nil {
				t.Fatal("missing terminal snapshot")
			}
			want := "["
			for i, block := range blocks {
				if i > 0 {
					want += ","
				}
				want += block
			}
			want += "]"
			decode := func(raw []byte) any {
				var v any
				d := json.NewDecoder(bytes.NewReader(raw))
				d.UseNumber()
				if decodeErr := d.Decode(&v); decodeErr != nil {
					t.Fatal(decodeErr)
				}
				return v
			}
			t.Logf("answer=%q snapshot=%s usage=%+v", answer, snapshot.ProviderRaw, usage)
			if !reflect.DeepEqual(decode([]byte(want)), decode(snapshot.ProviderRaw)) {
				t.Errorf("terminal replay snapshot lost source blocks or text metadata; want=%s", want)
			}
			if usage == nil || usage.PromptTokens != 11 || usage.CompletionTokens != 23 || usage.TotalTokens != 34 {
				t.Errorf("wrong asymmetric usage: %+v", usage)
			}
			_, err = p.Completion(
				t.Context(),
				providers.CompletionParams{
					Model: "local",
					Messages: []providers.Message{
						{Role: providers.RoleAssistant, Content: answer, Reasoning: snapshot},
					},
				},
			)
			if err != nil {
				t.Errorf("unaltered streamed output cannot replay: %v", err)
				return
			}
			var replay struct {
				Messages []struct {
					Content json.RawMessage `json:"content"`
				} `json:"messages"`
			}
			if err := json.Unmarshal(<-wire, &replay); err != nil {
				t.Fatal(err)
			}
			if len(replay.Messages) != 1 ||
				!reflect.DeepEqual(decode(replay.Messages[0].Content), decode([]byte(want))) {
				t.Errorf("replay request omitted wire metadata")
			}
		})
	}
}

func TestStreamOrdinaryTextArrayHasNoReasoning(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(
			w,
			"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"local\",\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"text\",\"text\":\"ordinary-43\"}]}}]}\n\ndata: {\"id\":\"c\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
		)
	}))
	defer server.Close()
	p, err := mistral.New(config.WithAPIKey("synthetic-audit-token"), config.WithBaseURL(server.URL))
	if err != nil {
		t.Fatal(err)
	}
	chunks, errs := p.CompletionStream(
		t.Context(),
		providers.CompletionParams{
			Model:    "local",
			Messages: []providers.Message{{Role: providers.RoleUser, Content: "input"}},
		},
	)
	answer := ""
	for chunk := range chunks {
		for _, choice := range chunk.Choices {
			answer += choice.Delta.Content
			if choice.Delta.Reasoning != nil {
				t.Errorf("ordinary text gained reasoning metadata: %+v", choice.Delta.Reasoning)
			}
		}
	}
	if err := <-errs; err != nil || answer != "ordinary-43" {
		t.Fatalf("ordinary text control: answer=%q err=%v", answer, err)
	}
}
