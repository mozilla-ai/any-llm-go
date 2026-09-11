package anthropic

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const reasoningContentJSON = `[
	{"type":"thinking","thinking":"first ","signature":"sig-a"},
	{"type":"text","text":"answer "},
	{"type":"redacted_thinking","data":"encrypted"},
	{"type":"thinking","thinking":"second","signature":"sig-b"},
	{"type":"tool_use","id":"tool-1","name":"lookup","input":{"key":"value"}},
	{"type":"text","text":"done"}
]`

func TestConvertResponsePreservesOrderedReasoningForToolResultReplay(t *testing.T) {
	t.Parallel()

	response := decodeAnthropicMessage(t, `{
		"id":"msg-1","type":"message","role":"assistant","model":"claude-opus-5",
		"content":`+reasoningContentJSON+`,"stop_reason":"tool_use",
		"usage":{"input_tokens":3,"output_tokens":8}
	}`)
	completion, err := convertResponse(&response)
	require.NoError(t, err)

	message := completion.Choices[0].Message

	require.Equal(t, "answer done", message.Content)
	require.Equal(t, "first second", message.Reasoning.Content)
	require.JSONEq(t, reasoningContentJSON, string(message.Reasoning.ProviderRaw))
	require.Equal(t, []providers.ToolCall{{
		ID: "tool-1", Type: "function",
		Function: providers.FunctionCall{Name: "lookup", Arguments: `{"key":"value"}`},
	}}, message.ToolCalls)

	messages, _, err := convertMessages([]providers.Message{
		message,
		{Role: providers.RoleTool, ToolCallID: "tool-1", Content: "result"},
	})
	require.NoError(t, err)
	body, err := json.Marshal(messages)
	require.NoError(t, err)
	require.JSONEq(t, `[
		{"role":"assistant","content":`+reasoningContentJSON+`},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"tool-1","is_error":false,
			"content":[{"type":"text","text":"result"}]}]}
	]`, string(body))
}

func TestConvertResponsePreservesSignatureOnlyAndRedactedThinking(t *testing.T) {
	t.Parallel()

	response := decodeAnthropicMessage(t, `{
		"id":"msg-1","type":"message","role":"assistant","model":"claude-opus-5",
		"content":[
			{"type":"thinking","thinking":"","signature":"signature-only"},
			{"type":"redacted_thinking","data":"encrypted"}
		],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}
	}`)
	completion, err := convertResponse(&response)
	require.NoError(t, err)

	message := completion.Choices[0].Message

	require.Empty(t, message.Reasoning.Content)
	require.JSONEq(t, `[
		{"type":"thinking","thinking":"","signature":"signature-only"},
		{"type":"redacted_thinking","data":"encrypted"}
	]`, string(message.Reasoning.ProviderRaw))
}

func TestAssistantReasoningReplayRejectsUnsafeRawBlocks(t *testing.T) {
	t.Parallel()

	tests := []string{
		`[{"type":"future_thinking","thinking":"thought","signature":"sig"}]`,
		`[{"thinking":"thought","signature":"sig"}]`,
		`[{"type":"thinking","thinking":"thought"}]`,
		`[{"type":"thinking","thinking":"thought","signature":7}]`,
		`[{"type":"thinking","thinking":{},"signature":"sig"}]`,
		`[{"type":"thinking","thinking":"","signature":null}]`,
		`[{"type":"thinking","thinking":null,"signature":"sig"}]`,
		`[{"type":"redacted_thinking","data":false}]`,
		`[{"type":"redacted_thinking","data":null}]`,
	}
	for _, raw := range tests {
		_, err := convertAssistantMessage(providers.Message{
			Role:      providers.RoleAssistant,
			Reasoning: &providers.Reasoning{ProviderRaw: json.RawMessage(raw), Provider: providerName},
		})
		require.ErrorIs(t, err, errors.ErrInvalidRequest, raw)
	}
}

func TestReasoningReplayPreservesFutureFields(t *testing.T) {
	t.Parallel()

	raw := json.RawMessage(`[
		{"type":"thinking","thinking":"thought","signature":"opaque-sig","future":{"version":2}},
		{"type":"redacted_thinking","data":"opaque-data","future_flag":true}
	]`)
	message, err := convertAssistantMessage(providers.Message{
		Role:      providers.RoleAssistant,
		Reasoning: &providers.Reasoning{Content: "thought", ProviderRaw: raw, Provider: providerName},
	})
	require.NoError(t, err)
	wire, err := json.Marshal(message)
	require.NoError(t, err)
	require.JSONEq(t, `{"role":"assistant","content":`+string(raw)+`}`, string(wire))

	response := decodeAnthropicMessage(t, `{
		"id":"msg-1","type":"message","role":"assistant","model":"claude-opus-5",
		"content":`+string(raw)+`,"stop_reason":"end_turn",
		"usage":{"input_tokens":1,"output_tokens":1}
	}`)
	completion, err := convertResponse(&response)
	require.NoError(t, err)
	require.JSONEq(t, string(raw), string(completion.Choices[0].Message.Reasoning.ProviderRaw))
}

func TestAssistantReasoningReplayIgnoresForeignProviderRaw(t *testing.T) {
	t.Parallel()

	message, err := convertAssistantMessage(providers.Message{
		Role:    providers.RoleAssistant,
		Content: "answer",
		Reasoning: &providers.Reasoning{
			Content:     "thought",
			ProviderRaw: json.RawMessage(`{"not":"anthropic content"}`),
			Provider:    "mistral",
		},
	})
	require.NoError(t, err)

	wire, err := json.Marshal(message)
	require.NoError(t, err)
	require.JSONEq(t, `{"role":"assistant","content":[{"type":"text","text":"answer"}]}`, string(wire))
}

func TestAssistantReasoningReplayRejectsMalformedOrStaleRaw(t *testing.T) {
	t.Parallel()

	valid := providers.Message{
		Role:    providers.RoleAssistant,
		Content: "answer done",
		Reasoning: &providers.Reasoning{
			Content:     "first second",
			ProviderRaw: json.RawMessage(reasoningContentJSON),
			Provider:    providerName,
		},
		ToolCalls: []providers.ToolCall{{
			ID: "tool-1", Type: "function",
			Function: providers.FunctionCall{Name: "lookup", Arguments: `{"key":"value"}`},
		}},
	}

	tests := []struct {
		name   string
		mutate func(*providers.Message)
	}{
		{name: "malformed raw", mutate: func(message *providers.Message) {
			message.Reasoning.ProviderRaw = json.RawMessage(`{`)
		}},
		{name: "edited text", mutate: func(message *providers.Message) { message.Content = "edited" }},
		{name: "edited reasoning", mutate: func(message *providers.Message) { message.Reasoning.Content = "edited" }},
		{name: "edited tool call", mutate: func(message *providers.Message) {
			message.ToolCalls[0].Function.Name = "edited"
		}},
		{name: "edited tool id", mutate: func(message *providers.Message) {
			message.ToolCalls[0].ID = "edited"
		}},
		{name: "edited tool type", mutate: func(message *providers.Message) {
			message.ToolCalls[0].Type = "edited"
		}},
		{name: "added tool metadata", mutate: func(message *providers.Message) {
			message.ToolCalls[0].Extra = map[string]providers.ProviderData{"anthropic": {"edited": true}}
		}},
		{name: "fragment without thinking", mutate: func(message *providers.Message) {
			message.Content = "answer done"
			message.Reasoning.Content = ""
			message.Reasoning.ProviderRaw = json.RawMessage(`[{"type":"text","text":"answer done"}]`)
			message.ToolCalls = nil
		}},
	}
	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			message := valid
			message.Reasoning = new(providers.Reasoning)
			*message.Reasoning = *valid.Reasoning
			message.ToolCalls = append([]providers.ToolCall(nil), valid.ToolCalls...)
			testCase.mutate(&message)

			_, err := convertAssistantMessage(message)
			require.ErrorIs(t, err, errors.ErrInvalidRequest)
		})
	}
}

func TestAssistantReasoningReplayComparesToolArgumentsAsJSON(t *testing.T) {
	t.Parallel()

	raw := json.RawMessage(`[
		{"type":"thinking","thinking":"thought","signature":"sig","future":true},
		{"type":"tool_use","id":"tool-1","name":"lookup","input":{"n":9007199254740993,"text":"a/b"},"future_flag":true}
	]`)
	valid := providers.Message{
		Role:      providers.RoleAssistant,
		Reasoning: &providers.Reasoning{Content: "thought", ProviderRaw: raw, Provider: providerName},
		ToolCalls: []providers.ToolCall{{
			ID: "tool-1", Type: "function",
			Function: providers.FunctionCall{
				Name: "lookup", Arguments: `{ "text": "a\/b", "n": 9007199254740993 }`,
			},
		}},
	}
	original, err := json.Marshal(valid)
	require.NoError(t, err)

	message, err := convertAssistantMessage(valid)
	require.NoError(t, err)
	wire, err := json.Marshal(message)
	require.NoError(t, err)
	require.JSONEq(t, `{"role":"assistant","content":`+string(raw)+`}`, string(wire))
	after, err := json.Marshal(valid)
	require.NoError(t, err)
	require.Equal(t, original, after)

	for _, arguments := range []string{
		`{"text":"a/b","n":9007199254740992}`,
		`{"text":"changed","n":9007199254740993}`,
		`{"text":"a/b","n":9007199254740993`,
		`{"text":"a/b","n":9007199254740993} {}`,
	} {
		changed := valid
		changed.ToolCalls = slices.Clone(valid.ToolCalls)
		changed.ToolCalls[0].Function.Arguments = arguments
		_, err := convertAssistantMessage(changed)
		require.ErrorIs(t, err, errors.ErrInvalidRequest)
	}
}

func TestCompletionStreamEmitsCompletedReasoningSnapshotAndReplaysIt(t *testing.T) {
	t.Parallel()

	events := []string{
		`{"type":"content_block_start","index":0,` +
			`"content_block":{"type":"thinking","thinking":"","signature":"","future":true}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thought"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,` +
			`"content_block":{"type":"redacted_thinking","data":"encrypted","future":true}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"answer"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}`,
		`{"type":"message_stop"}`,
	}
	provider := newStreamTestProvider(t, func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")

		if err := writeAnthropicSSE(writer, anthropicMessageStartSSE, events); err != nil {
			t.Errorf("write Anthropic reasoning stream: %v", err)
		}
	})

	chunks, errs := provider.CompletionStream(t.Context(), streamTestParams())

	var (
		reasoningText strings.Builder
		raw           json.RawMessage
		content       strings.Builder
	)

	for chunk := range chunks {
		delta := chunk.Choices[0].Delta
		content.WriteString(delta.Content)

		if delta.Reasoning != nil {
			reasoningText.WriteString(delta.Reasoning.Content)

			if len(delta.Reasoning.ProviderRaw) > 0 {
				raw = delta.Reasoning.ProviderRaw
			}
		}
	}

	require.NoError(t, <-errs)
	require.Equal(t, "thought", reasoningText.String())
	require.Equal(t, "answer", content.String())
	require.JSONEq(t, `[
		{"type":"thinking","thinking":"thought","signature":"sig","future":true},
		{"type":"redacted_thinking","data":"encrypted","future":true},
		{"type":"text","text":"answer"}
	]`, string(raw))

	replayed, err := convertAssistantMessage(providers.Message{
		Role: providers.RoleAssistant, Content: content.String(),
		Reasoning: &providers.Reasoning{Content: reasoningText.String(), ProviderRaw: raw, Provider: providerName},
	})
	require.NoError(t, err)
	replayedJSON, err := json.Marshal(replayed)
	require.NoError(t, err)
	require.JSONEq(t, `{"role":"assistant","content":[
		{"type":"thinking","thinking":"thought","signature":"sig","future":true},
		{"type":"redacted_thinking","data":"encrypted","future":true},
		{"type":"text","text":"answer"}
	]}`, string(replayedJSON))
}

func decodeAnthropicMessage(t *testing.T, raw string) anthropic.Message {
	t.Helper()

	var message anthropic.Message
	require.NoError(t, json.Unmarshal([]byte(raw), &message))

	return message
}

func writeAnthropicSSE(writer http.ResponseWriter, start string, events []string) error {
	if _, err := fmt.Fprint(writer, start); err != nil {
		return fmt.Errorf("writing message-start event: %w", err)
	}

	for _, event := range events {
		var envelope struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(event), &envelope); err != nil {
			return fmt.Errorf("decoding stream event: %w", err)
		}

		if _, err := fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", envelope.Type, event); err != nil {
			return fmt.Errorf("writing %s event: %w", envelope.Type, err)
		}
	}

	return nil
}

type failingSSEWriter struct {
	*httptest.ResponseRecorder

	calls  int
	failOn int
}

func (w *failingSSEWriter) Write(data []byte) (int, error) {
	w.calls++
	if w.calls == w.failOn {
		return 0, io.ErrClosedPipe
	}

	return w.ResponseRecorder.Write(data)
}

func TestWriteAnthropicSSEErrors(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name    string
		event   string
		failOn  int
		message string
	}{
		{name: "start write", failOn: 1, message: "writing message-start event"},
		{name: "event decode", event: `{`, message: "decoding stream event"},
		{name: "event write", event: `{"type":"message_stop"}`, failOn: 2, message: "writing message_stop event"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			writer := &failingSSEWriter{ResponseRecorder: httptest.NewRecorder(), failOn: test.failOn}
			err := writeAnthropicSSE(writer, "start", []string{test.event})
			require.ErrorContains(t, err, test.message)

			if test.failOn != 0 {
				require.ErrorIs(t, err, io.ErrClosedPipe)
			} else {
				var syntaxError *json.SyntaxError
				require.ErrorAs(t, err, &syntaxError)
			}
		})
	}
}
