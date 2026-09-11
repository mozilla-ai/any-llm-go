package mistral

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestReasoningAssemblyMixedEnvelopeLiveness(t *testing.T) {
	t.Parallel()

	upstream := make(chan providers.ChatCompletionChunk, 3)
	upstreamErrs := make(chan error)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	chunks, errs := assembleReasoningStream(ctx, cancel, upstream, upstreamErrs)

	terminal := reasoningDelta(
		4,
		`[{"type":"thinking","thinking":[{"type":"text","text":"thought"}],"signature":"sig","closed":true},{"type":"text","text":"answer"}]`,
		"answer",
	)
	terminal.FinishReason = "stop"
	mixed := providers.ChatCompletionChunk{
		ID: "mixed", Object: "chat.completion.chunk", Created: 123,
		Model: "model", SystemFingerprint: "fingerprint",
		Choices: []providers.ChunkChoice{terminal, reasoningDelta(1, "", "first")},
		Usage:   &providers.Usage{PromptTokens: 11, CompletionTokens: 3, TotalTokens: 14},
	}
	before, err := json.Marshal(mixed)
	require.NoError(t, err)
	upstream <- mixed
	upstream <- providers.ChatCompletionChunk{ID: "later", Choices: []providers.ChunkChoice{
		reasoningDelta(4, "", "after terminal"), reasoningDelta(1, "", "second"),
	}}
	upstream <- providers.ChatCompletionChunk{ID: "usage", Usage: &providers.Usage{TotalTokens: 19}}

	var received []providers.ChatCompletionChunk
	for range 3 {
		select {
		case chunk := <-chunks:
			received = append(received, chunk)
		case <-time.After(time.Second):
			t.Fatal("ordinary choice or usage withheld while upstream remains open")
		}
	}
	want := mixed
	want.Choices = []providers.ChunkChoice{mixed.Choices[1]}
	require.Equal(t, want, received[0])
	require.Equal(t, []providers.ChunkChoice{reasoningDelta(1, "", "second")}, received[1].Choices)
	require.Equal(t, "usage", received[2].ID)
	require.Equal(t, 19, received[2].Usage.TotalTokens)
	close(upstream)
	close(upstreamErrs)
	for chunk := range chunks {
		received = append(received, chunk)
	}
	require.NoError(t, <-errs)
	require.Len(t, received, 5)
	snapshot := received[3].Choices[0].Delta.Reasoning
	require.NotNil(t, snapshot)
	require.JSONEq(t, string(terminal.Delta.Reasoning.ProviderRaw), string(snapshot.ProviderRaw))
	require.Empty(t, snapshot.Content)
	terminal.Delta.Reasoning = snapshot
	want.Choices = []providers.ChunkChoice{terminal}
	want.Usage = nil
	require.Equal(t, want, received[3])
	require.Equal(t, []providers.ChunkChoice{reasoningDelta(4, "", "after terminal")}, received[4].Choices)
	require.Nil(t, received[4].Usage)
	after, err := json.Marshal(mixed)
	require.NoError(t, err)
	require.Equal(t, string(before), string(after), "upstream chunk mutated")
}
