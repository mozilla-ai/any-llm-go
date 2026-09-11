package mistral

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func reasoningDelta(index int, raw string, content string) providers.ChunkChoice {
	var reasoning *providers.Reasoning
	if raw != "" {
		reasoning = &providers.Reasoning{ProviderRaw: json.RawMessage(raw), Provider: providerName}
	}

	return providers.ChunkChoice{
		Index: index,
		Delta: providers.ChunkDelta{Content: content, Reasoning: reasoning},
	}
}

func TestReasoningAssemblyIsStructuralAndPerChoice(t *testing.T) {
	t.Parallel()

	upstreamChunks := make(chan providers.ChatCompletionChunk, 4)
	upstreamErrs := make(chan error)
	ctx, cancel := context.WithCancel(t.Context())

	chunks, errs := assembleReasoningStream(ctx, cancel, upstreamChunks, upstreamErrs)
	upstreamChunks <- providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{
		reasoningDelta(4,
			`[{"type":"thinking","thinking":[{"type":"text","text":"four-a"},`+
				`{"type":"reference","reference_ids":["ref-4"]}],"signature":"sig-4a","closed":false}]`, ""),
		reasoningDelta(1,
			`[{"type":"thinking","thinking":[{"type":"tool_reference","tool":"lookup","args":{"x":1}}],`+
				`"signature":"sig-1","closed":false}]`, ""),
	}}

	upstreamChunks <- providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{
		reasoningDelta(4,
			`[{"type":"thinking","thinking":[],"signature":"sig-4b","closed":true},`+
				`{"type":"text","text":"transition "}]`, "transition "),
		reasoningDelta(1, `[{"type":"thinking","thinking":[],"signature":null,"closed":true}]`, ""),
	}}

	upstreamChunks <- providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{
		reasoningDelta(4, "", "answer"),
		reasoningDelta(1, "", "other"),
	}}

	upstreamChunks <- providers.ChatCompletionChunk{
		ID: "terminal", Model: "model", SystemFingerprint: "fingerprint",
		Choices: []providers.ChunkChoice{{Index: 4, FinishReason: "stop"}, {Index: 1, FinishReason: "tool_calls"}},
	}

	close(upstreamChunks)
	close(upstreamErrs)

	var received []providers.ChatCompletionChunk
	for chunk := range chunks {
		received = append(received, chunk)
	}

	require.NoError(t, <-errs)
	require.Len(t, received, 4)

	terminal := received[3]
	require.Equal(t, "terminal", terminal.ID)
	require.Equal(t, "model", terminal.Model)
	require.Equal(t, "fingerprint", terminal.SystemFingerprint)
	require.Equal(t, 4, terminal.Choices[0].Index)
	require.Equal(t, 1, terminal.Choices[1].Index)
	require.JSONEq(t, "[\n\t\t"+
		`{"type":"thinking","thinking":[{"type":"text","text":"four-a"},`+
		`{"type":"reference","reference_ids":["ref-4"]}],"signature":"sig-4a","closed":false},`+
		"\n\t\t"+`{"type":"thinking","thinking":[],"signature":"sig-4b","closed":true},`+
		"\n\t\t"+`{"type":"text","text":"transition "},`+
		"\n\t\t"+`{"type":"text","text":"answer"}`+"\n\t]",
		string(terminal.Choices[0].Delta.Reasoning.ProviderRaw))
	require.JSONEq(t, "[\n\t\t"+
		`{"type":"thinking","thinking":[{"type":"tool_reference","tool":"lookup","args":{"x":1}}],`+
		`"signature":"sig-1","closed":false},`+
		"\n\t\t"+`{"type":"thinking","thinking":[],"signature":null,"closed":true},`+
		"\n\t\t"+`{"type":"text","text":"other"}`+"\n\t]",
		string(terminal.Choices[1].Delta.Reasoning.ProviderRaw))

	replayURL, capturedRequest := mistralReplayServer(t)
	replayProvider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(replayURL))
	require.NoError(t, err)
	_, err = replayProvider.Completion(t.Context(), providers.CompletionParams{
		Model: mistralReasoningModel,
		Messages: []providers.Message{{
			Role: providers.RoleAssistant, Content: "transition answer",
			Reasoning: terminal.Choices[0].Delta.Reasoning,
		}},
	})
	require.NoError(t, err)
	require.JSONEq(
		t,
		string(terminal.Choices[0].Delta.Reasoning.ProviderRaw),
		string(capturedRequest().Messages[0].Content),
	)
}

func TestReasoningAssemblyDoesNotSnapshotIncompleteOrErroredStreams(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		terminal   bool
		streamErr  error
		closed     bool
		wantChunks int
	}{
		{name: "no terminal", closed: true, wantChunks: 1},
		{name: "reasoning not closed", terminal: true, wantChunks: 2},
		{name: "error after terminal", terminal: true, streamErr: stderrors.New("malformed stream"), closed: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			upstreamChunks := make(chan providers.ChatCompletionChunk, 2)
			upstreamErrs := make(chan error, 1)
			ctx, cancel := context.WithCancel(t.Context())
			chunks, errs := assembleReasoningStream(ctx, cancel, upstreamChunks, upstreamErrs)

			raw := `[{"type":"thinking","thinking":[{"type":"text","text":"partial"}],"closed":` +
				map[bool]string{true: "true", false: "false"}[test.closed] + `}]`
			upstreamChunks <- providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{reasoningDelta(0, raw, "")}}

			if test.terminal {
				upstreamChunks <- providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{{Index: 0, FinishReason: "stop"}}}
			}

			close(upstreamChunks)

			if test.streamErr != nil {
				upstreamErrs <- test.streamErr
			}

			close(upstreamErrs)

			var received []providers.ChatCompletionChunk
			for chunk := range chunks {
				received = append(received, chunk)
			}

			if test.streamErr != nil {
				require.ErrorIs(t, <-errs, test.streamErr)
			} else {
				require.Len(t, received, test.wantChunks)
				require.NoError(t, <-errs)
			}

			for _, chunk := range received {
				if chunk.Choices[0].FinishReason != "" {
					require.Nil(t, chunk.Choices[0].Delta.Reasoning)
				}
			}
		})
	}
}

func TestReasoningAssemblyCancelsUpstreamWhenConsumerStops(t *testing.T) {
	t.Parallel()

	upstreamChunks := make(chan providers.ChatCompletionChunk)
	upstreamErrs := make(chan error)
	ctx, cancelParent := context.WithCancel(t.Context())
	streamCtx, cancelStream := context.WithCancel(ctx)

	t.Cleanup(cancelParent)

	chunks, _ := assembleReasoningStream(streamCtx, cancelStream, upstreamChunks, upstreamErrs)

	go func() {
		select {
		case upstreamChunks <- providers.ChatCompletionChunk{ID: "unread"}:
		case <-streamCtx.Done():
		}

		close(upstreamChunks)
		close(upstreamErrs)
	}()

	cancelParent()

	select {
	case <-streamCtx.Done():
	case <-time.After(time.Second):
		t.Fatal("stream context was not cancelled")
	}

	for range chunks {
	}
}

func TestReasoningAssemblyCancelsQuietUpstream(t *testing.T) {
	t.Parallel()

	upstreamChunks := make(chan providers.ChatCompletionChunk)
	upstreamErrs := make(chan error)

	t.Cleanup(func() {
		close(upstreamChunks)
		close(upstreamErrs)
	})
	ctx, cancel := context.WithCancel(t.Context())
	chunks, errs := assembleReasoningStream(ctx, cancel, upstreamChunks, upstreamErrs)
	cancel()

	select {
	case _, ok := <-chunks:
		require.False(t, ok)
	case <-time.After(time.Second):
		t.Fatal("cancelled assembly blocked on quiet upstream")
	}

	require.ErrorIs(t, <-errs, context.Canceled)
}

func TestReasoningAssemblyReopenedThinkingIsIncomplete(t *testing.T) {
	t.Parallel()

	for _, raw := range []string{
		`[{"type":"thinking","thinking":[],"closed":false}]`,
		`[{"type":"thinking","thinking":[]}]`,
	} {
		states := make(map[int]*streamedReasoning)
		accumulateReasoning(&providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{
			reasoningDelta(0, `[{"type":"thinking","thinking":[],"closed":true}]`, ""),
		}}, states)
		accumulateReasoning(&providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{
			reasoningDelta(0, raw, ""),
		}}, states)
		terminal := providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{{Index: 0, FinishReason: "stop"}}}
		attachReasoningSnapshots(&terminal, states)
		require.Nil(t, terminal.Choices[0].Delta.Reasoning, raw)
	}
}

func TestReasoningAssemblyOwnsDecodedFragments(t *testing.T) {
	t.Parallel()

	const first = `{"type":"thinking","thinking":[],"closed":true,"reference":9007199254740993}`
	const second = `{"type":"text","text":"answer-29"}`
	choice := reasoningDelta(7, "["+first+","+second+"]", "answer-29")
	states := make(map[int]*streamedReasoning)
	accumulateReasoning(&providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{choice}}, states)
	clear(choice.Delta.Reasoning.ProviderRaw)

	require.Len(t, states[7].chunks, 2)
	require.Equal(t, first, string(states[7].chunks[0]))
	require.Equal(t, second, string(states[7].chunks[1]))
	require.True(t, states[7].closed)
}

func TestReasoningSnapshotPreservesDeltaText(t *testing.T) {
	t.Parallel()

	for _, deltaText := range []string{"", "last step"} {
		states := map[int]*streamedReasoning{0: {
			chunks: []json.RawMessage{
				json.RawMessage(
					`{"type":"thinking","thinking":[{"type":"text","text":"previous step"}],"closed":true}`,
				),
			},
			closed: true,
		}}

		terminal := providers.ChatCompletionChunk{Choices: []providers.ChunkChoice{{Index: 0, FinishReason: "stop"}}}
		if deltaText != "" {
			terminal.Choices[0].Delta.Reasoning = &providers.Reasoning{Content: deltaText}
		}

		attachReasoningSnapshots(&terminal, states)
		require.NotNil(t, terminal.Choices[0].Delta.Reasoning)
		require.Equal(t, deltaText, terminal.Choices[0].Delta.Reasoning.Content)
		require.NotEmpty(t, terminal.Choices[0].Delta.Reasoning.ProviderRaw)
	}
}
