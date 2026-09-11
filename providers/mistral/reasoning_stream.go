package mistral

import (
	"context"
	"encoding/json"
	"slices"

	"github.com/mozilla-ai/any-llm-go/providers"
)

type streamedReasoning struct {
	chunks  []json.RawMessage
	closed  bool
	pending bool
}

func assembleReasoningStream(
	ctx context.Context,
	cancel context.CancelFunc,
	upstreamChunks <-chan providers.ChatCompletionChunk,
	upstreamErrs <-chan error,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer cancel()
		defer close(chunks)
		defer close(errs)

		states := make(map[int]*streamedReasoning)
		pending := make([]providers.ChatCompletionChunk, 0, 1)
		chunkInput, errInput := upstreamChunks, upstreamErrs

		for chunkInput != nil || errInput != nil {
			select {
			case <-ctx.Done():
				errs <- ctx.Err()

				return
			case chunk, ok := <-chunkInput:
				if !ok {
					chunkInput = nil

					continue
				}

				chunk.Choices = slices.Clone(chunk.Choices)
				accumulateReasoning(&chunk, states)
				attachReasoningSnapshots(&chunk, states)

				deferred := chunk
				deferred.Choices = nil
				choices := chunk.Choices
				chunk.Choices = chunk.Choices[:0]
				for _, choice := range choices {
					if state := states[choice.Index]; state != nil && state.pending {
						deferred.Choices = append(deferred.Choices, choice)
					} else {
						chunk.Choices = append(chunk.Choices, choice)
					}
				}

				if len(deferred.Choices) > 0 {
					// Only snapshots and subsequent deltas of their choices await stream success.
					if len(chunk.Choices) > 0 {
						deferred.Usage = nil
					}
					pending = append(pending, deferred)

					if len(chunk.Choices) == 0 {
						continue
					}
				}

				if !forwardChunk(ctx, chunks, chunk) {
					errs <- ctx.Err()

					return
				}
			case err, ok := <-errInput:
				if !ok {
					errInput = nil

					continue
				}

				if err != nil {
					errs <- err

					return
				}
			}
		}

		for _, chunk := range pending {
			if !forwardChunk(ctx, chunks, chunk) {
				errs <- ctx.Err()

				return
			}
		}
	}()

	return chunks, errs
}

func accumulateReasoning(chunk *providers.ChatCompletionChunk, states map[int]*streamedReasoning) {
	for choiceIndex := range chunk.Choices {
		choice := &chunk.Choices[choiceIndex]
		state := states[choice.Index]
		if choice.Delta.Reasoning != nil && len(choice.Delta.Reasoning.ProviderRaw) > 0 {
			var fragments []json.RawMessage
			if json.Unmarshal(choice.Delta.Reasoning.ProviderRaw, &fragments) == nil {
				if state == nil {
					state = &streamedReasoning{}
					states[choice.Index] = state
				}

				hasThinking := false
				for _, fragment := range fragments {
					state.chunks = append(state.chunks, fragment)

					var metadata struct {
						Type   *string `json:"type"`
						Closed *bool   `json:"closed"`
					}
					if json.Unmarshal(fragment, &metadata) == nil && metadata.Type != nil &&
						*metadata.Type == "thinking" {
						hasThinking = true
						state.closed = metadata.Closed != nil && *metadata.Closed
					}
				}
				if !hasThinking {
					choice.Delta.Reasoning = nil
				}
			}
			continue
		}

		if choice.Delta.Content != "" && choice.Delta.Reasoning == nil {
			if state == nil {
				state = &streamedReasoning{}
				states[choice.Index] = state
			}
			text, err := json.Marshal(struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{Type: "text", Text: choice.Delta.Content})
			if err == nil {
				state.chunks = append(state.chunks, text)
			}
		}
	}
}

func attachReasoningSnapshots(chunk *providers.ChatCompletionChunk, states map[int]*streamedReasoning) {
	for choiceIndex := range chunk.Choices {
		choice := &chunk.Choices[choiceIndex]

		state := states[choice.Index]
		if choice.FinishReason == "" || state == nil || !state.closed {
			continue
		}

		raw, err := json.Marshal(state.chunks)
		if err != nil {
			continue
		}

		_, reasoning, chunked, err := decodeContent(raw)
		if err == nil && chunked && reasoning != nil {
			reasoning.Content = ""
			if choice.Delta.Reasoning != nil {
				reasoning.Content = choice.Delta.Reasoning.Content
			}

			choice.Delta.Reasoning = reasoning
			state.pending = true
		}
	}
}

func forwardChunk(
	ctx context.Context,
	chunks chan<- providers.ChatCompletionChunk,
	chunk providers.ChatCompletionChunk,
) bool {
	select {
	case chunks <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}
