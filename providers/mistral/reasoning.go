package mistral

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	oaisdk "github.com/openai/openai-go/v3"

	"github.com/mozilla-ai/any-llm-go/providers"
)

type responseMessage struct {
	Content json.RawMessage `json:"content"`
}

type chunkDiscriminator struct {
	Type *string `json:"type"`
}

type textContentChunk struct {
	Text *string `json:"text"`
}

type thinkingContentChunk struct {
	Closed    *bool              `json:"closed"`
	Signature *string            `json:"signature"`
	Thinking  *[]json.RawMessage `json:"thinking"`
}

type chunkProjection struct {
	answer         *string
	reasoningParts []string
	thinking       bool
}

// transformResponse projects Mistral's content union while retaining the
// complete array required for multi-turn replay.
// https://docs.mistral.ai/studio-api/conversations/reasoning
func transformResponse(source *oaisdk.ChatCompletion, result *providers.ChatCompletion) error {
	for choiceIndex, choice := range source.Choices {
		var message responseMessage

		err := json.Unmarshal([]byte(choice.Message.RawJSON()), &message)
		if err != nil {
			return fmt.Errorf("decoding Mistral choice %d message: %w", choiceIndex, err)
		}

		content, reasoning, chunked, err := decodeContent(message.Content)
		if err != nil {
			return fmt.Errorf("decoding Mistral choice %d content: %w", choiceIndex, err)
		}

		if !chunked {
			continue
		}

		if content == nil {
			result.Choices[choiceIndex].Message.Content = nil
		} else {
			result.Choices[choiceIndex].Message.Content = *content
		}

		result.Choices[choiceIndex].Message.Reasoning = reasoning
	}

	return nil
}

// transformChunk retains metadata from this delta, not a cumulative message.
// Callers must not treat a fragment as complete reasoning replay content.
// https://docs.mistral.ai/studio-api/conversations/reasoning
func transformChunk(source *oaisdk.ChatCompletionChunk, result *providers.ChatCompletionChunk) error {
	for choiceIndex, choice := range source.Choices {
		var delta responseMessage

		err := json.Unmarshal([]byte(choice.Delta.RawJSON()), &delta)
		if err != nil {
			return fmt.Errorf("decoding Mistral choice %d delta: %w", choiceIndex, err)
		}

		content, reasoning, chunked, err := decodeContent(delta.Content)
		if err != nil {
			return fmt.Errorf("decoding Mistral choice %d delta content: %w", choiceIndex, err)
		}

		if !chunked {
			continue
		}

		if content == nil {
			result.Choices[choiceIndex].Delta.Content = ""
		} else {
			result.Choices[choiceIndex].Delta.Content = *content
		}

		// The assembler needs text-array metadata even before thinking begins.
		// It removes this text-only carrier before forwarding public chunks.
		if reasoning == nil {
			reasoning = &providers.Reasoning{ProviderRaw: delta.Content, Provider: providerName}
		}
		result.Choices[choiceIndex].Delta.Reasoning = reasoning
	}

	return nil
}

func decodeContent(
	raw json.RawMessage,
) (*string, *providers.Reasoning, bool, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return nil, nil, false, nil
	}

	var rawChunks []json.RawMessage

	err := json.Unmarshal(trimmed, &rawChunks)
	if err != nil {
		return nil, nil, false, fmt.Errorf("decoding content chunks: %w", err)
	}

	var (
		answerBuilder  strings.Builder
		content        *string
		hasAnswer      bool
		hasThinking    bool
		hasUnprojected bool
		reasoning      *providers.Reasoning
		reasoningParts []string
	)

	for _, rawChunk := range rawChunks {
		projection, err := decodeContentChunk(rawChunk)
		if err != nil {
			return nil, nil, false, err
		}

		if projection.answer != nil {
			answerBuilder.WriteString(*projection.answer)

			hasAnswer = true
		}

		if projection.thinking {
			hasThinking = true

			reasoningParts = append(reasoningParts, projection.reasoningParts...)
		}

		if projection.answer == nil && !projection.thinking {
			hasUnprojected = true
		}
	}

	// Thinking responses retain the complete array for replay. Otherwise there
	// is no raw-content slot in the normalized message, so do not return partial success.
	if hasUnprojected && !hasThinking {
		return nil, nil, false, errors.New("cannot preserve non-text content in this response")
	}

	if hasAnswer {
		content = new(answerBuilder.String())
	}

	if hasThinking {
		reasoning = &providers.Reasoning{
			Content:     strings.Join(reasoningParts, "\n"),
			ProviderRaw: bytes.Clone(raw),
			Provider:    providerName,
		}
	}

	return content, reasoning, true, nil
}

func decodeContentChunk(raw json.RawMessage) (chunkProjection, error) {
	var discriminator chunkDiscriminator

	err := json.Unmarshal(raw, &discriminator)
	if err != nil {
		return chunkProjection{}, fmt.Errorf("decoding content chunk: %w", err)
	}

	// An unknown union arm still has a discriminator; null/missing chunks do not.
	// https://github.com/mistralai/client-python/blob/df37126528859588193a9e954e820b12861ad106/
	// src/mistralai/client/utils/unions.py
	if discriminator.Type == nil {
		return chunkProjection{}, errors.New("content chunk is missing type")
	}

	switch *discriminator.Type {
	case "text":
		return decodeTextContentChunk(raw)
	case "thinking":
		return decodeThinkingContentChunk(raw)
	default:
		return chunkProjection{}, nil
	}
}

func decodeTextContentChunk(raw json.RawMessage) (chunkProjection, error) {
	var chunk textContentChunk

	err := json.Unmarshal(raw, &chunk)
	if err != nil {
		return chunkProjection{}, fmt.Errorf("decoding text content chunk: %w", err)
	}

	if chunk.Text == nil {
		return chunkProjection{}, errors.New("text content chunk is missing text")
	}

	return chunkProjection{answer: chunk.Text}, nil
}

func decodeThinkingContentChunk(raw json.RawMessage) (chunkProjection, error) {
	var chunk thinkingContentChunk

	err := json.Unmarshal(raw, &chunk)
	if err != nil {
		return chunkProjection{}, fmt.Errorf("decoding thinking content chunk: %w", err)
	}

	if chunk.Thinking == nil {
		return chunkProjection{}, errors.New("thinking content chunk is missing thinking")
	}

	parts := make([]string, 0, len(*chunk.Thinking))

	for _, rawThinking := range *chunk.Thinking {
		text, present, decodeErr := decodeThinking(rawThinking)
		if decodeErr != nil {
			return chunkProjection{}, decodeErr
		}

		if present {
			parts = append(parts, text)
		}
	}

	return chunkProjection{reasoningParts: parts, thinking: true}, nil
}

func decodeThinking(raw json.RawMessage) (string, bool, error) {
	var discriminator chunkDiscriminator

	err := json.Unmarshal(raw, &discriminator)
	if err != nil {
		return "", false, fmt.Errorf("decoding thinking chunk: %w", err)
	}

	if discriminator.Type == nil {
		return "", false, errors.New("thinking chunk is missing type")
	}

	switch *discriminator.Type {
	case "text":
		return decodeThinkingTextChunk(raw)
	case "reference", "tool_reference":
		// Reference metadata is replayed verbatim, not interpreted for text output.
		return "", false, nil
	default:
		return "", false, fmt.Errorf("unsupported thinking chunk type %q", *discriminator.Type)
	}
}

func decodeThinkingTextChunk(raw json.RawMessage) (string, bool, error) {
	var chunk textContentChunk

	err := json.Unmarshal(raw, &chunk)
	if err != nil {
		return "", false, fmt.Errorf("decoding thinking text chunk: %w", err)
	}

	if chunk.Text == nil {
		return "", false, errors.New("thinking text chunk is missing text")
	}

	return *chunk.Text, true, nil
}
