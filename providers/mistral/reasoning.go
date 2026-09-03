package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/big"
	"strings"

	oaisdk "github.com/openai/openai-go/v3"

	"github.com/mozilla-ai/any-llm-go/providers"
)

type responseMessage struct {
	Content json.RawMessage `json:"content"`
}

// Keep current documented fields typed even where mistralai v2.9.4 drops or
// coerces malformed optional values while selecting a union variant.
// https://docs.mistral.ai/api/endpoint/chat
type chunkDiscriminator struct {
	Type string `json:"type"`
}

type textContentChunk struct {
	Text *string `json:"text"`
}

type thinkingContentChunk struct {
	Closed    *bool              `json:"closed"`
	Signature *string            `json:"signature"`
	Thinking  *[]json.RawMessage `json:"thinking"`
}

type thinkingTextChunk struct {
	Text *string `json:"text"`
}

type thinkingReferenceChunk struct {
	ReferenceIDs *[]json.RawMessage `json:"reference_ids"`
}

type thinkingToolReferenceChunk struct {
	Description *string `json:"description"`
	Favicon     *string `json:"favicon"`
	Title       *string `json:"title"`
	Tool        *string `json:"tool"`
	URL         *string `json:"url"`
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

// transformChunk projects Mistral's streamed content union while retaining
// each complete array for replay in a later assistant message.
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
	}

	if hasAnswer {
		content = new(answerBuilder.String())
	}

	if hasThinking {
		reasoning = &providers.Reasoning{
			Content:     strings.Join(reasoningParts, "\n"),
			ProviderRaw: bytes.Clone(raw),
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

	switch discriminator.Type {
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
		return chunkProjection{}, fmt.Errorf("text content chunk is missing text")
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
		return chunkProjection{}, fmt.Errorf("thinking content chunk is missing thinking")
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

	switch discriminator.Type {
	case "text":
		return decodeThinkingTextChunk(raw)
	case "reference":
		return decodeThinkingReferenceChunk(raw)
	case "tool_reference":
		return decodeThinkingToolReferenceChunk(raw)
	default:
		return "", false, fmt.Errorf("unsupported thinking chunk type %q", discriminator.Type)
	}
}

func decodeThinkingTextChunk(raw json.RawMessage) (string, bool, error) {
	var chunk thinkingTextChunk
	err := json.Unmarshal(raw, &chunk)
	if err != nil {
		return "", false, fmt.Errorf("decoding thinking text chunk: %w", err)
	}

	if chunk.Text == nil {
		return "", false, fmt.Errorf("thinking text chunk is missing text")
	}

	return *chunk.Text, true, nil
}

func decodeThinkingReferenceChunk(raw json.RawMessage) (string, bool, error) {
	var chunk thinkingReferenceChunk
	err := json.Unmarshal(raw, &chunk)
	if err != nil {
		return "", false, fmt.Errorf("decoding thinking reference chunk: %w", err)
	}

	if chunk.ReferenceIDs == nil {
		return "", false, fmt.Errorf("thinking reference chunk is missing reference_ids")
	}

	for _, referenceID := range *chunk.ReferenceIDs {
		if !validReferenceID(referenceID) {
			return "", false, fmt.Errorf("thinking reference_id must be an integer or string")
		}
	}

	return "", false, nil
}

func decodeThinkingToolReferenceChunk(raw json.RawMessage) (string, bool, error) {
	var chunk thinkingToolReferenceChunk
	err := json.Unmarshal(raw, &chunk)
	if err != nil {
		return "", false, fmt.Errorf("decoding thinking tool_reference chunk: %w", err)
	}

	if chunk.Tool == nil || chunk.Title == nil {
		return "", false, fmt.Errorf("thinking tool_reference chunk is missing tool or title")
	}

	return "", false, nil
}

func validReferenceID(raw json.RawMessage) bool {
	var text *string
	err := json.Unmarshal(raw, &text)
	if err == nil && text != nil {
		return true
	}

	var number *json.Number
	if json.Unmarshal(raw, &number) != nil || number == nil {
		return false
	}

	rational, ok := new(big.Rat).SetString(number.String())
	return ok && rational.IsInt()
}
