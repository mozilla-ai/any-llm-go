package mistral

import (
	"bytes"
	"encoding/json"
	"fmt"

	oaisdk "github.com/openai/openai-go/v3"

	"github.com/mozilla-ai/any-llm-go/providers"
)

func replayReasoning(
	messages []providers.Message,
	converted []oaisdk.ChatCompletionMessageParamUnion,
) error {
	for messageIndex, message := range messages {
		if message.Role != providers.RoleAssistant || message.Reasoning == nil ||
			len(message.Reasoning.ProviderRaw) == 0 {
			continue
		}

		_, reasoning, chunked, err := decodeContent(message.Reasoning.ProviderRaw)
		if err != nil {
			return fmt.Errorf("encoding Mistral message %d reasoning provider_raw: %w", messageIndex, err)
		}

		if !chunked || reasoning == nil {
			return fmt.Errorf("encoding Mistral message %d: invalid reasoning provider_raw", messageIndex)
		}

		assistant := converted[messageIndex].OfAssistant
		// The official OpenAI SDK exposes provider-specific request fields only
		// through SetExtraFields, whose signature requires map[string]any.
		assistant.SetExtraFields(map[string]any{
			"content": json.RawMessage(bytes.Clone(message.Reasoning.ProviderRaw)),
		})
	}

	return nil
}
