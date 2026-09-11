package mistral

import (
	"fmt"

	oaisdk "github.com/openai/openai-go/v3"

	"github.com/mozilla-ai/any-llm-go/providers"
)

func replayReasoning(
	messages []providers.Message,
	converted []oaisdk.ChatCompletionMessageParamUnion,
) error {
	for messageIndex, message := range messages {
		if message.Role != providers.RoleAssistant || message.Reasoning == nil {
			continue
		}
		if message.Reasoning.Provider != providerName || len(message.Reasoning.ProviderRaw) == 0 {
			if message.Reasoning.Content == "" {
				continue
			}

			answer, isText := message.Content.(string)
			if message.Content != nil && !isText {
				return fmt.Errorf(
					"encoding Mistral message %d: reasoning replay requires string or nil content",
					messageIndex,
				)
			}

			content := []map[string]any{{
				"type": "thinking",
				"thinking": []map[string]string{{
					"type": "text",
					"text": message.Reasoning.Content,
				}},
			}}
			if answer != "" {
				content = append(content, map[string]any{
					"type": "text",
					"text": answer,
				})
			}

			assistant := converted[messageIndex].OfAssistant
			assistant.SetExtraFields(map[string]any{"content": content})

			continue
		}

		content, reasoning, chunked, err := decodeContent(message.Reasoning.ProviderRaw)
		if err != nil {
			return fmt.Errorf("encoding Mistral message %d reasoning provider_raw: %w", messageIndex, err)
		}

		if !chunked || reasoning == nil {
			return fmt.Errorf("encoding Mistral message %d: invalid reasoning provider_raw", messageIndex)
		}

		answer, isText := message.Content.(string)
		if message.Content != nil && !isText {
			return fmt.Errorf(
				"encoding Mistral message %d: reasoning replay requires string or nil content",
				messageIndex,
			)
		}

		projectedAnswer := ""
		if content != nil {
			projectedAnswer = *content
		}
		// A stream fragment must not overwrite a caller's accumulated answer.
		// Matching text does not prove that raw thinking blocks are complete.
		if projectedAnswer != answer {
			return fmt.Errorf(
				"encoding Mistral message %d: reasoning provider_raw does not match assistant content",
				messageIndex,
			)
		}

		assistant := converted[messageIndex].OfAssistant
		// The official OpenAI SDK exposes provider-specific request fields only
		// through SetExtraFields, whose signature requires map[string]any.
		assistant.SetExtraFields(map[string]any{
			"content": reasoning.ProviderRaw,
		})
	}

	return nil
}
