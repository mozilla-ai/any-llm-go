package deepseek

import (
	"encoding/json"
	"fmt"
	"net/http"

	oaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

type deepSeekChatContent struct {
	Content          json.RawMessage `json:"content"`
	ReasoningContent *string         `json:"reasoning_content"`
}

type deepSeekChoiceLogprobs struct {
	ReasoningContent []providers.ChatCompletionTokenLogprob `json:"reasoning_content"`
}

type deepSeekThinking struct {
	Type string `json:"type"`
}

// transformRequest maps normalized controls to DeepSeek's current Chat schema.
// https://api-docs.deepseek.com/api/create-chat-completion
func transformRequest(
	params providers.CompletionParams,
	req *oaisdk.ChatCompletionNewParams,
) error {
	if params.ParallelToolCalls != nil {
		return errors.NewUnsupportedParamError(providerName, "parallel_tool_calls")
	}
	if params.Seed != nil {
		return errors.NewUnsupportedParamError(providerName, "seed")
	}

	if req.MaxCompletionTokens.Valid() {
		req.MaxTokens = oaisdk.Int(req.MaxCompletionTokens.Value)
	}
	// DeepSeek documents max_tokens rather than OpenAI's active max_completion_tokens.
	req.MaxCompletionTokens = param.Opt[int64]{}

	thinking, effort, err := mapReasoningEffort(params.ReasoningEffort)
	if err != nil {
		return err
	}

	// DeepSeek names this field user_id and does not document OpenAI's user.
	req.User = param.Opt[string]{}
	req.ReasoningEffort = shared.ReasoningEffort(effort)

	extraFields := make(map[string]any, 2)
	if params.User != "" {
		extraFields["user_id"] = params.User
	}
	if thinking != "" {
		extraFields["thinking"] = deepSeekThinking{Type: thinking}
	}
	if len(extraFields) > 0 {
		// openai-go models neither DeepSeek extension; SetExtraFields is the
		// official SDK's typed request extension boundary.
		req.SetExtraFields(extraFields)
	}

	if len(params.Tools) == 0 {
		return nil
	}
	for i, message := range params.Messages {
		if message.Role != providers.RoleAssistant || message.Reasoning == nil {
			continue
		}

		// DeepSeek requires every prior assistant reasoning_content when the
		// request carries tools, including turns that did not call a tool.
		req.Messages[i].OfAssistant.SetExtraFields(map[string]any{
			"reasoning_content": message.Reasoning.Content,
		})
	}

	return nil
}

// transformAPIError follows DeepSeek's published status table instead of
// applying OpenAI-specific status and error-code assumptions.
// https://api-docs.deepseek.com/quick_start/error_codes
func transformAPIError(apiErr *oaisdk.Error, originalErr error) error {
	switch apiErr.StatusCode {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return errors.NewInvalidRequestError(providerName, originalErr)
	case http.StatusUnauthorized:
		return errors.NewAuthenticationError(providerName, originalErr)
	case http.StatusPaymentRequired:
		return errors.NewInsufficientFundsError(providerName, originalErr)
	case http.StatusTooManyRequests:
		return errors.NewRateLimitError(providerName, originalErr)
	default:
		return errors.NewProviderError(providerName, originalErr)
	}
}

func mapReasoningEffort(effort providers.ReasoningEffort) (thinking string, mapped string, err error) {
	switch effort {
	case "", providers.ReasoningEffortAuto:
		return "", "", nil
	case providers.ReasoningEffortNone:
		return "disabled", "", nil
	case providers.ReasoningEffortLow:
		return "enabled", "low", nil
	case providers.ReasoningEffortMedium, providers.ReasoningEffortHigh, providers.ReasoningEffortXHigh:
		return "enabled", "high", nil
	case providers.ReasoningEffortMax:
		return "enabled", "max", nil
	default:
		return "", "", errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("unsupported reasoning_effort %q", effort),
		)
	}
}

// transformResponse preserves DeepSeek's provider-specific reasoning output.
// https://api-docs.deepseek.com/guides/thinking_mode/
func transformResponse(source *oaisdk.ChatCompletion, result *providers.ChatCompletion) error {
	for i, choice := range source.Choices {
		content, err := decodeChatContent(choice.Message.RawJSON())
		if err != nil {
			return fmt.Errorf("decoding DeepSeek choice %d message: %w", i, err)
		}
		if len(content.Content) == 0 {
			return fmt.Errorf("decoding DeepSeek choice %d content: missing required field", i)
		}

		messageContent, err := decodeOptionalString(content.Content)
		if err != nil {
			return fmt.Errorf("decoding DeepSeek choice %d content: %w", i, err)
		}
		if messageContent == nil {
			result.Choices[i].Message.Content = nil
		} else {
			result.Choices[i].Message.Content = *messageContent
		}
		if content.ReasoningContent != nil {
			result.Choices[i].Message.Reasoning = &providers.Reasoning{Content: *content.ReasoningContent}
		}
		if err := preserveReasoningLogprobs(choice.Logprobs.RawJSON(), result.Choices[i].Logprobs); err != nil {
			return fmt.Errorf("decoding DeepSeek choice %d reasoning_content logprobs: %w", i, err)
		}
	}

	return nil
}

// transformChunk preserves streamed DeepSeek reasoning deltas.
// https://api-docs.deepseek.com/guides/thinking_mode/
func transformChunk(source *oaisdk.ChatCompletionChunk, result *providers.ChatCompletionChunk) error {
	for i, choice := range source.Choices {
		content, err := decodeChatContent(choice.Delta.RawJSON())
		if err != nil {
			return fmt.Errorf("decoding DeepSeek choice %d delta: %w", i, err)
		}
		if _, err := decodeOptionalString(content.Content); err != nil {
			return fmt.Errorf("decoding DeepSeek choice %d delta content: %w", i, err)
		}
		if content.ReasoningContent != nil {
			result.Choices[i].Delta.Reasoning = &providers.Reasoning{Content: *content.ReasoningContent}
		}
		if err := preserveReasoningLogprobs(choice.Logprobs.RawJSON(), result.Choices[i].Logprobs); err != nil {
			return fmt.Errorf("decoding DeepSeek choice %d reasoning_content logprobs: %w", i, err)
		}
	}

	return nil
}

func preserveReasoningLogprobs(raw string, result *providers.ChatCompletionLogprobs) error {
	if raw == "" || raw == "null" {
		return nil
	}

	var logprobs deepSeekChoiceLogprobs
	if err := json.Unmarshal([]byte(raw), &logprobs); err != nil {
		return err
	}
	result.ReasoningContent = logprobs.ReasoningContent
	return nil
}

func decodeChatContent(raw string) (deepSeekChatContent, error) {
	var content deepSeekChatContent
	if err := json.Unmarshal([]byte(raw), &content); err != nil {
		return deepSeekChatContent{}, err
	}
	return content, nil
}

func decodeOptionalString(raw json.RawMessage) (*string, error) {
	if len(raw) == 0 {
		return nil, nil
	}

	var value *string
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, err
	}
	return value, nil
}
