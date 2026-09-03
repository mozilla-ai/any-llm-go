package deepseek

import (
	"fmt"

	oaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

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
