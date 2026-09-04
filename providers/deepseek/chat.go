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

type deepSeekVisionContentPart struct {
	Type     string              `json:"type"`
	Text     string              `json:"text,omitempty"`
	ImageURL *providers.ImageURL `json:"image_url,omitempty"`
	FileID   string              `json:"file_id,omitempty"`
	FileData string              `json:"file_data,omitempty"`
	Filename string              `json:"filename,omitempty"`
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
	if err := transformVisionMessages(params.Messages, req.Messages); err != nil {
		return err
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

// transformVisionMessages validates DeepSeek's user-only image contract and
// restores detail values omitted by the generic compatible converter.
// https://api-docs.deepseek.com/guides/vision
func transformVisionMessages(
	messages []providers.Message,
	reqMessages []oaisdk.ChatCompletionMessageParamUnion,
) error {
	for i, message := range messages {
		if !message.IsMultiModal() {
			continue
		}
		if message.Role != providers.RoleUser {
			return errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("messages[%d] content parts are only supported for the user role", i),
			)
		}

		wireParts := make([]deepSeekVisionContentPart, 0, len(message.ContentParts()))
		for j, part := range message.ContentParts() {
			switch part.Type {
			case "text":
				if part.ImageURL != nil || part.File != nil {
					return invalidVisionPart(i, j, "text content cannot include image_url or file")
				}
				wireParts = append(wireParts, deepSeekVisionContentPart{Type: part.Type, Text: part.Text})
			case "image_url":
				if part.ImageURL == nil || part.ImageURL.URL == "" || part.Text != "" || part.File != nil {
					return invalidVisionPart(i, j, "image_url content requires only a non-empty URL")
				}
				switch part.ImageURL.Detail {
				case "", "auto", "low", "high", "original":
				default:
					return invalidVisionPart(i, j, fmt.Sprintf("unsupported image detail %q", part.ImageURL.Detail))
				}
				wireParts = append(wireParts, deepSeekVisionContentPart{Type: part.Type, ImageURL: part.ImageURL})
			case "file":
				if part.File == nil || part.Text != "" || part.ImageURL != nil {
					return invalidVisionPart(i, j, "file content requires only file data")
				}
				hasFileID := part.File.FileID != ""
				hasFileData := part.File.FileData != ""
				if hasFileID == hasFileData {
					return invalidVisionPart(i, j, "file content requires exactly one of file_id or file_data")
				}
				if hasFileID && part.File.Filename != "" {
					return invalidVisionPart(i, j, "filename is only supported with file_data")
				}
				wireParts = append(wireParts, deepSeekVisionContentPart{
					Type:     part.Type,
					FileID:   part.File.FileID,
					FileData: part.File.FileData,
					Filename: part.File.Filename,
				})
			default:
				return invalidVisionPart(i, j, fmt.Sprintf("unsupported content type %q", part.Type))
			}
		}

		// openai-go models Chat file inputs under a nested file object. DeepSeek
		// documents file_id and file_data beside type, so replace the generated
		// content at the SDK's request-extension boundary.
		reqMessages[i].OfUser.SetExtraFields(map[string]any{"content": wireParts})
	}

	return nil
}

func invalidVisionPart(messageIndex, partIndex int, reason string) error {
	return errors.NewInvalidRequestError(
		providerName,
		fmt.Errorf("messages[%d].content[%d]: %s", messageIndex, partIndex, reason),
	)
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
