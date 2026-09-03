package openai

import (
	"fmt"

	"github.com/openai/openai-go/v3"

	"github.com/mozilla-ai/any-llm-go/providers"
)

const (
	contentTypeImageURL = "image_url"
	contentTypeText     = "text"
	toolTypeFunction    = "function"
)

type chatCompletionMessageConverter func(providers.Message) (openai.ChatCompletionMessageParamUnion, error)

func convertCompatibleAssistantMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	toolCalls, err := convertFunctionToolCalls(msg.ToolCalls)
	if err != nil {
		return openai.ChatCompletionMessageParamUnion{}, err
	}

	if len(toolCalls) == 0 {
		return openai.AssistantMessage(msg.ContentString()), nil
	}

	return openai.ChatCompletionMessageParamUnion{
		OfAssistant: &openai.ChatCompletionAssistantMessageParam{
			Content: openai.ChatCompletionAssistantMessageParamContentUnion{
				OfString: openai.String(msg.ContentString()),
			},
			ToolCalls: toolCalls,
		},
	}, nil
}

func convertOpenAIAssistantMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	toolCalls, err := convertFunctionToolCalls(msg.ToolCalls)
	if err != nil {
		return openai.ChatCompletionMessageParamUnion{}, err
	}

	assistant := &openai.ChatCompletionAssistantMessageParam{}
	if msg.Name != "" {
		assistant.Name = openai.String(msg.Name)
	}

	if msg.IsMultiModal() {
		parts, err := convertTextContentParts(msg)
		if err != nil {
			return openai.ChatCompletionMessageParamUnion{}, err
		}

		assistant.Content.OfArrayOfContentParts = make(
			[]openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion,
			0,
			len(parts),
		)
		for i := range parts {
			assistant.Content.OfArrayOfContentParts = append(
				assistant.Content.OfArrayOfContentParts,
				openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{OfText: &parts[i]},
			)
		}
		// OpenAI permits content to be omitted only when tool_calls or the deprecated
		// function_call is present. The normalized type does not expose function_call.
		// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
	} else if msg.Content != nil || len(msg.ToolCalls) == 0 {
		assistant.Content.OfString = openai.String(msg.ContentString())
	}

	if len(toolCalls) > 0 {
		assistant.ToolCalls = toolCalls
	}

	return openai.ChatCompletionMessageParamUnion{OfAssistant: assistant}, nil
}

func convertOpenAIMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	switch msg.Role {
	case providers.RoleAssistant:
		return convertOpenAIAssistantMessage(msg)
	case providers.RoleDeveloper:
		return convertOpenAIDeveloperMessage(msg)
	case providers.RoleSystem:
		return convertOpenAISystemMessage(msg)
	case providers.RoleTool:
		return convertTextMessage(
			msg,
			func(content string) openai.ChatCompletionMessageParamUnion {
				return openai.ToolMessage(content, msg.ToolCallID)
			},
			func(content []openai.ChatCompletionContentPartTextParam) openai.ChatCompletionMessageParamUnion {
				return openai.ToolMessage(content, msg.ToolCallID)
			},
		)
	case providers.RoleUser:
		return convertOpenAIUserMessage(msg)
	default:
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unknown message role: %q", msg.Role)
	}
}

func convertCompatibleMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	switch msg.Role {
	case providers.RoleAssistant:
		return convertCompatibleAssistantMessage(msg)
	case providers.RoleSystem:
		return openai.SystemMessage(msg.ContentString()), nil
	case providers.RoleTool:
		return openai.ToolMessage(msg.ContentString(), msg.ToolCallID), nil
	case providers.RoleUser:
		return convertCompatibleUserMessage(msg), nil
	default:
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unknown message role: %q", msg.Role)
	}
}

func convertOpenAIDeveloperMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	message, err := convertTextMessage(
		msg,
		openai.DeveloperMessage[string],
		openai.DeveloperMessage[[]openai.ChatCompletionContentPartTextParam],
	)
	if err == nil && msg.Name != "" {
		message.OfDeveloper.Name = openai.String(msg.Name)
	}

	return message, err
}

func convertOpenAISystemMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	message, err := convertTextMessage(
		msg,
		openai.SystemMessage[string],
		openai.SystemMessage[[]openai.ChatCompletionContentPartTextParam],
	)
	if err == nil && msg.Name != "" {
		message.OfSystem.Name = openai.String(msg.Name)
	}

	return message, err
}

func convertFunctionToolCalls(
	calls []providers.ToolCall,
) ([]openai.ChatCompletionMessageToolCallUnionParam, error) {
	toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(calls))
	for _, call := range calls {
		// The normalized ToolCall currently carries only a function payload. OpenAI
		// also documents custom tool calls, so reject them instead of silently
		// serializing the wrong union variant.
		// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
		if call.Type != "" && call.Type != toolTypeFunction {
			return nil, fmt.Errorf("unsupported tool call type: %q", call.Type)
		}
		toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
			OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
				ID: call.ID,
				Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
					Name:      call.Function.Name,
					Arguments: call.Function.Arguments,
				},
			},
		})
	}

	return toolCalls, nil
}

func convertTextMessage(
	msg providers.Message,
	fromString func(string) openai.ChatCompletionMessageParamUnion,
	fromParts func([]openai.ChatCompletionContentPartTextParam) openai.ChatCompletionMessageParamUnion,
) (openai.ChatCompletionMessageParamUnion, error) {
	if !msg.IsMultiModal() {
		return fromString(msg.ContentString()), nil
	}

	parts, err := convertTextContentParts(msg)
	if err != nil {
		return openai.ChatCompletionMessageParamUnion{}, err
	}

	return fromParts(parts), nil
}

func convertTextContentParts(msg providers.Message) ([]openai.ChatCompletionContentPartTextParam, error) {
	parts := make([]openai.ChatCompletionContentPartTextParam, 0, len(msg.ContentParts()))
	for _, part := range msg.ContentParts() {
		if part.Type != contentTypeText {
			return nil, fmt.Errorf("unsupported %s content part type: %q", msg.Role, part.Type)
		}

		if part.ImageURL != nil {
			return nil, fmt.Errorf("text content requires only text")
		}
		parts = append(parts, openai.ChatCompletionContentPartTextParam{Text: part.Text})
	}

	return parts, nil
}

func convertMessagesWith(
	messages []providers.Message,
	converter chatCompletionMessageConverter,
) ([]openai.ChatCompletionMessageParamUnion, error) {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, msg := range messages {
		converted, err := converter(msg)
		if err != nil {
			return nil, err
		}

		result = append(result, converted)
	}

	return result, nil
}

func convertCompatibleUserMessage(msg providers.Message) openai.ChatCompletionMessageParamUnion {
	if !msg.IsMultiModal() {
		return openai.UserMessage(msg.ContentString())
	}

	// Preserve the package's historical compatible-provider schema. OpenAI-only
	// fields remain gated until each provider's public contract confirms them.
	parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(msg.ContentParts()))
	for _, part := range msg.ContentParts() {
		switch part.Type {
		case contentTypeText:
			parts = append(parts, openai.TextContentPart(part.Text))
		case contentTypeImageURL:
			if part.ImageURL != nil {
				parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
					URL: part.ImageURL.URL,
				}))
			}
		}
	}

	return openai.UserMessage(parts)
}

func convertOpenAIUserMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	if !msg.IsMultiModal() {
		message := openai.UserMessage(msg.ContentString())
		if msg.Name != "" {
			message.OfUser.Name = openai.String(msg.Name)
		}

		return message, nil
	}

	parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(msg.ContentParts()))
	for _, part := range msg.ContentParts() {
		switch part.Type {
		case contentTypeText:
			if part.ImageURL != nil {
				return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("text content requires only text")
			}

			parts = append(parts, openai.TextContentPart(part.Text))
		case contentTypeImageURL:
			image, err := convertOpenAIImageContentPart(part)
			if err != nil {
				return openai.ChatCompletionMessageParamUnion{}, err
			}

			parts = append(parts, image)
		default:
			return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf(
				"unsupported user content part type: %q",
				part.Type,
			)
		}
	}

	message := openai.UserMessage(parts)
	if msg.Name != "" {
		message.OfUser.Name = openai.String(msg.Name)
	}

	return message, nil
}

func convertOpenAIImageContentPart(
	part providers.ContentPart,
) (openai.ChatCompletionContentPartUnionParam, error) {
	if part.ImageURL == nil || part.ImageURL.URL == "" || part.Text != "" {
		return openai.ChatCompletionContentPartUnionParam{}, fmt.Errorf(
			"image_url content requires only a non-empty URL",
		)
	}

	return openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
		URL:    part.ImageURL.URL,
		Detail: part.ImageURL.Detail,
	}), nil
}

func (p *CompatibleProvider) messageConverter() chatCompletionMessageConverter {
	if p.compatibleConfig.OpenAIMessageSchema {
		return convertOpenAIMessage
	}

	return convertCompatibleMessage
}
