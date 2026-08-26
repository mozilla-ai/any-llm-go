package openai

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Responses calls the OpenAI Responses API.
func (p *CompatibleProvider) Responses(
	ctx context.Context,
	params providers.ResponsesParams,
) (*providers.ResponsesResult, error) {
	if err := validateResponsesParams(params); err != nil {
		return nil, err
	}

	req, err := convertResponsesParams(params)
	if err != nil {
		return nil, err
	}

	resp, err := p.client.Responses.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	output := resp.OutputText()
	if output == "" {
		return nil, errors.NewInvalidRequestError(p.Name(), fmt.Errorf("empty output text returned from Responses API"))
	}

	return &providers.ResponsesResult{
		ID:     resp.ID,
		Model:  string(resp.Model),
		Output: output,
	}, nil
}

// convertResponsesParams converts providers.ResponsesParams to an OpenAI Responses request.
func convertResponsesParams(params providers.ResponsesParams) (responses.ResponseNewParams, error) {
	items := make(responses.ResponseInputParam, 0, len(params.Input))
	for _, item := range params.Input {
		role, err := responsesRole(item.Role)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
		items = append(items, responses.ResponseInputItemParamOfMessage(item.Content, role))
	}

	req := responses.ResponseNewParams{
		Model: shared.ResponsesModel(params.Model),
		Input: responses.ResponseNewParamsInputUnion{
			OfInputItemList: items,
		},
	}

	if params.Instructions != "" {
		req.Instructions = openai.String(params.Instructions)
	}
	if params.MaxTokens != nil {
		req.MaxOutputTokens = openai.Int(int64(*params.MaxTokens))
	}
	if params.Reasoning != "" && params.Reasoning != providers.ReasoningEffortNone {
		req.Reasoning = shared.ReasoningParam{
			Effort: shared.ReasoningEffort(params.Reasoning),
		}
	}

	return req, nil
}

// responsesRole maps a normalized role onto a Responses API role.
func responsesRole(role string) (responses.EasyInputMessageRole, error) {
	switch role {
	case providers.RoleUser:
		return responses.EasyInputMessageRoleUser, nil
	case providers.RoleAssistant:
		return responses.EasyInputMessageRoleAssistant, nil
	case providers.RoleSystem:
		return responses.EasyInputMessageRoleSystem, nil
	default:
		return "", fmt.Errorf("unsupported responses role %q", role)
	}
}

// validateResponsesParams validates Responses API parameters.
func validateResponsesParams(params providers.ResponsesParams) error {
	if params.Model == "" {
		return errors.NewInvalidRequestError("", fmt.Errorf("model is required"))
	}
	if len(params.Input) == 0 {
		return errors.NewInvalidRequestError("", fmt.Errorf("at least one input item is required"))
	}
	for _, item := range params.Input {
		if _, err := responsesRole(item.Role); err != nil {
			return errors.NewInvalidRequestError("", err)
		}
	}
	return nil
}
