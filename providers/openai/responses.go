package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// CreateResponse sends the official SDK request without narrowing its input,
// tool, structured-output, or provider extension unions.
func (p *CompatibleProvider) CreateResponse(
	ctx context.Context,
	params responses.ResponseNewParams,
) (*responses.Response, error) {
	if err := p.requireCapability(p.Capabilities().Responses, "responses"); err != nil {
		return nil, err
	}

	resp, err := p.client.Responses.New(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return resp, nil
}

// Responses sends the portable request subset and normalizes common output
// items. CreateResponse remains available for the complete SDK surface.
func (p *CompatibleProvider) Responses(
	ctx context.Context,
	params providers.ResponsesParams,
) (*providers.ResponsesResult, error) {
	req, err := convertResponsesParams(p.Name(), params)
	if err != nil {
		return nil, err
	}

	resp, err := p.CreateResponse(ctx, req)
	if err != nil {
		return nil, err
	}

	return normalizeResponse(p.Name(), resp)
}

// StreamResponse exposes each typed SDK event. A stream that ends without a
// documented terminal lifecycle event is reported as truncated.
func (p *CompatibleProvider) StreamResponse(
	ctx context.Context,
	params responses.ResponseNewParams,
) (<-chan responses.ResponseStreamEventUnion, <-chan error) {
	events := make(chan responses.ResponseStreamEventUnion)
	errs := make(chan error, 1)

	go func() {
		defer close(events)
		defer close(errs)

		if err := p.requireCapability(p.Capabilities().ResponsesStreaming, "responses streaming"); err != nil {
			reportResponseStreamError(errs, err)
			return
		}

		stream := p.client.Responses.NewStreaming(ctx, params)
		defer func() {
			if err := stream.Close(); err != nil {
				reportResponseStreamError(errs, p.ConvertError(err))
			}
		}()

		terminal := false
		for stream.Next() {
			event := stream.Current()
			switch event.Type {
			case "response.completed", "response.incomplete", "response.failed":
				terminal = true
			}

			select {
			case events <- event:
			case <-ctx.Done():
				reportResponseStreamError(errs, ctx.Err())
				return
			}
		}

		if err := ctx.Err(); err != nil {
			reportResponseStreamError(errs, err)
			return
		}
		if err := stream.Err(); err != nil {
			reportResponseStreamError(errs, p.ConvertError(err))
			return
		}
		if !terminal {
			reportResponseStreamError(errs, p.ConvertError(fmt.Errorf(
				"response stream ended without a terminal event: %w",
				io.ErrUnexpectedEOF,
			)))
		}
	}()

	return events, errs
}

func (p *CompatibleProvider) requireCapability(enabled bool, operation string) error {
	if enabled {
		return nil
	}

	return errors.NewUnsupportedOperationError(p.Name(), operation, nil)
}

func convertResponsesParams(
	providerName string,
	params providers.ResponsesParams,
) (responses.ResponseNewParams, error) {
	req := responses.ResponseNewParams{Model: shared.ResponsesModel(params.Model)}
	if len(params.Input) > 0 {
		items := make(responses.ResponseInputParam, 0, len(params.Input))
		for _, item := range params.Input {
			var role responses.EasyInputMessageRole
			switch item.Role {
			case providers.ResponsesInputRoleAssistant:
				role = responses.EasyInputMessageRoleAssistant
			case providers.ResponsesInputRoleDeveloper:
				role = responses.EasyInputMessageRoleDeveloper
			case providers.ResponsesInputRoleSystem:
				role = responses.EasyInputMessageRoleSystem
			case providers.ResponsesInputRoleUser:
				role = responses.EasyInputMessageRoleUser
			default:
				return responses.ResponseNewParams{}, errors.NewInvalidRequestError(
					providerName,
					fmt.Errorf("unsupported Responses role %q", item.Role),
				)
			}
			items = append(items, responses.ResponseInputItemParamOfMessage(item.Content, role))
		}
		req.Input = responses.ResponseNewParamsInputUnion{OfInputItemList: items}
	}

	if params.Instructions != nil {
		req.Instructions = openaisdk.String(*params.Instructions)
	}
	if params.MaxOutputTokens != nil {
		req.MaxOutputTokens = openaisdk.Int(int64(*params.MaxOutputTokens))
	}
	if params.ReasoningEffort != "" && params.ReasoningEffort != providers.ReasoningEffortAuto {
		// OpenAI documents this complete API-wide set. Individual models can
		// support a subset and remain responsible for model-specific validation.
		// https://developers.openai.com/api/reference/resources/responses/methods/create
		switch params.ReasoningEffort {
		case providers.ReasoningEffortNone,
			providers.ReasoningEffortMinimal,
			providers.ReasoningEffortLow,
			providers.ReasoningEffortMedium,
			providers.ReasoningEffortHigh,
			providers.ReasoningEffortXHigh,
			providers.ReasoningEffortMax:
		default:
			return responses.ResponseNewParams{}, errors.NewUnsupportedParamError(
				providerName,
				"reasoning_effort",
			)
		}
		req.Reasoning = shared.ReasoningParam{
			Effort: shared.ReasoningEffort(params.ReasoningEffort),
		}
	}

	return req, nil
}

func normalizeResponse(providerName string, resp *responses.Response) (*providers.ResponsesResult, error) {
	if !resp.JSON.ID.Valid() || !resp.JSON.Model.Valid() || !resp.JSON.Object.Valid() ||
		resp.Object != "response" || !resp.JSON.Output.Valid() {
		return nil, errors.NewProviderError(providerName, fmt.Errorf("malformed Responses API result"))
	}

	result := &providers.ResponsesResult{
		ID:     resp.ID,
		Model:  string(resp.Model),
		Status: string(resp.Status),
		// A valid response can contain only tool or reasoning output.
		OutputText:  resp.OutputText(),
		OutputItems: normalizeResponseOutput(resp.Output),
		ProviderRaw: json.RawMessage(resp.RawJSON()),
	}
	if resp.JSON.Error.Valid() {
		result.Error = &providers.ResponsesError{
			Code:        string(resp.Error.Code),
			Message:     resp.Error.Message,
			ProviderRaw: json.RawMessage(resp.Error.RawJSON()),
		}
	}
	if resp.JSON.IncompleteDetails.Valid() {
		result.IncompleteDetails = &providers.ResponsesIncompleteDetails{
			Reason:      resp.IncompleteDetails.Reason,
			ProviderRaw: json.RawMessage(resp.IncompleteDetails.RawJSON()),
		}
	}
	if resp.JSON.Usage.Valid() {
		result.Usage = &providers.ResponsesUsage{
			InputTokens:     int(resp.Usage.InputTokens),
			OutputTokens:    int(resp.Usage.OutputTokens),
			TotalTokens:     int(resp.Usage.TotalTokens),
			CachedTokens:    int(resp.Usage.InputTokensDetails.CachedTokens),
			ReasoningTokens: int(resp.Usage.OutputTokensDetails.ReasoningTokens),
		}
	}

	return result, nil
}

func normalizeResponseOutput(items []responses.ResponseOutputItemUnion) []providers.ResponsesOutputItem {
	result := make([]providers.ResponsesOutputItem, 0, len(items))
	for _, item := range items {
		out := providers.ResponsesOutputItem{
			Type:        item.Type,
			ID:          item.ID,
			Status:      item.Status,
			ProviderRaw: json.RawMessage(item.RawJSON()),
		}
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				out.Content = append(out.Content, providers.ResponsesOutputContent{
					Type:        content.Type,
					Text:        content.Text,
					Refusal:     content.Refusal,
					ProviderRaw: json.RawMessage(content.RawJSON()),
				})
			}
		case "function_call":
			call := item.AsFunctionCall()
			out.Name = call.Name
			out.CallID = call.CallID
			out.Arguments = call.Arguments
		case "reasoning":
			for _, summary := range item.Summary {
				out.Summary = append(out.Summary, summary.Text)
			}
		}
		result = append(result, out)
	}

	return result
}

func reportResponseStreamError(errs chan<- error, err error) {
	select {
	case errs <- err:
	default:
	}
}
