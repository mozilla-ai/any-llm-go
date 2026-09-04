package openai

import (
	"context"
	"fmt"
	"io"

	"github.com/openai/openai-go/v3/responses"

	"github.com/mozilla-ai/any-llm-go/errors"
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

func reportResponseStreamError(errs chan<- error, err error) {
	select {
	case errs <- err:
	default:
	}
}
