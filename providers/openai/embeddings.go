package openai

import (
	"context"
	"fmt"
	"slices"

	"github.com/openai/openai-go/v3"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Embedding generates embeddings for the given input.
func (p *CompatibleProvider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	if !p.Capabilities().Embedding {
		return nil, errors.NewUnsupportedOperationError(p.Name(), "embedding", nil)
	}

	req, err := convertEmbeddingParams(params)
	if err != nil {
		return nil, errors.NewInvalidRequestError(p.Name(), err)
	}

	resp, err := p.client.Embeddings.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertEmbeddingResponse(resp), nil
}

// convertEmbeddingParams converts the four input shapes documented by the
// OpenAI Embeddings API. The public binding also accepts Go's native int shape
// and converts it without changing the wire representation.
// https://developers.openai.com/api/reference/resources/embeddings/methods/create
func convertEmbeddingParams(params providers.EmbeddingParams) (openai.EmbeddingNewParams, error) {
	req := openai.EmbeddingNewParams{
		Model: params.Model,
	}

	switch input := params.Input.(type) {
	case string:
		req.Input = openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(input),
		}
	case []string:
		req.Input = openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: input,
		}
	case []int:
		tokens := make([]int64, len(input))
		for index, token := range input {
			tokens[index] = int64(token)
		}
		req.Input = openai.EmbeddingNewParamsInputUnion{OfArrayOfTokens: tokens}
	case []int64:
		req.Input = openai.EmbeddingNewParamsInputUnion{OfArrayOfTokens: input}
	case [][]int:
		tokenArrays := make([][]int64, len(input))
		for arrayIndex, tokenArray := range input {
			tokenArrays[arrayIndex] = make([]int64, len(tokenArray))
			for tokenIndex, token := range tokenArray {
				tokenArrays[arrayIndex][tokenIndex] = int64(token)
			}
		}
		req.Input = openai.EmbeddingNewParamsInputUnion{OfArrayOfTokenArrays: tokenArrays}
	case [][]int64:
		req.Input = openai.EmbeddingNewParamsInputUnion{OfArrayOfTokenArrays: input}
	default:
		return openai.EmbeddingNewParams{}, fmt.Errorf(
			"embedding input must be string, []string, []int, []int64, [][]int, or [][]int64, got %T",
			params.Input,
		)
	}

	if params.EncodingFormat != "" {
		req.EncodingFormat = openai.EmbeddingNewParamsEncodingFormat(params.EncodingFormat)
	}

	if params.Dimensions != nil {
		req.Dimensions = openai.Int(int64(*params.Dimensions))
	}

	if params.User != "" {
		req.User = openai.String(params.User)
	}

	return req, nil
}

// convertEmbeddingResponse converts an OpenAI embedding response to provider format.
func convertEmbeddingResponse(resp *openai.CreateEmbeddingResponse) *providers.EmbeddingResponse {
	data := make([]providers.EmbeddingData, 0, len(resp.Data))
	for _, d := range resp.Data {
		data = append(data, providers.EmbeddingData{
			Object:    objectEmbedding,
			Embedding: slices.Clone(d.Embedding),
			Index:     int(d.Index),
		})
	}

	result := &providers.EmbeddingResponse{
		Object: objectList,
		Data:   data,
		Model:  resp.Model,
	}

	if resp.Usage.PromptTokens > 0 || resp.Usage.TotalTokens > 0 {
		result.Usage = &providers.EmbeddingUsage{
			PromptTokens: int(resp.Usage.PromptTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
		}
	}

	return result
}
