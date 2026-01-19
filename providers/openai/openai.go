// Package openai provides an OpenAI provider implementation for any-llm.
package openai

import (
	"context"
	"encoding/json"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"

	llm "github.com/mozilla-ai/any-llm-go"
)

const (
	providerName = "openai"
	envAPIKey    = "OPENAI_API_KEY"
)

// Provider implements the llm.Provider interface for OpenAI.
type Provider struct {
	client *openai.Client
	config *llm.Config
}

// Ensure Provider implements the required interfaces.
var (
	_ llm.Provider           = (*Provider)(nil)
	_ llm.EmbeddingProvider  = (*Provider)(nil)
	_ llm.ModelLister        = (*Provider)(nil)
	_ llm.CapabilityProvider = (*Provider)(nil)
)

// New creates a new OpenAI provider.
func New(opts ...llm.Option) (*Provider, error) {
	cfg := llm.DefaultConfig()
	cfg.ApplyOptions(opts...)

	apiKey := cfg.GetAPIKeyFromEnv(envAPIKey)
	if apiKey == "" {
		return nil, llm.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	if cfg.BaseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(cfg.BaseURL))
	}

	client := openai.NewClient(clientOpts...)

	return &Provider{
		client: &client,
		config: cfg,
	}, nil
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() llm.ProviderCapabilities {
	return llm.ProviderCapabilities{
		Completion:          true,
		CompletionStreaming: true,
		CompletionReasoning: true,
		CompletionImage:     true,
		CompletionPDF:       false,
		Embedding:           true,
		ListModels:          true,
	}
}

// Completion performs a chat completion request.
func (p *Provider) Completion(ctx context.Context, params llm.CompletionParams) (*llm.ChatCompletion, error) {
	req := convertParams(params)

	resp, err := p.client.Chat.Completions.New(ctx, req)
	if err != nil {
		return nil, llm.ConvertError(providerName, err)
	}

	return convertResponse(resp), nil
}

// CompletionStream performs a streaming chat completion request.
func (p *Provider) CompletionStream(ctx context.Context, params llm.CompletionParams) (<-chan llm.ChatCompletionChunk, <-chan error) {
	chunks := make(chan llm.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		req := convertParams(params)

		stream := p.client.Chat.Completions.NewStreaming(ctx, req)

		for stream.Next() {
			chunk := stream.Current()
			chunks <- convertChunk(&chunk)
		}

		if err := stream.Err(); err != nil {
			errs <- llm.ConvertError(providerName, err)
		}
	}()

	return chunks, errs
}

// Embedding performs an embedding request.
func (p *Provider) Embedding(ctx context.Context, params llm.EmbeddingParams) (*llm.EmbeddingResponse, error) {
	req := convertEmbeddingParams(params)

	resp, err := p.client.Embeddings.New(ctx, req)
	if err != nil {
		return nil, llm.ConvertError(providerName, err)
	}

	return convertEmbeddingResponse(resp), nil
}

// ListModels lists available models.
func (p *Provider) ListModels(ctx context.Context) (*llm.ModelsResponse, error) {
	resp, err := p.client.Models.List(ctx)
	if err != nil {
		return nil, llm.ConvertError(providerName, err)
	}

	models := make([]llm.Model, 0, len(resp.Data))
	for _, model := range resp.Data {
		models = append(models, llm.Model{
			ID:      model.ID,
			Object:  "model",
			Created: model.Created,
			OwnedBy: string(model.OwnedBy),
		})
	}

	return &llm.ModelsResponse{
		Object: "list",
		Data:   models,
	}, nil
}

// convertParams converts llm.CompletionParams to OpenAI request parameters.
func convertParams(params llm.CompletionParams) openai.ChatCompletionNewParams {
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(params.Messages))
	for _, msg := range params.Messages {
		messages = append(messages, convertMessage(msg))
	}

	req := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(params.Model),
		Messages: messages,
	}

	if params.Temperature != nil {
		req.Temperature = openai.Float(*params.Temperature)
	}

	if params.TopP != nil {
		req.TopP = openai.Float(*params.TopP)
	}

	if params.MaxTokens != nil {
		req.MaxCompletionTokens = openai.Int(int64(*params.MaxTokens))
	}

	if len(params.Stop) > 0 {
		req.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: params.Stop,
		}
	}

	if len(params.Tools) > 0 {
		tools := make([]openai.ChatCompletionToolParam, 0, len(params.Tools))
		for _, tool := range params.Tools {
			tools = append(tools, convertTool(tool))
		}
		req.Tools = tools
	}

	if params.ToolChoice != nil {
		req.ToolChoice = convertToolChoice(params.ToolChoice)
	}

	if params.ParallelToolCalls != nil {
		req.ParallelToolCalls = openai.Bool(*params.ParallelToolCalls)
	}

	if params.ResponseFormat != nil {
		req.ResponseFormat = convertResponseFormat(params.ResponseFormat)
	}

	if params.Seed != nil {
		req.Seed = openai.Int(int64(*params.Seed))
	}

	if params.User != "" {
		req.User = openai.String(params.User)
	}

	if params.ReasoningEffort != "" && params.ReasoningEffort != llm.ReasoningEffortNone {
		req.ReasoningEffort = shared.ReasoningEffort(params.ReasoningEffort)
	}

	if params.StreamOptions != nil && params.StreamOptions.IncludeUsage {
		req.StreamOptions = openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		}
	}

	return req
}

// convertMessage converts an llm.Message to an OpenAI message parameter.
func convertMessage(msg llm.Message) openai.ChatCompletionMessageParamUnion {
	switch msg.Role {
	case llm.RoleSystem:
		return openai.SystemMessage(msg.GetContentString())

	case llm.RoleUser:
		if msg.IsMultiModal() {
			parts := make([]openai.ChatCompletionContentPartUnionParam, 0)
			for _, part := range msg.GetContentParts() {
				if part.Type == "text" {
					parts = append(parts, openai.TextContentPart(part.Text))
				} else if part.Type == "image_url" && part.ImageURL != nil {
					parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
						URL: part.ImageURL.URL,
					}))
				}
			}
			return openai.UserMessage(parts)
		}
		return openai.UserMessage(msg.GetContentString())

	case llm.RoleAssistant:
		if len(msg.ToolCalls) > 0 {
			toolCalls := make([]openai.ChatCompletionMessageToolCallParam, 0, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallParam{
					ID: tc.ID,
					Function: openai.ChatCompletionMessageToolCallFunctionParam{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
			return openai.ChatCompletionMessageParamUnion{
				OfAssistant: &openai.ChatCompletionAssistantMessageParam{
					Content: openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: openai.String(msg.GetContentString()),
					},
					ToolCalls: toolCalls,
				},
			}
		}
		return openai.AssistantMessage(msg.GetContentString())

	case llm.RoleTool:
		return openai.ToolMessage(msg.ToolCallID, msg.GetContentString())

	default:
		return openai.UserMessage(msg.GetContentString())
	}
}

// convertTool converts an llm.Tool to an OpenAI tool parameter.
func convertTool(tool llm.Tool) openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Function: shared.FunctionDefinitionParam{
			Name:        tool.Function.Name,
			Description: openai.String(tool.Function.Description),
			Parameters:  shared.FunctionParameters(tool.Function.Parameters),
		},
	}
}

// convertToolChoice converts anyllm tool choice to OpenAI format.
func convertToolChoice(choice any) openai.ChatCompletionToolChoiceOptionUnionParam {
	switch v := choice.(type) {
	case string:
		// For string values like "auto", "none", "required", use the OfAuto field
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String(v),
		}
	case llm.ToolChoice:
		if v.Function != nil {
			return openai.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
				openai.ChatCompletionNamedToolChoiceFunctionParam{
					Name: v.Function.Name,
				},
			)
		}
	}
	return openai.ChatCompletionToolChoiceOptionUnionParam{
		OfAuto: openai.String("auto"),
	}
}

// convertResponseFormat converts anyllm response format to OpenAI format.
func convertResponseFormat(format *llm.ResponseFormat) openai.ChatCompletionNewParamsResponseFormatUnion {
	if format == nil {
		return openai.ChatCompletionNewParamsResponseFormatUnion{}
	}

	switch format.Type {
	case "json_object":
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		}
	case "json_schema":
		if format.JSONSchema != nil {
			schemaBytes, _ := json.Marshal(format.JSONSchema.Schema)
			var schema interface{}
			_ = json.Unmarshal(schemaBytes, &schema) // Ignore error: use nil on failure
			strict := format.JSONSchema.Strict != nil && *format.JSONSchema.Strict
			return openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
					JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        format.JSONSchema.Name,
						Description: openai.String(format.JSONSchema.Description),
						Schema:      schema,
						Strict:      openai.Bool(strict),
					},
				},
			}
		}
	}

	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfText: &shared.ResponseFormatTextParam{},
	}
}

// convertResponse converts an OpenAI response to anyllm format.
func convertResponse(resp *openai.ChatCompletion) *llm.ChatCompletion {
	choices := make([]llm.Choice, 0, len(resp.Choices))
	for _, choice := range resp.Choices {
		choices = append(choices, llm.Choice{
			Index:        int(choice.Index),
			Message:      convertResponseMessage(choice.Message),
			FinishReason: string(choice.FinishReason),
		})
	}

	result := &llm.ChatCompletion{
		ID:                resp.ID,
		Object:            "chat.completion",
		Created:           resp.Created,
		Model:             resp.Model,
		Choices:           choices,
		SystemFingerprint: resp.SystemFingerprint,
	}

	if resp.Usage.PromptTokens > 0 || resp.Usage.CompletionTokens > 0 {
		result.Usage = &llm.Usage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		}
		if resp.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
			result.Usage.ReasoningTokens = int(resp.Usage.CompletionTokensDetails.ReasoningTokens)
		}
	}

	return result
}

// convertResponseMessage converts an OpenAI response message to anyllm format.
func convertResponseMessage(msg openai.ChatCompletionMessage) llm.Message {
	result := llm.Message{
		Role:    string(msg.Role),
		Content: msg.Content,
	}

	if len(msg.ToolCalls) > 0 {
		result.ToolCalls = make([]llm.ToolCall, 0, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			result.ToolCalls = append(result.ToolCalls, llm.ToolCall{
				ID:   tc.ID,
				Type: string(tc.Type),
				Function: llm.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
	}

	return result
}

// convertChunk converts an OpenAI streaming chunk to anyllm format.
func convertChunk(chunk *openai.ChatCompletionChunk) llm.ChatCompletionChunk {
	choices := make([]llm.ChunkChoice, 0, len(chunk.Choices))
	for _, choice := range chunk.Choices {
		chunkChoice := llm.ChunkChoice{
			Index: int(choice.Index),
			Delta: llm.ChunkDelta{
				Role:    string(choice.Delta.Role),
				Content: choice.Delta.Content,
			},
			FinishReason: string(choice.FinishReason),
		}

		if len(choice.Delta.ToolCalls) > 0 {
			chunkChoice.Delta.ToolCalls = make([]llm.ToolCall, 0, len(choice.Delta.ToolCalls))
			for _, tc := range choice.Delta.ToolCalls {
				chunkChoice.Delta.ToolCalls = append(chunkChoice.Delta.ToolCalls, llm.ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: llm.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
		}

		choices = append(choices, chunkChoice)
	}

	result := llm.ChatCompletionChunk{
		ID:                chunk.ID,
		Object:            "chat.completion.chunk",
		Created:           chunk.Created,
		Model:             chunk.Model,
		Choices:           choices,
		SystemFingerprint: chunk.SystemFingerprint,
	}

	if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
		result.Usage = &llm.Usage{
			PromptTokens:     int(chunk.Usage.PromptTokens),
			CompletionTokens: int(chunk.Usage.CompletionTokens),
			TotalTokens:      int(chunk.Usage.TotalTokens),
		}
	}

	return result
}

// convertEmbeddingParams converts anyllm embedding params to OpenAI format.
func convertEmbeddingParams(params llm.EmbeddingParams) openai.EmbeddingNewParams {
	req := openai.EmbeddingNewParams{
		Model: openai.EmbeddingModel(params.Model),
	}

	switch v := params.Input.(type) {
	case string:
		req.Input = openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(v),
		}
	case []string:
		req.Input = openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: v,
		}
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

	return req
}

// convertEmbeddingResponse converts an OpenAI embedding response to anyllm format.
func convertEmbeddingResponse(resp *openai.CreateEmbeddingResponse) *llm.EmbeddingResponse {
	data := make([]llm.EmbeddingData, 0, len(resp.Data))
	for _, d := range resp.Data {
		embedding := make([]float64, len(d.Embedding))
		copy(embedding, d.Embedding)
		data = append(data, llm.EmbeddingData{
			Object:    "embedding",
			Embedding: embedding,
			Index:     int(d.Index),
		})
	}

	result := &llm.EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  resp.Model,
	}

	if resp.Usage.PromptTokens > 0 || resp.Usage.TotalTokens > 0 {
		result.Usage = &llm.EmbeddingUsage{
			PromptTokens: int(resp.Usage.PromptTokens),
			TotalTokens:  int(resp.Usage.TotalTokens),
		}
	}

	return result
}

// init registers the OpenAI provider.
func init() {
	llm.Register(providerName, func(opts ...llm.Option) (llm.Provider, error) {
		return New(opts...)
	})
}
