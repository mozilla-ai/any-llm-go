// Package openai provides an OpenAI provider implementation for any-llm.
// It also exports a base provider for other OpenAI-compatible services.
package openai

import (
	"context"
	stderrors "errors"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// OpenAI API error codes.
const (
	apiCodeContentFilter         = "content_filter"
	apiCodeContentPolicyViolated = "content_policy_violation"
	apiCodeContextLengthExceeded = "context_length_exceeded"
	apiCodeInvalidAPIKey         = "invalid_api_key"
	apiCodeModelNotFound         = "model_not_found"
	apiCodeRateLimitExceeded     = "rate_limit_exceeded"
)

// Object type constants.
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectEmbedding           = "embedding"
	objectList                = "list"
	objectModel               = "model"
)

// Response format types.
const (
	responseFormatJSONObject = "json_object"
	responseFormatJSONSchema = "json_schema"
)

// CompatibleConfig contains the configuration for an OpenAI-compatible provider.
// Fields are ordered alphabetically.
type CompatibleConfig struct {
	// APIKeyEnvVar is the environment variable for the API key.
	APIKeyEnvVar string

	// BaseURLEnvVar is the environment variable for the base URL.
	BaseURLEnvVar string

	// Capabilities describes what the provider supports.
	Capabilities providers.Capabilities

	// ClientOptions replaces CompatibleProvider's API key, HTTP client, and base
	// URL options when set. Providers with non-OpenAI authentication can supply
	// configured SDK options while sharing request, stream, and error handling.
	ClientOptions []option.RequestOption

	// DefaultAPIKey is used when RequireAPIKey is false (e.g., for local servers).
	DefaultAPIKey string

	// DefaultBaseURL is the default API base URL.
	DefaultBaseURL string

	// Name is the provider name used in error messages.
	Name string

	// ChatCompletionRequestTransform is an optional function that modifies the chat
	// completion request after convertParams() builds it and before it is serialized
	// to the wire. Providers that are not fully OpenAI-compatible use this to adjust
	// wire-level fields (e.g. swapping max_completion_tokens back to max_tokens).
	// The pointer refers to a locally-constructed value owned by the caller; the
	// function must not retain it beyond the call. Nil means no transformation.
	ChatCompletionRequestTransform func(*openai.ChatCompletionNewParams)

	// OpenAIMessageSchema enables OpenAI-specific message roles and content fields.
	// Compatible providers leave this false until their own schemas have been
	// verified because OpenAI compatibility does not imply message-schema parity.
	// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
	OpenAIMessageSchema bool

	// RequireAPIKey indicates whether an API key is required.
	RequireAPIKey bool

	// RequireBaseURL indicates whether a base URL must be resolvable from
	// WithBaseURL, BaseURLEnvVar, or DefaultBaseURL. When true, NewCompatible
	// returns an error if none of those yield a value. Used by providers that
	// have no sensible default endpoint (e.g. gateway).
	RequireBaseURL bool
}

// Ensure CompatibleProvider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*CompatibleProvider)(nil)
	_ providers.EmbeddingProvider  = (*CompatibleProvider)(nil)
	_ providers.ErrorConverter     = (*CompatibleProvider)(nil)
	_ providers.ModelLister        = (*CompatibleProvider)(nil)
	_ providers.Provider           = (*CompatibleProvider)(nil)
)

// CompatibleProvider implements the providers.Provider interface for OpenAI-compatible APIs.
// It can be embedded by other providers that use OpenAI-compatible endpoints.
type CompatibleProvider struct {
	compatibleConfig CompatibleConfig
	client           openai.Client
}

// NewCompatible creates a new OpenAI-compatible provider.
func NewCompatible(compatCfg CompatibleConfig, opts ...config.Option) (*CompatibleProvider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	if validErr := validateCompatibleConfig(compatCfg); validErr != nil {
		return nil, validErr
	}

	baseURL, err := cfg.ResolveBaseURL(compatCfg.BaseURLEnvVar, compatCfg.DefaultBaseURL)
	if err != nil {
		return nil, err
	}

	if baseURL == "" && compatCfg.RequireBaseURL {
		if compatCfg.BaseURLEnvVar == "" {
			return nil, fmt.Errorf(
				"%s base URL is required (set via WithBaseURL option)",
				compatCfg.Name,
			)
		}

		return nil, fmt.Errorf(
			"%s base URL is required (set via WithBaseURL option or %q env var)",
			compatCfg.Name,
			compatCfg.BaseURLEnvVar,
		)
	}

	apiKey := resolveAPIKey(cfg, compatCfg)

	if apiKey == "" && compatCfg.RequireAPIKey {
		return nil, errors.NewMissingAPIKeyError(compatCfg.Name, compatCfg.APIKeyEnvVar)
	}
	if apiKey == "" {
		apiKey = compatCfg.DefaultAPIKey
	}

	clientOpts := compatCfg.ClientOptions
	if len(clientOpts) == 0 {
		clientOpts = []option.RequestOption{
			option.WithAPIKey(apiKey),
			option.WithHTTPClient(cfg.HTTPClient()),
		}

		if baseURL != "" {
			clientOpts = append(clientOpts, option.WithBaseURL(baseURL))
		}
	}

	return &CompatibleProvider{
		compatibleConfig: compatCfg,
		client:           openai.NewClient(clientOpts...),
	}, nil
}

// Capabilities returns the provider's capabilities.
func (p *CompatibleProvider) Capabilities() providers.Capabilities {
	return p.compatibleConfig.Capabilities
}

// Completion performs a chat completion request.
func (p *CompatibleProvider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	converter := p.messageConverter()

	err := validateCompletionParamsWith(params, converter)
	if err != nil {
		return nil, err
	}

	req := convertParamsWith(params, converter)
	if p.compatibleConfig.ChatCompletionRequestTransform != nil {
		p.compatibleConfig.ChatCompletionRequestTransform(&req)
	}

	resp, err := p.client.Chat.Completions.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertResponse(resp), nil
}

// CompletionStream performs a streaming chat completion request.
func (p *CompatibleProvider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		converter := p.messageConverter()

		err := validateCompletionParamsWith(params, converter)
		if err != nil {
			errs <- err
			return
		}

		req := convertParamsWith(params, converter)
		if p.compatibleConfig.ChatCompletionRequestTransform != nil {
			p.compatibleConfig.ChatCompletionRequestTransform(&req)
		}
		stream := p.client.Chat.Completions.NewStreaming(ctx, req)

		for stream.Next() {
			chunk := stream.Current()
			select {
			case chunks <- convertChunk(&chunk):
			case <-ctx.Done():
				// Caller cancelled mid-stream; surface ctx.Err() so the
				// consumer can tell a cancelled stream apart from one
				// that completed cleanly, rather than seeing a bare
				// nil on the error channel.
				errs <- ctx.Err()
				return
			}
		}

		if err := stream.Err(); err != nil {
			errs <- p.ConvertError(err)
		}
	}()

	return chunks, errs
}

// ConvertError converts OpenAI-compatible errors to unified error types.
// Implements providers.ErrorConverter.
func (p *CompatibleProvider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	name := p.compatibleConfig.Name

	// Check for OpenAI API error type.
	var apiErr *openai.Error
	if stderrors.As(err, &apiErr) {
		return convertAPIError(name, apiErr, err)
	}

	// Network-level errors are wrapped as provider errors.
	// Note: We check for "connection refused" string as a fallback since
	// Go's net package doesn't expose typed errors for all network conditions.
	return errors.NewProviderError(name, err)
}

// Embedding generates embeddings for the given input.
func (p *CompatibleProvider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	req := convertEmbeddingParams(params)

	resp, err := p.client.Embeddings.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertEmbeddingResponse(resp), nil
}

// ListModels returns a list of available models.
func (p *CompatibleProvider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.client.Models.List(ctx)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	models := make([]providers.Model, 0, len(resp.Data))
	for _, model := range resp.Data {
		models = append(models, providers.Model{
			ID:      model.ID,
			Object:  objectModel,
			Created: model.Created,
			OwnedBy: string(model.OwnedBy),
		})
	}

	return &providers.ModelsResponse{
		Object: objectList,
		Data:   models,
	}, nil
}

// Name returns the provider name.
func (p *CompatibleProvider) Name() string {
	return p.compatibleConfig.Name
}

// convertAPIError converts an OpenAI API error to a unified error type.
func convertAPIError(name string, apiErr *openai.Error, originalErr error) error {
	switch apiErr.StatusCode {
	case 400:
		if apiErr.Code == apiCodeContextLengthExceeded {
			return errors.NewContextLengthError(name, originalErr)
		}
		if apiErr.Code == apiCodeContentFilter || apiErr.Code == apiCodeContentPolicyViolated {
			return errors.NewContentFilterError(name, originalErr)
		}
		return errors.NewInvalidRequestError(name, originalErr)
	case 401:
		return errors.NewAuthenticationError(name, originalErr)
	case 404:
		return errors.NewModelNotFoundError(name, originalErr)
	case 429:
		return errors.NewRateLimitError(name, originalErr)
	}

	// Check error code for additional classification.
	switch apiErr.Code {
	case apiCodeInvalidAPIKey:
		return errors.NewAuthenticationError(name, originalErr)
	case apiCodeModelNotFound:
		return errors.NewModelNotFoundError(name, originalErr)
	case apiCodeRateLimitExceeded:
		return errors.NewRateLimitError(name, originalErr)
	}

	return errors.NewProviderError(name, originalErr)
}

// convertChunk converts an OpenAI streaming chunk to provider format.
func convertChunk(chunk *openai.ChatCompletionChunk) providers.ChatCompletionChunk {
	choices := make([]providers.ChunkChoice, 0, len(chunk.Choices))
	for _, choice := range chunk.Choices {
		chunkChoice := providers.ChunkChoice{
			Index: int(choice.Index),
			Delta: providers.ChunkDelta{
				Role:    string(choice.Delta.Role),
				Content: choice.Delta.Content,
			},
			FinishReason: string(choice.FinishReason),
		}

		if len(choice.Delta.ToolCalls) > 0 {
			chunkChoice.Delta.ToolCalls = make([]providers.ToolCall, 0, len(choice.Delta.ToolCalls))
			for _, tc := range choice.Delta.ToolCalls {
				chunkChoice.Delta.ToolCalls = append(chunkChoice.Delta.ToolCalls, providers.ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: providers.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
		}

		choices = append(choices, chunkChoice)
	}

	// Preserve the existing normalized field even though the v3 SDK marks the
	// documented response field as deprecated.
	// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
	result := providers.ChatCompletionChunk{
		ID:                chunk.ID,
		Object:            objectChatCompletionChunk,
		Created:           chunk.Created,
		Model:             chunk.Model,
		Choices:           choices,
		SystemFingerprint: chunk.SystemFingerprint, //nolint:staticcheck
	}

	if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
		result.Usage = &providers.Usage{
			PromptTokens:     int(chunk.Usage.PromptTokens),
			CompletionTokens: int(chunk.Usage.CompletionTokens),
			TotalTokens:      int(chunk.Usage.TotalTokens),
		}
	}

	return result
}

// convertEmbeddingParams converts provider embedding params to OpenAI format.
func convertEmbeddingParams(params providers.EmbeddingParams) openai.EmbeddingNewParams {
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
	default:
		// For unsupported types, convert to string representation.
		req.Input = openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String(fmt.Sprintf("%v", params.Input)),
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

// convertEmbeddingResponse converts an OpenAI embedding response to provider format.
func convertEmbeddingResponse(resp *openai.CreateEmbeddingResponse) *providers.EmbeddingResponse {
	data := make([]providers.EmbeddingData, 0, len(resp.Data))
	for _, d := range resp.Data {
		embedding := make([]float64, len(d.Embedding))
		copy(embedding, d.Embedding)
		data = append(data, providers.EmbeddingData{
			Object:    objectEmbedding,
			Embedding: embedding,
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

// convertParams converts providers.CompletionParams to OpenAI request parameters.
func convertParams(params providers.CompletionParams) openai.ChatCompletionNewParams {
	return convertParamsWith(params, convertOpenAIMessage)
}

func convertParamsWith(
	params providers.CompletionParams,
	converter chatCompletionMessageConverter,
) openai.ChatCompletionNewParams {
	messages, _ := convertMessagesWith(params.Messages, converter) // Error already checked during validation.

	req := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(params.Model),
		Messages: messages,
	}

	if params.Temperature != nil {
		req.Temperature = openai.Float(*params.Temperature)
	}

	if params.TopP != nil {
		req.TopP = openai.Float(*params.TopP)
	}

	// OpenAI documents max_completion_tokens as the replacement for the
	// deprecated max_tokens field. Keep the binding's existing MaxTokens
	// abstraction on the current wire field.
	// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
	if params.MaxTokens != nil {
		req.MaxCompletionTokens = openai.Int(int64(*params.MaxTokens))
	}

	if len(params.Stop) > 0 {
		req.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: params.Stop,
		}
	}

	if len(params.Tools) > 0 {
		req.Tools = convertTools(params.Tools)
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

	// auto is the binding's omission sentinel. OpenAI documents none as an
	// explicit value, so it must reach the wire unchanged.
	// https://developers.openai.com/api/docs/guides/latest-model
	if params.ReasoningEffort != "" && params.ReasoningEffort != providers.ReasoningEffortAuto {
		req.ReasoningEffort = shared.ReasoningEffort(params.ReasoningEffort)
	}

	if params.StreamOptions != nil && params.StreamOptions.IncludeUsage {
		req.StreamOptions = openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		}
	}

	return req
}

// convertResponse converts an OpenAI response to provider format.
func convertResponse(resp *openai.ChatCompletion) *providers.ChatCompletion {
	choices := make([]providers.Choice, 0, len(resp.Choices))
	for _, choice := range resp.Choices {
		choices = append(choices, providers.Choice{
			Index:        int(choice.Index),
			Message:      convertResponseMessage(choice.Message),
			FinishReason: string(choice.FinishReason),
		})
	}

	// Preserve the existing normalized field even though the v3 SDK marks the
	// documented response field as deprecated.
	// https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
	result := &providers.ChatCompletion{
		ID:                resp.ID,
		Object:            objectChatCompletion,
		Created:           resp.Created,
		Model:             resp.Model,
		Choices:           choices,
		SystemFingerprint: resp.SystemFingerprint, //nolint:staticcheck
	}

	if resp.Usage.PromptTokens > 0 || resp.Usage.CompletionTokens > 0 {
		result.Usage = &providers.Usage{
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

// convertResponseFormat converts provider response format to OpenAI format.
func convertResponseFormat(format *providers.ResponseFormat) openai.ChatCompletionNewParamsResponseFormatUnion {
	if format == nil {
		return openai.ChatCompletionNewParamsResponseFormatUnion{}
	}

	switch format.Type {
	case responseFormatJSONObject:
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &openai.ResponseFormatJSONObjectParam{},
		}
	case responseFormatJSONSchema:
		if format.JSONSchema != nil {
			strict := format.JSONSchema.Strict != nil && *format.JSONSchema.Strict
			return openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &openai.ResponseFormatJSONSchemaParam{
					JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
						Name:        format.JSONSchema.Name,
						Description: openai.String(format.JSONSchema.Description),
						Schema:      format.JSONSchema.Schema,
						Strict:      openai.Bool(strict),
					},
				},
			}
		}
	}

	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfText: &openai.ResponseFormatTextParam{},
	}
}

// convertResponseMessage converts an OpenAI response message to provider format.
func convertResponseMessage(msg openai.ChatCompletionMessage) providers.Message {
	result := providers.Message{
		Role:    string(msg.Role),
		Content: msg.Content,
	}

	if len(msg.ToolCalls) > 0 {
		result.ToolCalls = make([]providers.ToolCall, 0, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			result.ToolCalls = append(result.ToolCalls, providers.ToolCall{
				ID:   tc.ID,
				Type: string(tc.Type),
				Function: providers.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
	}

	return result
}

// convertToolChoice converts provider tool choice to OpenAI format.
func convertToolChoice(choice any) openai.ChatCompletionToolChoiceOptionUnionParam {
	switch v := choice.(type) {
	case string:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String(v),
		}
	case providers.ToolChoice:
		if v.Function != nil {
			return openai.ToolChoiceOptionFunctionToolChoice(
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

// convertTools converts provider tools to OpenAI format.
func convertTools(tools []providers.Tool) []openai.ChatCompletionToolUnionParam {
	result := make([]openai.ChatCompletionToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		result = append(result, openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: openai.FunctionDefinitionParam{
					Name:        tool.Function.Name,
					Description: openai.String(tool.Function.Description),
					Parameters:  openai.FunctionParameters(tool.Function.Parameters),
				},
			},
		})
	}
	return result
}

// resolveAPIKey resolves the API key from config or environment.
func resolveAPIKey(cfg *config.Config, compatCfg CompatibleConfig) string {
	if compatCfg.APIKeyEnvVar != "" {
		return cfg.ResolveAPIKey(compatCfg.APIKeyEnvVar)
	}
	return cfg.APIKey
}

// validateCompatibleConfig validates the compatible provider configuration.
func validateCompatibleConfig(cfg CompatibleConfig) error {
	if cfg.Name == "" {
		return fmt.Errorf("provider name is required")
	}
	return nil
}

// validateCompletionParams validates completion parameters.
func validateCompletionParams(params providers.CompletionParams) error {
	return validateCompletionParamsWith(params, convertOpenAIMessage)
}

func validateCompletionParamsWith(
	params providers.CompletionParams,
	converter chatCompletionMessageConverter,
) error {
	if params.Model == "" {
		return errors.NewInvalidRequestError("", fmt.Errorf("model is required"))
	}
	if len(params.Messages) == 0 {
		return errors.NewInvalidRequestError("", fmt.Errorf("at least one message is required"))
	}

	// Validate message roles.
	for _, msg := range params.Messages {
		_, err := converter(msg)
		if err != nil {
			return errors.NewInvalidRequestError("", err)
		}
	}

	return nil
}
