// Package openai provides an OpenAI provider implementation for any-llm.
// It also exports a base provider for other OpenAI-compatible services.
package openai

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/pagination"
	"github.com/openai/openai-go/v3/packages/respjson"
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

// Content part types.
const (
	contentTypeImageURL = "image_url"
	contentTypeText     = "text"
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
	_ providers.FileProvider       = (*CompatibleProvider)(nil)
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

	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(cfg.HTTPClient()),
	}

	if baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(baseURL))
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
	if err := validateCompletionParams(params); err != nil {
		return nil, err
	}

	req := convertParams(params)
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

		if err := validateCompletionParams(params); err != nil {
			errs <- err
			return
		}

		req := convertParams(params)
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

// UploadFile uploads a file through an OpenAI-compatible Files API.
func (p *CompatibleProvider) UploadFile(
	ctx context.Context,
	params providers.UploadFileParams,
) (*providers.File, error) {
	if err := p.requireFiles("upload file"); err != nil {
		return nil, err
	}

	req := openai.FileNewParams{
		File:    params.File,
		Purpose: openai.FilePurpose(params.Purpose),
	}
	if params.ExpiresAfter != nil {
		// DeepSeek documents bracketed multipart fields, while openai-go v3.42.0
		// serializes this generated nested struct with dotted field names.
		// https://api-docs.deepseek.com/guides/files_api/#upload-a-file
		req.SetExtraFields(map[string]any{
			"expires_after[anchor]":  "created_at",
			"expires_after[seconds]": fmt.Sprint(*params.ExpiresAfter),
		})
	}
	resp, err := p.client.Files.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertFile(resp)
}

// ListFiles lists one page from an OpenAI-compatible Files API.
func (p *CompatibleProvider) ListFiles(
	ctx context.Context,
	opts providers.ListFilesOptions,
) (*providers.FileList, error) {
	if err := p.requireFiles("list files"); err != nil {
		return nil, err
	}

	req := openai.FileListParams{}
	if opts.After != "" {
		req.After = openai.String(opts.After)
	}
	if opts.Limit != nil {
		req.Limit = openai.Int(int64(*opts.Limit))
	}
	if opts.Order != "" {
		req.Order = openai.FileListParamsOrder(opts.Order)
	}
	if opts.Purpose != "" {
		req.Purpose = openai.String(opts.Purpose)
	}
	resp, err := p.client.Files.List(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertFileList(resp)
}

// RetrieveFile returns metadata for an uploaded file.
func (p *CompatibleProvider) RetrieveFile(ctx context.Context, fileID string) (*providers.File, error) {
	if err := p.requireFiles("retrieve file"); err != nil {
		return nil, err
	}
	resp, err := p.client.Files.Get(ctx, fileID)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertFile(resp)
}

// DeleteFile deletes an uploaded file.
func (p *CompatibleProvider) DeleteFile(ctx context.Context, fileID string) (*providers.DeletedFile, error) {
	if err := p.requireFiles("delete file"); err != nil {
		return nil, err
	}
	resp, err := p.client.Files.Delete(ctx, fileID)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	return convertDeletedFile(resp)
}

func (p *CompatibleProvider) requireFiles(operation string) error {
	if p.compatibleConfig.Capabilities.Files {
		return nil
	}
	return errors.NewUnsupportedOperationError(p.Name(), operation, nil)
}

func convertFile(source *openai.FileObject) (*providers.File, error) {
	if err := requireFileFields([]fileField{
		{"id", source.JSON.ID},
		{"object", source.JSON.Object},
		{"bytes", source.JSON.Bytes},
		{"created_at", source.JSON.CreatedAt},
		{"filename", source.JSON.Filename},
		{"purpose", source.JSON.Purpose},
	}); err != nil {
		return nil, err
	}
	result := &providers.File{
		ID:        source.ID,
		Object:    string(source.Object),
		Bytes:     source.Bytes,
		CreatedAt: source.CreatedAt,
		Filename:  source.Filename,
		Purpose:   string(source.Purpose),
	}
	if source.JSON.ExpiresAt.Raw() != "" && !source.JSON.ExpiresAt.Valid() {
		return nil, fmt.Errorf("decoding file: missing or invalid expires_at")
	}
	if source.JSON.ExpiresAt.Valid() {
		result.ExpiresAt = new(source.ExpiresAt)
	}
	return result, nil
}

func convertFileList(source *pagination.CursorPage[openai.FileObject]) (*providers.FileList, error) {
	var metadata struct {
		Object  string `json:"object"`
		FirstID string `json:"first_id"`
		LastID  string `json:"last_id"`
	}
	if err := json.Unmarshal([]byte(source.RawJSON()), &metadata); err != nil {
		return nil, fmt.Errorf("decoding file list metadata: %w", err)
	}

	files := make([]providers.File, 0, len(source.Data))
	for i := range source.Data {
		file, err := convertFile(&source.Data[i])
		if err != nil {
			return nil, err
		}
		files = append(files, *file)
	}
	return &providers.FileList{
		Object: metadata.Object, Data: files, FirstID: metadata.FirstID,
		LastID: metadata.LastID, HasMore: source.HasMore,
	}, nil
}

type fileField struct {
	name  string
	value respjson.Field
}

func requireFileFields(fields []fileField) error {
	for _, field := range fields {
		if !field.value.Valid() {
			return fmt.Errorf("decoding file: missing or invalid %s", field.name)
		}
	}
	return nil
}

func convertDeletedFile(source *openai.FileDeleted) (*providers.DeletedFile, error) {
	if err := requireFileFields([]fileField{
		{"id", source.JSON.ID},
		{"object", source.JSON.Object},
		{"deleted", source.JSON.Deleted},
	}); err != nil {
		return nil, err
	}
	return &providers.DeletedFile{ID: source.ID, Object: string(source.Object), Deleted: source.Deleted}, nil
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

// convertAssistantMessage converts an assistant message to OpenAI format.
func convertAssistantMessage(msg providers.Message) openai.ChatCompletionMessageParamUnion {
	if len(msg.ToolCalls) > 0 {
		toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
				OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
					ID: tc.ID,
					Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				},
			})
		}
		return openai.ChatCompletionMessageParamUnion{
			OfAssistant: &openai.ChatCompletionAssistantMessageParam{
				Content: openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(msg.ContentString()),
				},
				ToolCalls: toolCalls,
			},
		}
	}
	return openai.AssistantMessage(msg.ContentString())
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

// convertMessage converts a single message to OpenAI format.
func convertMessage(msg providers.Message) (openai.ChatCompletionMessageParamUnion, error) {
	switch msg.Role {
	case providers.RoleAssistant:
		return convertAssistantMessage(msg), nil
	case providers.RoleSystem:
		return openai.SystemMessage(msg.ContentString()), nil
	case providers.RoleTool:
		return openai.ToolMessage(msg.ContentString(), msg.ToolCallID), nil
	case providers.RoleUser:
		return convertUserMessage(msg), nil
	default:
		return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("unknown message role: %q", msg.Role)
	}
}

// convertMessages converts provider messages to OpenAI format.
func convertMessages(messages []providers.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))
	for _, msg := range messages {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}
		result = append(result, converted)
	}
	return result, nil
}

// convertParams converts providers.CompletionParams to OpenAI request parameters.
func convertParams(params providers.CompletionParams) openai.ChatCompletionNewParams {
	messages, _ := convertMessages(params.Messages) // Error already checked in validateCompletionParams

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
		function := openai.FunctionDefinitionParam{
			Name:        tool.Function.Name,
			Description: openai.String(tool.Function.Description),
			Parameters:  openai.FunctionParameters(tool.Function.Parameters),
		}
		if tool.Function.Strict != nil {
			function.Strict = openai.Bool(*tool.Function.Strict)
		}
		result = append(result, openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: function,
			},
		})
	}
	return result
}

// convertUserMessage converts a user message to OpenAI format.
func convertUserMessage(msg providers.Message) openai.ChatCompletionMessageParamUnion {
	if msg.IsMultiModal() {
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
	return openai.UserMessage(msg.ContentString())
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
	if params.Model == "" {
		return errors.NewInvalidRequestError("", fmt.Errorf("model is required"))
	}
	if len(params.Messages) == 0 {
		return errors.NewInvalidRequestError("", fmt.Errorf("at least one message is required"))
	}

	// Validate message roles.
	for _, msg := range params.Messages {
		if _, err := convertMessage(msg); err != nil {
			return errors.NewInvalidRequestError("", err)
		}
	}

	return nil
}
