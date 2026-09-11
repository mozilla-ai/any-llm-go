// Package ollama provides an Ollama provider implementation for any-llm.
package ollama

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"maps"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"time"

	"github.com/ollama/ollama/api"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	defaultBaseURL = "http://localhost:11434"
	defaultNumCtx  = 32000
	envBaseURL     = "OLLAMA_HOST"
	providerName   = "ollama"
)

// Ollama done reasons.
const (
	doneReasonLength = "length"
	doneReasonStop   = "stop"
)

// Ollama option keys.
const (
	optionNumCtx      = "num_ctx"
	optionNumPredict  = "num_predict"
	optionSeed        = "seed"
	optionStop        = "stop"
	optionTemperature = "temperature"
	optionTopP        = "top_p"
)

// JSON schema keys and types.
const (
	schemaKeyDescription = "description"
	schemaKeyProperties  = "properties"
	schemaKeyRequired    = "required"
	schemaKeyType        = "type"
	schemaTypeObject     = "object"
)

// Tool and response format constants.
const (
	ollamaFormatJSON     = "json"
	responseFormatJSON   = "json_object"
	responseFormatSchema = "json_schema"
	toolCallIDFormat     = "call_%d"
	toolTypeFunction     = "function"
)

// Object type constants.
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectEmbedding           = "embedding"
	objectList                = "list"
	objectModel               = "model"
)

// Content part constants.
const (
	contentTypeImageURL = "image_url"
	contentTypeText     = "text"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Ollama.
type Provider struct {
	client *api.Client
	config *config.Config
}

// streamState tracks accumulated state during streaming.
type streamState struct {
	id      string
	model   string
	created int64
	done    bool
}

// New creates a new Ollama provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = cfg.ResolveEnv(envBaseURL)
	}
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}

	client := api.NewClient(parsedURL, cfg.HTTPClient())

	return &Provider{
		client: client,
		config: cfg,
	}, nil
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       false,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           true,
		ListModels:          true,
	}
}

// Completion performs a chat completion request.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	req, err := p.convertParams(params)
	if err != nil {
		return nil, err
	}

	// Disable streaming for non-stream requests.
	stream := false
	req.Stream = &stream

	var response api.ChatResponse

	done := false

	err = p.client.Chat(ctx, req, func(resp api.ChatResponse) error {
		response = resp
		done = resp.Done
		return nil
	})
	if err != nil {
		return nil, p.ConvertError(err)
	}

	if !done {
		return nil, errors.NewProviderError(providerName, stderrors.New("response ended before the terminal event"))
	}

	return convertResponse(&response), nil
}

// CompletionStream performs a streaming chat completion request.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		req, err := p.convertParams(params)
		if err != nil {
			errs <- err

			return
		}

		state := newStreamState()

		err = p.client.Chat(ctx, req, func(resp api.ChatResponse) error {
			chunk := state.handleChunk(&resp)
			select {
			case chunks <- chunk:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		if err != nil {
			errs <- p.ConvertError(err)

			return
		}

		if !state.done {
			errs <- errors.NewProviderError(providerName, stderrors.New("stream ended before the terminal event"))
		}
	}()

	return chunks, errs
}

// ConvertError converts Ollama errors to unified error types.
// Implements providers.ErrorConverter.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	// Check for authorization error (401).
	var authErr api.AuthorizationError
	if stderrors.As(err, &authErr) {
		return errors.NewAuthenticationError(providerName, err)
	}

	// Check for HTTP status errors.
	var statusErr api.StatusError
	if stderrors.As(err, &statusErr) {
		switch statusErr.StatusCode {
		case http.StatusUnauthorized:
			return errors.NewAuthenticationError(providerName, err)
		case http.StatusNotFound:
			return errors.NewModelNotFoundError(providerName, err)
		case http.StatusTooManyRequests:
			return errors.NewRateLimitError(providerName, err)
		case http.StatusBadRequest:
			if strings.Contains(statusErr.ErrorMessage, "context") {
				return errors.NewContextLengthError(providerName, err)
			}
			return errors.NewInvalidRequestError(providerName, err)
		}
	}

	// Network-level errors (connection refused, etc.) - string check acceptable here.
	if strings.Contains(err.Error(), "connection refused") {
		return errors.NewProviderError(providerName, fmt.Errorf("ollama server not running: %w", err))
	}

	return errors.NewProviderError(providerName, err)
}

// Embedding generates embeddings for the given input.
func (p *Provider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	req := &api.EmbedRequest{
		Model: params.Model,
		Input: params.Input,
	}

	resp, err := p.client.Embed(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertEmbeddingResponse(resp, params.Model), nil
}

// ListModels returns a list of available models.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.client.List(ctx)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertModelsResponse(resp), nil
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// convertParams converts providers.CompletionParams to Ollama ChatRequest.
func (p *Provider) convertParams(params providers.CompletionParams) (*api.ChatRequest, error) {
	if unsupported := unsupportedCompletionParam(params); unsupported != "" {
		return nil, errors.NewUnsupportedParamError(providerName, unsupported)
	}

	messages, err := convertMessages(params.Messages)
	if err != nil {
		return nil, err
	}

	tools, err := convertTools(params.Tools)
	if err != nil {
		return nil, err
	}

	format, err := convertResponseFormat(params.ResponseFormat)
	if err != nil {
		return nil, err
	}
	req := &api.ChatRequest{
		Model:    params.Model,
		Messages: messages,
		Options:  convertOptions(params),
		Tools:    tools,
		Format:   format,
	}
	if err := applyReasoningEffort(req, params.ReasoningEffort); err != nil {
		return nil, err
	}

	return req, nil
}

func convertOptions(params providers.CompletionParams) map[string]any {
	// Keep requests independent of Ollama server and model-specific context defaults.
	options := map[string]any{optionNumCtx: defaultNumCtx}
	if params.Temperature != nil {
		options[optionTemperature] = *params.Temperature
	}

	if params.TopP != nil {
		options[optionTopP] = *params.TopP
	}

	if len(params.Stop) > 0 {
		options[optionStop] = params.Stop
	}

	if params.MaxTokens != nil {
		options[optionNumPredict] = *params.MaxTokens
	}

	if params.Seed != nil {
		options[optionSeed] = *params.Seed
	}

	return options
}

func applyReasoningEffort(req *api.ChatRequest, effort providers.ReasoningEffort) error {
	// https://docs.ollama.com/api/chat defines think as a boolean or one of
	// low, medium, high, and max. Preserve "none" as false on that wire type.
	switch effort {
	case "", providers.ReasoningEffortAuto:
		return nil
	case providers.ReasoningEffortNone:
		req.Think = &api.ThinkValue{Value: false}
	case providers.ReasoningEffortLow,
		providers.ReasoningEffortMedium,
		providers.ReasoningEffortHigh,
		providers.ReasoningEffortMax:
		req.Think = &api.ThinkValue{Value: string(effort)}
	default:
		return errors.NewUnsupportedParamError(providerName, "reasoning_effort")
	}

	return nil
}

func unsupportedCompletionParam(params providers.CompletionParams) string {
	switch {
	case params.StreamOptions != nil:
		return "stream_options"
	case params.ToolChoice != nil:
		return "tool_choice"
	case params.ParallelToolCalls != nil:
		return "parallel_tool_calls"
	case params.User != "":
		return "user"
	case len(params.Extra) > 0:
		return "extra"
	default:
		return ""
	}
}

// newStreamState creates a new stream state.
func newStreamState() *streamState {
	return &streamState{
		id:      generateID(),
		created: time.Now().Unix(),
	}
}

// chunk creates a ChatCompletionChunk with common fields populated.
func (s *streamState) chunk() providers.ChatCompletionChunk {
	return providers.ChatCompletionChunk{
		ID:      s.id,
		Object:  objectChatCompletionChunk,
		Created: s.created,
		Model:   s.model,
		Choices: []providers.ChunkChoice{{Index: 0}},
	}
}

// handleChunk processes a streaming response and returns a chunk.
func (s *streamState) handleChunk(resp *api.ChatResponse) providers.ChatCompletionChunk {
	s.updateMetadata(resp)

	if resp.Done {
		s.done = true
	}

	chunk := s.chunk()
	chunk.Choices[0].Delta = s.buildDelta(resp)

	if resp.Done {
		s.handleDone(resp, &chunk)
	}

	return chunk
}

// updateMetadata updates stream state metadata from response.
func (s *streamState) updateMetadata(resp *api.ChatResponse) {
	if s.model == "" {
		s.model = resp.Model
	}
	if resp.CreatedAt.Unix() > 0 {
		s.created = resp.CreatedAt.Unix()
	}
}

// buildDelta constructs the delta content from a response.
func (s *streamState) buildDelta(resp *api.ChatResponse) providers.ChunkDelta {
	delta := providers.ChunkDelta{}

	// Handle content.
	if resp.Message.Content != "" {
		delta.Content = resp.Message.Content
	}

	// Handle thinking/reasoning.
	if resp.Message.Thinking != "" {
		delta.Reasoning = &providers.Reasoning{Content: resp.Message.Thinking}
	}

	// Handle tool calls.
	if len(resp.Message.ToolCalls) > 0 {
		delta.ToolCalls = convertToolCalls(resp.Message.ToolCalls)
	}

	return delta
}

// handleDone processes the final chunk when streaming is complete.
func (s *streamState) handleDone(resp *api.ChatResponse, chunk *providers.ChatCompletionChunk) {
	finishReason := providers.FinishReasonToolCalls
	if len(resp.Message.ToolCalls) == 0 {
		finishReason = convertDoneReason(resp.DoneReason)
	}

	chunk.Choices[0].FinishReason = finishReason
	chunk.Usage = &providers.Usage{
		PromptTokens:     resp.PromptEvalCount,
		CompletionTokens: resp.EvalCount,
		TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
	}
}

// convertDoneReason converts Ollama done reason to OpenAI finish reason.
func convertDoneReason(reason string) string {
	switch reason {
	case doneReasonLength:
		return providers.FinishReasonLength
	case "", doneReasonStop:
		return providers.FinishReasonStop
	default:
		return reason
	}
}

// convertEmbeddingResponse converts an Ollama embedding response to provider format.
func convertEmbeddingResponse(resp *api.EmbedResponse, model string) *providers.EmbeddingResponse {
	data := make([]providers.EmbeddingData, 0, len(resp.Embeddings))

	for i, embedding := range resp.Embeddings {
		// Convert []float32 to []float64.
		floats := make([]float64, len(embedding))
		for j, f := range embedding {
			floats[j] = float64(f)
		}

		data = append(data, providers.EmbeddingData{
			Object:    objectEmbedding,
			Embedding: floats,
			Index:     i,
		})
	}

	return &providers.EmbeddingResponse{
		Object: objectList,
		Data:   data,
		Model:  model,
		Usage: &providers.EmbeddingUsage{
			PromptTokens: resp.PromptEvalCount,
			TotalTokens:  resp.PromptEvalCount,
		},
	}
}

// convertMessage converts a single message to Ollama's documented wire model.
func convertMessage(msg providers.Message) (*api.Message, error) {
	if err := validateMessageMetadata(msg); err != nil {
		return nil, err
	}

	content, images, err := convertMessageContent(msg)
	if err != nil {
		return nil, err
	}

	toolCalls, err := convertRequestToolCalls(msg.ToolCalls)
	if err != nil {
		return nil, err
	}

	converted := &api.Message{
		Role:      msg.Role,
		Content:   content,
		Images:    images,
		ToolCalls: toolCalls,
	}
	if msg.Role == providers.RoleTool {
		converted.ToolName = msg.Name
	}

	if msg.Reasoning != nil {
		converted.Thinking = msg.Reasoning.Content
	}

	return converted, nil
}

func validateMessageMetadata(msg providers.Message) error {
	switch msg.Role {
	case providers.RoleSystem, providers.RoleUser, providers.RoleAssistant, providers.RoleTool:
	default:
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("unsupported message role %q", msg.Role))
	}

	if msg.Role != providers.RoleTool && (msg.Name != "" || msg.ToolCallID != "") {
		return errors.NewUnsupportedParamError(providerName, "messages.name/tool_call_id")
	}

	if msg.Role != providers.RoleAssistant && len(msg.ToolCalls) > 0 {
		return errors.NewUnsupportedParamError(providerName, "messages.tool_calls")
	}

	if msg.Role != providers.RoleAssistant && msg.Reasoning != nil {
		return errors.NewUnsupportedParamError(providerName, "messages.reasoning")
	}

	return nil
}

func convertRequestToolCalls(toolCalls []providers.ToolCall) ([]api.ToolCall, error) {
	if len(toolCalls) == 0 {
		return nil, nil
	}

	converted := make([]api.ToolCall, 0, len(toolCalls))
	for _, toolCall := range toolCalls {
		if toolCall.Type != toolTypeFunction {
			return nil, errors.NewUnsupportedParamError(providerName, "messages.tool_calls.type")
		}

		var arguments api.ToolCallFunctionArguments
		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); err != nil {
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("tool arguments must be a JSON object: %w", err),
			)
		}

		converted = append(converted, api.ToolCall{
			Function: api.ToolCallFunction{
				Name:      toolCall.Function.Name,
				Arguments: arguments,
			},
		})
	}

	return converted, nil
}

// convertMessages converts provider messages to Ollama format.
func convertMessages(messages []providers.Message) ([]api.Message, error) {
	result := make([]api.Message, 0, len(messages))
	toolNames := make(map[string]string)

	for _, msg := range messages {
		converted, err := convertMessage(msg)
		if err != nil {
			return nil, err
		}

		for _, toolCall := range msg.ToolCalls {
			if toolCall.ID != "" {
				toolNames[toolCall.ID] = toolCall.Function.Name
			}
		}
		// https://docs.ollama.com/capabilities/tool-calling identifies tool
		// results by tool_name. Resolve the normalized call ID before encoding.
		if msg.Role == providers.RoleTool && converted.ToolName == "" {
			converted.ToolName = toolNames[msg.ToolCallID]
			if converted.ToolName == "" {
				return nil, errors.NewInvalidRequestError(
					providerName,
					stderrors.New("tool result requires a name or a matching tool call ID"),
				)
			}
		}

		result = append(result, *converted)
	}

	return result, nil
}

func convertMessageContent(msg providers.Message) (string, []api.ImageData, error) {
	if content, ok := msg.Content.(string); ok {
		return content, nil, nil
	}

	if msg.Content == nil {
		return "", nil, nil
	}

	switch msg.Content.(type) {
	case []providers.ContentPart, []any:
	default:
		return "", nil, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("unsupported message content type %T", msg.Content),
		)
	}

	parts := msg.ContentParts()
	if raw, ok := msg.Content.([]any); ok && len(parts) != len(raw) {
		return "", nil, errors.NewInvalidRequestError(
			providerName,
			stderrors.New("message contains an invalid content part"),
		)
	}

	var (
		content strings.Builder
		images  []api.ImageData
	)

	for _, part := range parts {
		text, image, err := convertContentPart(part)
		if err != nil {
			return "", nil, err
		}

		content.WriteString(text)

		if image != nil {
			images = append(images, image)
		}
	}

	return content.String(), images, nil
}

func convertContentPart(part providers.ContentPart) (string, api.ImageData, error) {
	switch part.Type {
	case contentTypeText:
		if part.ImageURL != nil {
			return "", nil, errors.NewInvalidRequestError(
				providerName,
				stderrors.New("text content cannot include image_url"),
			)
		}

		return part.Text, nil, nil
	case contentTypeImageURL:
		if part.Text != "" || part.ImageURL == nil {
			return "", nil, errors.NewInvalidRequestError(
				providerName,
				stderrors.New("image content requires image_url only"),
			)
		}

		decoded, err := decodeImageDataURL(part.ImageURL.URL)
		if err != nil {
			return "", nil, err
		}

		return "", api.ImageData(decoded), nil
	default:
		return "", nil, errors.NewUnsupportedParamError(providerName, "messages.content.type")
	}
}

func decodeImageDataURL(dataURL string) ([]byte, error) {
	metadata, encoded, ok := strings.Cut(dataURL, ",")
	if !ok || !strings.HasPrefix(metadata, "data:") || !strings.HasSuffix(metadata, ";base64") {
		return nil, errors.NewUnsupportedParamError(providerName, "messages.content.image_url")
	}

	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("image content must contain valid base64: %w", err),
		)
	}

	return decoded, nil
}

// convertModelsResponse converts an Ollama list response to provider format.
func convertModelsResponse(resp *api.ListResponse) *providers.ModelsResponse {
	models := make([]providers.Model, 0, len(resp.Models))

	for _, m := range resp.Models {
		models = append(models, providers.Model{
			ID:      m.Model,
			Object:  objectModel,
			Created: m.ModifiedAt.Unix(),
			OwnedBy: providerName,
		})
	}

	return &providers.ModelsResponse{
		Object: objectList,
		Data:   models,
	}
}

// convertResponse converts an Ollama response to provider format.
func convertResponse(resp *api.ChatResponse) *providers.ChatCompletion {
	message := providers.Message{
		Role:    providers.RoleAssistant,
		Content: resp.Message.Content,
	}
	// Native thinking takes precedence; older models may embed reasoning in content.
	if resp.Message.Thinking != "" {
		message.Reasoning = &providers.Reasoning{Content: resp.Message.Thinking}
	} else if before, tagged, found := strings.Cut(resp.Message.Content, "<think>"); found {
		if thinking, after, closed := strings.Cut(tagged, "</think>"); closed {
			message.Content = strings.TrimSpace(before + after)
			message.Reasoning = &providers.Reasoning{Content: thinking}
		}
	}

	// Handle tool calls.
	if len(resp.Message.ToolCalls) > 0 {
		message.ToolCalls = convertToolCalls(resp.Message.ToolCalls)
	}

	finishReason := providers.FinishReasonToolCalls
	if len(resp.Message.ToolCalls) == 0 {
		finishReason = convertDoneReason(resp.DoneReason)
	}

	return &providers.ChatCompletion{
		ID:      generateID(),
		Object:  objectChatCompletion,
		Created: resp.CreatedAt.Unix(),
		Model:   resp.Model,
		Choices: []providers.Choice{{
			Index:        0,
			Message:      message,
			FinishReason: finishReason,
		}},
		Usage: &providers.Usage{
			PromptTokens:     resp.PromptEvalCount,
			CompletionTokens: resp.EvalCount,
			TotalTokens:      resp.PromptEvalCount + resp.EvalCount,
		},
	}
}

// convertResponseFormat converts a response format to Ollama JSON schema.
func convertResponseFormat(format *providers.ResponseFormat) (json.RawMessage, error) {
	if format == nil {
		return nil, nil
	}

	if format.Type == responseFormatJSON {
		return json.RawMessage(`"` + ollamaFormatJSON + `"`), nil
	}

	if format.Type == responseFormatSchema {
		if format.JSONSchema == nil || format.JSONSchema.Schema == nil {
			return nil, errors.NewInvalidRequestError(
				providerName,
				stderrors.New("json_schema response format requires a schema"),
			)
		}

		if format.JSONSchema.Strict != nil && *format.JSONSchema.Strict {
			return nil, errors.NewUnsupportedParamError(providerName, "response_format.json_schema.strict")
		}

		schemaBytes, err := json.Marshal(format.JSONSchema.Schema)
		if err != nil {
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("response schema must be valid JSON: %w", err),
			)
		}

		return schemaBytes, nil
	}

	return nil, errors.NewUnsupportedParamError(providerName, "response_format.type")
}

// convertToolCalls converts Ollama tool calls to provider format.
func convertToolCalls(toolCalls []api.ToolCall) []providers.ToolCall {
	result := make([]providers.ToolCall, 0, len(toolCalls))

	for i, tc := range toolCalls {
		args := tc.Function.Arguments.String()

		toolCallID := tc.ID
		if toolCallID == "" {
			toolCallID = fmt.Sprintf(toolCallIDFormat, i)
		}

		result = append(result, providers.ToolCall{
			ID:   toolCallID,
			Type: toolTypeFunction,
			Function: providers.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: args,
			},
		})
	}

	return result
}

// convertTools converts provider tools to Ollama format.
func convertTools(tools []providers.Tool) (api.Tools, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	result := make(api.Tools, 0, len(tools))

	for _, tool := range tools {
		if tool.Type != toolTypeFunction {
			return nil, errors.NewUnsupportedParamError(providerName, "tools.type")
		}

		parameters, err := convertToolParameters(tool.Function.Parameters)
		if err != nil {
			return nil, err
		}

		result = append(result, api.Tool{
			Type: toolTypeFunction,
			Function: api.ToolFunction{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  parameters,
			},
		})
	}

	return result, nil
}

func convertToolParameters(parameters map[string]any) (api.ToolFunctionParameters, error) {
	normalizedParameters := map[string]any{
		schemaKeyType:       schemaTypeObject,
		schemaKeyProperties: nil,
	}
	maps.Copy(normalizedParameters, parameters)
	parameters = normalizedParameters

	encoded, err := json.Marshal(parameters)
	if err != nil {
		return api.ToolFunctionParameters{}, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("tool schema must be valid JSON: %w", err),
		)
	}

	var converted api.ToolFunctionParameters
	if unmarshalErr := json.Unmarshal(encoded, &converted); unmarshalErr != nil {
		return api.ToolFunctionParameters{}, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("tool schema is invalid: %w", unmarshalErr),
		)
	}

	normalized, decodeErr := decodeJSONUseNumber(encoded)
	if decodeErr != nil {
		return api.ToolFunctionParameters{}, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("tool schema cannot be compared: %w", decodeErr),
		)
	}
	normalizeToolSchemaSDKOmissions(normalized)

	// https://docs.ollama.com/api/chat accepts JSON Schema, while the Go SDK
	// exposes a typed subset. Reject schemas the SDK would silently weaken.
	roundTripJSON, err := json.Marshal(converted)
	if err != nil {
		return api.ToolFunctionParameters{}, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("tool schema cannot be encoded: %w", err),
		)
	}

	roundTrip, decodeErr := decodeJSONUseNumber(roundTripJSON)
	if decodeErr != nil {
		return api.ToolFunctionParameters{}, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("tool schema cannot be compared: %w", decodeErr),
		)
	}

	if !reflect.DeepEqual(normalized, roundTrip) {
		return api.ToolFunctionParameters{}, errors.NewUnsupportedParamError(providerName, "tools.function.parameters")
	}

	return converted, nil
}

func normalizeToolSchemaSDKOmissions(value any) {
	schema, ok := value.(map[string]any)
	if !ok {
		return
	}

	if required, exists := schema[schemaKeyRequired]; exists {
		if values, isArray := required.([]any); isArray && len(values) == 0 {
			delete(schema, schemaKeyRequired)
		}
	}

	properties, ok := schema[schemaKeyProperties].(map[string]any)
	if !ok {
		return
	}
	for _, propertyValue := range properties {
		property, isObject := propertyValue.(map[string]any)
		if !isObject {
			continue
		}
		if description, exists := property[schemaKeyDescription]; exists && description == "" {
			delete(property, schemaKeyDescription)
		}
		normalizeToolSchemaSDKOmissions(property)
	}
}

func decodeJSONUseNumber(data []byte) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()

	var value any

	err := decoder.Decode(&value)

	return value, err
}

// generateID generates a unique ID for responses using crypto/rand.
func generateID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return fmt.Sprintf("chatcmpl-%d-%s", time.Now().UnixNano(), hex.EncodeToString(b))
}
