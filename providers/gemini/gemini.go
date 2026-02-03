// Package gemini provides a Google Gemini provider implementation for any-llm.
package gemini

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"log"
	"strings"
	"time"

	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	envAPIKey       = "GEMINI_API_KEY"
	envAPIKeyGoogle = "GOOGLE_API_KEY"
	providerName    = "gemini"
)

// Default thinking budgets for reasoning effort levels.
// These match the Python any-llm library.
const (
	thinkingBudgetHigh   int32 = 24576
	thinkingBudgetLow    int32 = 1024
	thinkingBudgetMedium int32 = 8192
)

// Content part types.
const (
	contentPartTypeImageURL = "image_url"
	contentPartTypeText     = "text"
)

// Gemini role constants.
const (
	roleModel = "model"
	roleUser  = "user"
)

// Object type constants (Gemini doesn't provide these; we set them ourselves).
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectEmbedding           = "embedding"
	objectList                = "list"
	objectModel               = "model"
)

// Response format and tool type constants.
const (
	responseMIMETypeJSON = "application/json"
	responseFormatJSON   = "json_object"
	toolCallFallbackName = "function"
	toolCallType         = "function"
)

// ID prefix constants for generated identifiers.
const (
	idPrefixCompletion = "gemini-"
	idPrefixToolCall   = "call_"
)

// Default MIME type for image URLs when type cannot be determined.
const defaultImageMIMEType = "image/jpeg"

// Error message patterns for 400 error classification.
// The Gemini SDK doesn't expose typed errors for these conditions,
// so we rely on message matching as a pragmatic fallback.
const (
	errMsgContext = "context"
	errMsgToken   = "token"
	errMsgSafety  = "safety"
	errMsgBlock   = "block"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Google Gemini.
type Provider struct {
	client *genai.Client
	config *config.Config
}

// streamState tracks accumulated state during streaming.
type streamState struct {
	content      strings.Builder
	finishReason genai.FinishReason
	messageID    string
	model        string
	reasoning    strings.Builder
	toolCalls    []providers.ToolCall
	usage        *providers.Usage
}

// New creates a new Gemini provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	apiKey := cfg.ResolveAPIKey(envAPIKey)
	if apiKey == "" {
		apiKey = cfg.ResolveEnv(envAPIKeyGoogle)
	}
	if apiKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:     apiKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: cfg.HTTPClient(),
	})
	if err != nil {
		return nil, fmt.Errorf("creating Gemini client: %w", err)
	}

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
		Embedding:           true,
		ListModels:          true,
	}
}

// Completion performs a chat completion request.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	contents, cfg := p.convertParams(params)

	resp, err := p.client.Models.GenerateContent(ctx, params.Model, contents, cfg)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertResponse(resp, params.Model)
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

		contents, cfg := p.convertParams(params)
		state, err := newStreamState(params.Model)
		if err != nil {
			select {
			case errs <- err:
			case <-ctx.Done():
			}
			return
		}

		for resp, err := range p.client.Models.GenerateContentStream(ctx, params.Model, contents, cfg) {
			if err != nil {
				select {
				case errs <- p.ConvertError(err):
				case <-ctx.Done():
				}
				return
			}

			responseChunks, err := state.processResponse(resp)
			if err != nil {
				select {
				case errs <- err:
				case <-ctx.Done():
				}
				return
			}

			for _, chunk := range responseChunks {
				select {
				case chunks <- chunk:
				case <-ctx.Done():
					return
				}
			}
		}

		// Emit final chunk with finish reason and usage.
		if finalChunk := state.finalChunk(); finalChunk != nil {
			select {
			case chunks <- *finalChunk:
			case <-ctx.Done():
			}
		}
	}()

	return chunks, errs
}

// ConvertError converts a Gemini SDK error to a unified error type.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *genai.APIError
	if !stderrors.As(err, &apiErr) {
		return errors.NewProviderError(providerName, err)
	}

	switch apiErr.Code {
	case 401, 403:
		return errors.NewAuthenticationError(providerName, err)
	case 404:
		return errors.NewModelNotFoundError(providerName, err)
	case 429:
		return errors.NewRateLimitError(providerName, err)
	case 400:
		// The Gemini SDK doesn't expose typed errors for context length or content
		// filter violations, so we use message matching as a pragmatic fallback.
		msg := strings.ToLower(apiErr.Message)
		if strings.Contains(msg, errMsgContext) || strings.Contains(msg, errMsgToken) {
			return errors.NewContextLengthError(providerName, err)
		}
		if strings.Contains(msg, errMsgSafety) || strings.Contains(msg, errMsgBlock) {
			return errors.NewContentFilterError(providerName, err)
		}
		return errors.NewInvalidRequestError(providerName, err)
	default:
		return errors.NewProviderError(providerName, err)
	}
}

// Embedding generates embeddings for the given input.
func (p *Provider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	content := convertEmbeddingInput(params.Input)

	resp, err := p.client.Models.EmbedContent(ctx, params.Model, []*genai.Content{content}, nil)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	data := make([]providers.EmbeddingData, 0, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		values := make([]float64, len(emb.Values))
		for j, v := range emb.Values {
			values[j] = float64(v)
		}
		data = append(data, providers.EmbeddingData{
			Object:    objectEmbedding,
			Embedding: values,
			Index:     i,
		})
	}

	return &providers.EmbeddingResponse{
		Object: objectList,
		Data:   data,
		Model:  params.Model,
	}, nil
}

// ListModels returns available models.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	var models []providers.Model

	page, err := p.client.Models.List(ctx, nil)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	for {
		for _, m := range page.Items {
			models = append(models, providers.Model{
				ID:      m.Name,
				Object:  objectModel,
				OwnedBy: "google",
			})
		}

		if page.NextPageToken == "" {
			break
		}

		page, err = page.Next(ctx)
		if stderrors.Is(err, genai.ErrPageDone) {
			break
		}
		if err != nil {
			return nil, p.ConvertError(err)
		}
	}

	return &providers.ModelsResponse{
		Object: objectList,
		Data:   models,
	}, nil
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// convertParams converts providers.CompletionParams to Gemini request format.
func (p *Provider) convertParams(params providers.CompletionParams) ([]*genai.Content, *genai.GenerateContentConfig) {
	contents, systemInstruction := convertMessages(params.Messages)

	cfg := &genai.GenerateContentConfig{}

	if systemInstruction != nil {
		cfg.SystemInstruction = systemInstruction
	}

	if params.Temperature != nil {
		t := float32(*params.Temperature)
		cfg.Temperature = &t
	}

	if params.TopP != nil {
		tp := float32(*params.TopP)
		cfg.TopP = &tp
	}

	if params.MaxTokens != nil {
		cfg.MaxOutputTokens = int32(*params.MaxTokens)
	}

	if len(params.Stop) > 0 {
		cfg.StopSequences = params.Stop
	}

	if len(params.Tools) > 0 {
		cfg.Tools = convertTools(params.Tools)
	}

	if params.ToolChoice != nil {
		cfg.ToolConfig = convertToolChoice(params.ToolChoice)
	}

	applyThinking(cfg, params.ReasoningEffort)

	if params.ResponseFormat != nil {
		applyResponseFormat(cfg, params.ResponseFormat)
	}

	return contents, cfg
}

// newStreamState creates a new stream state.
func newStreamState(model string) (*streamState, error) {
	id, err := generateID(idPrefixCompletion)
	if err != nil {
		return nil, err
	}
	return &streamState{
		messageID: id,
		model:     model,
	}, nil
}

// chunk creates a ChatCompletionChunk with the given delta.
func (s *streamState) chunk(delta providers.ChunkDelta) providers.ChatCompletionChunk {
	return providers.ChatCompletionChunk{
		ID:     s.messageID,
		Object: objectChatCompletionChunk,
		Model:  s.model,
		Choices: []providers.ChunkChoice{{
			Index: 0,
			Delta: delta,
		}},
	}
}

// finalChunk returns the final chunk with finish reason and usage.
func (s *streamState) finalChunk() *providers.ChatCompletionChunk {
	chunk := s.chunk(providers.ChunkDelta{})

	finishReason := convertFinishReason(s.finishReason)
	if len(s.toolCalls) > 0 && finishReason == providers.FinishReasonStop {
		finishReason = providers.FinishReasonToolCalls
	}

	chunk.Choices[0].FinishReason = finishReason
	chunk.Usage = s.usage
	return &chunk
}

// processResponse processes a streaming response and returns chunks to emit.
func (s *streamState) processResponse(resp *genai.GenerateContentResponse) ([]providers.ChatCompletionChunk, error) {
	var result []providers.ChatCompletionChunk

	if resp.UsageMetadata != nil {
		s.usage = &providers.Usage{
			PromptTokens:     int(resp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(resp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      int(resp.UsageMetadata.PromptTokenCount + resp.UsageMetadata.CandidatesTokenCount),
			ReasoningTokens:  int(resp.UsageMetadata.ThoughtsTokenCount),
		}
	}

	if len(resp.Candidates) == 0 {
		return result, nil
	}

	candidate := resp.Candidates[0]

	if candidate.FinishReason != "" {
		s.finishReason = candidate.FinishReason
	}

	if candidate.Content == nil {
		return result, nil
	}

	for _, part := range candidate.Content.Parts {
		switch {
		case part.FunctionCall != nil:
			toolCall, err := convertFunctionCallToToolCall(part.FunctionCall)
			if err != nil {
				return nil, err
			}
			s.toolCalls = append(s.toolCalls, toolCall)
			result = append(result, s.chunk(providers.ChunkDelta{
				ToolCalls: []providers.ToolCall{toolCall},
			}))
		case part.Thought:
			s.reasoning.WriteString(part.Text)
			result = append(result, s.chunk(providers.ChunkDelta{
				Reasoning: &providers.Reasoning{Content: part.Text},
			}))
		case part.Text != "":
			s.content.WriteString(part.Text)
			result = append(result, s.chunk(providers.ChunkDelta{
				Content: part.Text,
			}))
		}
	}

	return result, nil
}

// applyResponseFormat configures the response format on the config.
func applyResponseFormat(cfg *genai.GenerateContentConfig, format *providers.ResponseFormat) {
	if format.Type == responseFormatJSON {
		cfg.ResponseMIMEType = responseMIMETypeJSON
	}
}

// applyThinking configures thinking/reasoning on the config if applicable.
func applyThinking(cfg *genai.GenerateContentConfig, effort providers.ReasoningEffort) {
	if effort == "" || effort == providers.ReasoningEffortNone {
		return
	}

	budget, ok := thinkingBudget(effort)
	if !ok {
		return
	}

	cfg.ThinkingConfig = &genai.ThinkingConfig{
		IncludeThoughts: true,
		ThinkingBudget:  &budget,
	}
}

// convertAssistantMessage converts an assistant message to Gemini format.
func convertAssistantMessage(msg providers.Message) *genai.Content {
	var parts []*genai.Part

	text := msg.ContentString()
	if text != "" {
		parts = append(parts, &genai.Part{Text: text})
	}

	for _, tc := range msg.ToolCalls {
		var args map[string]any
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		parts = append(parts, genai.NewPartFromFunctionCall(tc.Function.Name, args))
	}

	if len(parts) == 0 {
		return nil
	}

	return &genai.Content{
		Role:  roleModel,
		Parts: parts,
	}
}

// convertEmbeddingInput converts embedding input to Gemini content.
func convertEmbeddingInput(input any) *genai.Content {
	switch v := input.(type) {
	case string:
		return genai.NewContentFromText(v, roleUser)
	case []string:
		parts := make([]*genai.Part, len(v))
		for i, s := range v {
			parts[i] = genai.NewPartFromText(s)
		}
		return genai.NewContentFromParts(parts, roleUser)
	default:
		return genai.NewContentFromText(fmt.Sprintf("%v", v), roleUser)
	}
}

// convertFinishReason converts a Gemini finish reason to OpenAI format.
func convertFinishReason(reason genai.FinishReason) string {
	switch reason {
	case genai.FinishReasonStop:
		return providers.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		return providers.FinishReasonLength
	case genai.FinishReasonSafety, genai.FinishReasonBlocklist, genai.FinishReasonProhibitedContent:
		return providers.FinishReasonContentFilter
	case genai.FinishReasonRecitation:
		return providers.FinishReasonStop
	default:
		return providers.FinishReasonStop
	}
}

// convertFunctionCallToToolCall converts a Gemini function call to a providers tool call.
func convertFunctionCallToToolCall(fc *genai.FunctionCall) (providers.ToolCall, error) {
	argsJSON := ""
	if fc.Args != nil {
		if b, err := json.Marshal(fc.Args); err == nil {
			argsJSON = string(b)
		}
	}

	id, err := generateID(idPrefixToolCall)
	if err != nil {
		return providers.ToolCall{}, err
	}

	return providers.ToolCall{
		ID:   id,
		Type: toolCallType,
		Function: providers.FunctionCall{
			Name:      fc.Name,
			Arguments: argsJSON,
		},
	}, nil
}

// convertImagePart converts an image URL to Gemini part format.
// For data URLs, it extracts the base64-encoded data and MIME type.
// For regular URLs, it treats them as file URIs with a default MIME type.
func convertImagePart(img *providers.ImageURL) *genai.Part {
	url := img.URL

	if strings.HasPrefix(url, "data:") {
		parts := strings.SplitN(url, ",", 2)
		if len(parts) == 2 {
			mediaTypePart := strings.TrimPrefix(parts[0], "data:")
			mediaType := strings.Split(mediaTypePart, ";")[0]
			data, err := base64.StdEncoding.DecodeString(parts[1])
			if err == nil {
				return genai.NewPartFromBytes(data, mediaType)
			}
			// Base64 decoding failed for data URL; fall through to treat as file URI.
			// This handles malformed data URLs gracefully.
		}
	}

	return &genai.Part{
		FileData: &genai.FileData{
			FileURI:  url,
			MIMEType: defaultImageMIMEType,
		},
	}
}

// convertMessage converts a single message to Gemini format.
// Returns nil for unknown roles (with a warning logged).
func convertMessage(msg providers.Message) *genai.Content {
	switch msg.Role {
	case providers.RoleUser:
		return convertUserMessage(msg)
	case providers.RoleAssistant:
		return convertAssistantMessage(msg)
	case providers.RoleTool:
		return convertToolMessage(msg)
	default:
		log.Printf("gemini: unknown message role %q, skipping message", msg.Role)
		return nil
	}
}

// convertMessages converts providers messages to Gemini format.
// Returns the contents and the system instruction (if any).
func convertMessages(messages []providers.Message) ([]*genai.Content, *genai.Content) {
	var contents []*genai.Content
	var systemParts []string

	for _, msg := range messages {
		if msg.Role == providers.RoleSystem {
			systemParts = append(systemParts, msg.ContentString())
			continue
		}

		if converted := convertMessage(msg); converted != nil {
			contents = append(contents, converted)
		}
	}

	var systemInstruction *genai.Content
	if len(systemParts) > 0 {
		systemInstruction = genai.NewContentFromText(strings.Join(systemParts, "\n"), roleUser)
	}

	return contents, systemInstruction
}

// extractResponseContent extracts content, reasoning, tool calls, and finish reason from a Gemini response.
func extractResponseContent(
	resp *genai.GenerateContentResponse,
) (string, *providers.Reasoning, []providers.ToolCall, string, error) {
	if len(resp.Candidates) == 0 {
		return "", nil, nil, "", nil
	}

	candidate := resp.Candidates[0]
	finishReason := convertFinishReason(candidate.FinishReason)

	if candidate.Content == nil {
		return "", nil, nil, finishReason, nil
	}

	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	var toolCalls []providers.ToolCall

	for _, part := range candidate.Content.Parts {
		switch {
		case part.FunctionCall != nil:
			toolCall, err := convertFunctionCallToToolCall(part.FunctionCall)
			if err != nil {
				return "", nil, nil, "", err
			}
			toolCalls = append(toolCalls, toolCall)
		case part.Thought:
			reasoningBuilder.WriteString(part.Text)
		case part.Text != "":
			contentBuilder.WriteString(part.Text)
		}
	}

	var reasoning *providers.Reasoning
	if reasoningBuilder.Len() > 0 {
		reasoning = &providers.Reasoning{Content: reasoningBuilder.String()}
	}

	return contentBuilder.String(), reasoning, toolCalls, finishReason, nil
}

// convertResponse converts a Gemini response to providers format.
func convertResponse(resp *genai.GenerateContentResponse, model string) (*providers.ChatCompletion, error) {
	content, reasoning, toolCalls, finishReason, err := extractResponseContent(resp)
	if err != nil {
		return nil, err
	}

	if len(toolCalls) > 0 && finishReason == providers.FinishReasonStop {
		finishReason = providers.FinishReasonToolCalls
	}

	message := providers.Message{
		Role:      providers.RoleAssistant,
		Content:   content,
		ToolCalls: toolCalls,
		Reasoning: reasoning,
	}

	id, err := generateID(idPrefixCompletion)
	if err != nil {
		return nil, err
	}

	completion := &providers.ChatCompletion{
		ID:      id,
		Object:  objectChatCompletion,
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []providers.Choice{{
			Index:        0,
			Message:      message,
			FinishReason: finishReason,
		}},
	}

	if resp.UsageMetadata != nil {
		completion.Usage = &providers.Usage{
			PromptTokens:     int(resp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(resp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      int(resp.UsageMetadata.PromptTokenCount + resp.UsageMetadata.CandidatesTokenCount),
			ReasoningTokens:  int(resp.UsageMetadata.ThoughtsTokenCount),
		}
	}

	return completion, nil
}

// convertToolChoice converts providers tool choice to Gemini format.
func convertToolChoice(choice any) *genai.ToolConfig {
	switch v := choice.(type) {
	case string:
		switch v {
		case "auto":
			return &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAuto,
				},
			}
		case "none":
			return &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeNone,
				},
			}
		case "required", "any":
			return &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			}
		}
	case providers.ToolChoice:
		if v.Function != nil {
			return &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{v.Function.Name},
				},
			}
		}
	}

	return nil
}

// convertToolMessage converts a tool result message to Gemini format.
func convertToolMessage(msg providers.Message) *genai.Content {
	name := msg.Name
	if name == "" {
		name = toolCallFallbackName
	}

	content := msg.ContentString()

	// Try to parse content as JSON first (structured tool responses).
	// If parsing fails, wrap the raw content as {"result": content}.
	var response map[string]any
	if err := json.Unmarshal([]byte(content), &response); err != nil {
		response = map[string]any{
			"result": content,
		}
	}

	return &genai.Content{
		Role:  roleUser,
		Parts: []*genai.Part{genai.NewPartFromFunctionResponse(name, response)},
	}
}

// convertTools converts providers tools to Gemini format.
func convertTools(tools []providers.Tool) []*genai.Tool {
	declarations := make([]*genai.FunctionDeclaration, 0, len(tools))

	for _, tool := range tools {
		decl := &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
		}

		if tool.Function.Parameters != nil {
			decl.ParametersJsonSchema = tool.Function.Parameters
		}

		declarations = append(declarations, decl)
	}

	return []*genai.Tool{{
		FunctionDeclarations: declarations,
	}}
}

// convertUserMessage converts a user message to Gemini format.
func convertUserMessage(msg providers.Message) *genai.Content {
	if !msg.IsMultiModal() {
		return genai.NewContentFromText(msg.ContentString(), roleUser)
	}

	var parts []*genai.Part
	for _, part := range msg.ContentParts() {
		switch part.Type {
		case contentPartTypeText:
			parts = append(parts, genai.NewPartFromText(part.Text))
		case contentPartTypeImageURL:
			if part.ImageURL != nil {
				parts = append(parts, convertImagePart(part.ImageURL))
			}
		}
	}

	return genai.NewContentFromParts(parts, roleUser)
}

// generateID generates a random ID with the given prefix.
func generateID(prefix string) (string, error) {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("generating ID: %w", err)
	}
	return prefix + hex.EncodeToString(b), nil
}

// thinkingBudget returns the token budget for the given reasoning effort.
func thinkingBudget(effort providers.ReasoningEffort) (int32, bool) {
	switch effort {
	case providers.ReasoningEffortLow:
		return thinkingBudgetLow, true
	case providers.ReasoningEffortMedium:
		return thinkingBudgetMedium, true
	case providers.ReasoningEffortHigh:
		return thinkingBudgetHigh, true
	default:
		return 0, false
	}
}
