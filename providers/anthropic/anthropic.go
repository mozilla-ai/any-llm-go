// Package anthropic provides an Anthropic provider implementation for any-llm.
package anthropic

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	defaultMaxTokens = 4096
	envAPIKey        = "ANTHROPIC_API_KEY"
	providerName     = "anthropic"
)

// Anthropic content block types.
const (
	blockTypeText     = "text"
	blockTypeThinking = "thinking"
	blockTypeToolUse  = "tool_use"
)

// Anthropic delta types.
const (
	deltaTypeInputJSON = "input_json_delta"
	deltaTypeText      = "text_delta"
	deltaTypeThinking  = "thinking_delta"
)

// Anthropic error response patterns (checked in raw JSON).
const (
	errorPatternContextLength = "context_length"
	errorPatternToken         = "token"
	errorPatternContent       = "content"
	errorPatternSafety        = "safety"
)

// Anthropic streaming event types.
const (
	eventContentBlockDelta = "content_block_delta"
	eventContentBlockStart = "content_block_start"
	eventMessageDelta      = "message_delta"
	eventMessageStart      = "message_start"
)

// Anthropic stop reasons.
const (
	stopReasonEndTurn      = "end_turn"
	stopReasonMaxTokens    = "max_tokens"
	stopReasonStopSequence = "stop_sequence"
	stopReasonToolUse      = "tool_use"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for Anthropic.
type Provider struct {
	client *anthropic.Client
	config *config.Config
}

// streamState tracks accumulated state during streaming.
// Note: Only accessed from a single goroutine, so no synchronization needed.
type streamState struct {
	messageID      string
	model          string
	content        strings.Builder
	reasoning      strings.Builder
	toolCalls      []providers.ToolCall
	currentToolIdx int
	inputUsage     int64
}

// New creates a new Anthropic provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	apiKey := cfg.ResolveAPIKey(envAPIKey)
	if apiKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	if cfg.BaseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(cfg.BaseURL))
	}

	client := anthropic.NewClient(clientOpts...)

	return &Provider{
		client: &client,
		config: cfg,
	}, nil
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionStreaming: true,
		CompletionReasoning: true,
		CompletionImage:     true,
		CompletionPDF:       true,
		Embedding:           false,
		ListModels:          false,
	}
}

// Completion performs a chat completion request.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	req := p.convertParams(params)

	resp, err := p.client.Messages.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return convertResponse(resp), nil
}

// convertParams converts providers.CompletionParams to Anthropic request parameters.
func (p *Provider) convertParams(params providers.CompletionParams) anthropic.MessageNewParams {
	messages, system := convertMessages(params.Messages)

	maxTokens := int64(defaultMaxTokens)
	if params.MaxTokens != nil {
		maxTokens = int64(*params.MaxTokens)
	}

	req := anthropic.MessageNewParams{
		Model:     anthropic.Model(params.Model),
		Messages:  messages,
		MaxTokens: maxTokens,
	}

	if system != "" {
		req.System = []anthropic.TextBlockParam{
			{Text: system},
		}
	}

	if params.Temperature != nil {
		req.Temperature = anthropic.Float(*params.Temperature)
	}

	if params.TopP != nil {
		req.TopP = anthropic.Float(*params.TopP)
	}

	if len(params.Stop) > 0 {
		req.StopSequences = params.Stop
	}

	if len(params.Tools) > 0 {
		tools := make([]anthropic.ToolUnionParam, 0, len(params.Tools))
		for _, tool := range params.Tools {
			tools = append(tools, convertTool(tool))
		}
		req.Tools = tools
	}

	if params.ToolChoice != nil {
		req.ToolChoice = convertToolChoice(params.ToolChoice, params.ParallelToolCalls)
	}

	applyThinking(&req, params.ReasoningEffort, maxTokens)

	return req
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

		req := p.convertParams(params)
		stream := p.client.Messages.NewStreaming(ctx, req)
		state := newStreamState()

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case eventMessageStart:
				chunks <- state.handleMessageStart(event.AsMessageStart())

			case eventContentBlockStart:
				state.handleContentBlockStart(event.AsContentBlockStart())

			case eventContentBlockDelta:
				if chunk := state.handleContentBlockDelta(event.AsContentBlockDelta()); chunk != nil {
					chunks <- *chunk
				}

			case eventMessageDelta:
				chunks <- state.handleMessageDelta(event.AsMessageDelta())
			}
		}

		if err := stream.Err(); err != nil {
			errs <- p.ConvertError(err)
		}
	}()

	return chunks, errs
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// newStreamState creates a new stream state with default values.
func newStreamState() *streamState {
	return &streamState{
		currentToolIdx: -1,
	}
}

// chunk creates a ChatCompletionChunk with the given delta.
func (s *streamState) chunk(delta providers.ChunkDelta) providers.ChatCompletionChunk {
	return providers.ChatCompletionChunk{
		ID:     s.messageID,
		Object: "chat.completion.chunk",
		Model:  s.model,
		Choices: []providers.ChunkChoice{{
			Index: 0,
			Delta: delta,
		}},
	}
}

// handleContentBlockDelta processes a content_block_delta event and returns a chunk if applicable.
func (s *streamState) handleContentBlockDelta(event anthropic.ContentBlockDeltaEvent) *providers.ChatCompletionChunk {
	switch event.Delta.Type {
	case deltaTypeText:
		return s.handleTextDelta(event.Delta.Text)
	case deltaTypeThinking:
		return s.handleThinkingDelta(event.Delta.Thinking)
	case deltaTypeInputJSON:
		return s.handleInputJSONDelta(event.Delta.PartialJSON)
	default:
		return nil
	}
}

// handleContentBlockStart processes a content_block_start event.
func (s *streamState) handleContentBlockStart(event anthropic.ContentBlockStartEvent) {
	switch event.ContentBlock.Type {
	case blockTypeThinking:
		// Reasoning block started - no action needed.
	case blockTypeToolUse:
		s.currentToolIdx++
		// TODO: Extract to newToolCallFromBlock() if this pattern is needed elsewhere.
		tc := providers.ToolCall{
			ID:   event.ContentBlock.ID,
			Type: "function",
			Function: providers.FunctionCall{
				Name: event.ContentBlock.Name,
			},
		}
		s.toolCalls = append(s.toolCalls, tc)
	}
}

// handleInputJSONDelta processes a tool input JSON delta and returns a chunk if applicable.
func (s *streamState) handleInputJSONDelta(partialJSON string) *providers.ChatCompletionChunk {
	if s.currentToolIdx < 0 || s.currentToolIdx >= len(s.toolCalls) {
		return nil
	}

	s.toolCalls[s.currentToolIdx].Function.Arguments += partialJSON
	chunk := s.chunk(providers.ChunkDelta{
		ToolCalls: []providers.ToolCall{s.toolCalls[s.currentToolIdx]},
	})
	return &chunk
}

// handleMessageDelta processes a message_delta event and returns the final chunk.
func (s *streamState) handleMessageDelta(event anthropic.MessageDeltaEvent) providers.ChatCompletionChunk {
	finishReason := convertStopReason(string(event.Delta.StopReason))
	chunk := s.chunk(providers.ChunkDelta{})
	chunk.Choices[0].FinishReason = finishReason
	chunk.Usage = &providers.Usage{
		PromptTokens:     int(s.inputUsage),
		CompletionTokens: int(event.Usage.OutputTokens),
		TotalTokens:      int(s.inputUsage + event.Usage.OutputTokens),
	}
	return chunk
}

// handleMessageStart processes a message_start event and returns the initial chunk.
func (s *streamState) handleMessageStart(event anthropic.MessageStartEvent) providers.ChatCompletionChunk {
	s.messageID = event.Message.ID
	s.model = string(event.Message.Model)
	s.inputUsage = event.Message.Usage.InputTokens

	return s.chunk(providers.ChunkDelta{Role: providers.RoleAssistant})
}

// handleThinkingDelta processes a thinking delta and returns a chunk.
func (s *streamState) handleThinkingDelta(thinking string) *providers.ChatCompletionChunk {
	s.reasoning.WriteString(thinking)
	chunk := s.chunk(providers.ChunkDelta{
		Reasoning: &providers.Reasoning{Content: thinking},
	})
	return &chunk
}

// handleTextDelta processes a text delta and returns a chunk.
func (s *streamState) handleTextDelta(text string) *providers.ChatCompletionChunk {
	s.content.WriteString(text)
	chunk := s.chunk(providers.ChunkDelta{Content: text})
	return &chunk
}

// applyThinking configures thinking/reasoning on the request if applicable.
func applyThinking(req *anthropic.MessageNewParams, effort providers.ReasoningEffort, maxTokens int64) {
	if effort == "" || effort == providers.ReasoningEffortNone {
		return
	}

	budget, ok := thinkingBudget(effort)
	if !ok {
		return
	}

	req.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)

	// Increase max tokens to accommodate thinking.
	minTokens := budget * 2
	if maxTokens < minTokens {
		req.MaxTokens = minTokens
	}
}

// convertAssistantMessage converts an assistant message to Anthropic format.
func convertAssistantMessage(msg providers.Message) *anthropic.MessageParam {
	if len(msg.ToolCalls) == 0 {
		m := anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.ContentString()))
		return &m
	}

	content := make([]anthropic.ContentBlockParamUnion, 0)
	if msg.ContentString() != "" {
		content = append(content, anthropic.NewTextBlock(msg.ContentString()))
	}

	for _, tc := range msg.ToolCalls {
		content = append(content, convertToolCall(tc))
	}

	m := anthropic.NewAssistantMessage(content...)
	return &m
}

// convertImagePart converts an image URL to Anthropic format.
func convertImagePart(img *providers.ImageURL) anthropic.ContentBlockParamUnion {
	url := img.URL

	// Check if it's a base64 data URL.
	if strings.HasPrefix(url, "data:") {
		// Parse data URL: data:image/jpeg;base64,<data>.
		parts := strings.SplitN(url, ",", 2)
		if len(parts) == 2 {
			// Extract media type from the first part.
			mediaTypePart := strings.TrimPrefix(parts[0], "data:")
			mediaType := strings.Split(mediaTypePart, ";")[0]
			data := parts[1]

			return anthropic.NewImageBlockBase64(mediaType, data)
		}
	}

	// Regular URL.
	return anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: url})
}

// convertMessage converts a single message to Anthropic format.
func convertMessage(msg providers.Message) *anthropic.MessageParam {
	switch msg.Role {
	case providers.RoleUser:
		return convertUserMessage(msg)
	case providers.RoleAssistant:
		return convertAssistantMessage(msg)
	case providers.RoleTool:
		return convertToolMessage(msg)
	default:
		return nil
	}
}

// convertMessages converts providers messages to Anthropic format.
// Returns the messages and the combined system message.
func convertMessages(messages []providers.Message) ([]anthropic.MessageParam, string) {
	result := make([]anthropic.MessageParam, 0, len(messages))
	var systemParts []string

	for _, msg := range messages {
		if msg.Role == providers.RoleSystem {
			systemParts = append(systemParts, msg.ContentString())
			continue
		}

		if converted := convertMessage(msg); converted != nil {
			result = append(result, *converted)
		}
	}

	return result, strings.Join(systemParts, "\n")
}

// convertResponse converts an Anthropic response to providers format.
func convertResponse(resp *anthropic.Message) *providers.ChatCompletion {
	var content string
	var reasoning *providers.Reasoning
	var toolCalls []providers.ToolCall

	for _, block := range resp.Content {
		switch block.Type {
		case blockTypeText:
			content += block.Text
		case blockTypeThinking:
			reasoning = &providers.Reasoning{
				Content: block.Thinking,
			}
		case blockTypeToolUse:
			inputJSON := ""
			if block.Input != nil {
				if inputBytes, err := json.Marshal(block.Input); err == nil {
					inputJSON = string(inputBytes)
				}
			}
			toolCalls = append(toolCalls, providers.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: providers.FunctionCall{
					Name:      block.Name,
					Arguments: inputJSON,
				},
			})
		}
	}

	message := providers.Message{
		Role:      providers.RoleAssistant,
		Content:   content,
		ToolCalls: toolCalls,
		Reasoning: reasoning,
	}

	finishReason := convertStopReason(string(resp.StopReason))

	return &providers.ChatCompletion{
		ID:     resp.ID,
		Object: "chat.completion",
		Model:  string(resp.Model),
		Choices: []providers.Choice{{
			Index:        0,
			Message:      message,
			FinishReason: finishReason,
		}},
		Usage: &providers.Usage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
	}
}

// convertStopReason converts Anthropic stop reason to OpenAI finish reason.
func convertStopReason(reason string) string {
	switch reason {
	case stopReasonEndTurn:
		return providers.FinishReasonStop
	case stopReasonMaxTokens:
		return providers.FinishReasonLength
	case stopReasonToolUse:
		return providers.FinishReasonToolCalls
	case stopReasonStopSequence:
		return providers.FinishReasonStop
	default:
		return providers.FinishReasonStop
	}
}

// convertTool converts a providers.Tool to Anthropic format.
func convertTool(tool providers.Tool) anthropic.ToolUnionParam {
	inputSchema := anthropic.ToolInputSchemaParam{
		Type: "object",
	}
	if props, ok := tool.Function.Parameters["properties"]; ok {
		inputSchema.Properties = props
	}
	if req, ok := tool.Function.Parameters["required"]; ok {
		if reqArr, ok := req.([]any); ok {
			required := make([]string, len(reqArr))
			for i, r := range reqArr {
				if s, ok := r.(string); ok {
					required[i] = s
				}
			}
			inputSchema.Required = required
		}
	}

	return anthropic.ToolUnionParam{
		OfTool: &anthropic.ToolParam{
			Name:        tool.Function.Name,
			Description: anthropic.String(tool.Function.Description),
			InputSchema: inputSchema,
		},
	}
}

// convertToolCall converts a tool call to Anthropic content block format.
func convertToolCall(tc providers.ToolCall) anthropic.ContentBlockParamUnion {
	var input map[string]any
	_ = json.Unmarshal([]byte(tc.Function.Arguments), &input) // Ignore error: use nil on failure.

	return anthropic.ContentBlockParamUnion{
		OfToolUse: &anthropic.ToolUseBlockParam{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		},
	}
}

// convertToolChoice converts providers tool choice to Anthropic format.
func convertToolChoice(choice any, parallelToolCalls *bool) anthropic.ToolChoiceUnionParam {
	disableParallel := parallelToolCalls != nil && !*parallelToolCalls

	switch v := choice.(type) {
	case string:
		switch v {
		case "auto":
			return anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{
					DisableParallelToolUse: anthropic.Bool(disableParallel),
				},
			}
		case "none":
			return anthropic.ToolChoiceUnionParam{
				OfNone: &anthropic.ToolChoiceNoneParam{},
			}
		case "required", "any":
			return anthropic.ToolChoiceUnionParam{
				OfAny: &anthropic.ToolChoiceAnyParam{
					DisableParallelToolUse: anthropic.Bool(disableParallel),
				},
			}
		}
	case providers.ToolChoice:
		if v.Function != nil {
			return anthropic.ToolChoiceUnionParam{
				OfTool: &anthropic.ToolChoiceToolParam{
					Name:                   v.Function.Name,
					DisableParallelToolUse: anthropic.Bool(disableParallel),
				},
			}
		}
	}

	return anthropic.ToolChoiceUnionParam{
		OfAuto: &anthropic.ToolChoiceAutoParam{
			DisableParallelToolUse: anthropic.Bool(disableParallel),
		},
	}
}

// convertToolMessage converts a tool result message to Anthropic format.
func convertToolMessage(msg providers.Message) *anthropic.MessageParam {
	m := anthropic.NewUserMessage(
		anthropic.NewToolResultBlock(msg.ToolCallID, msg.ContentString(), false),
	)
	return &m
}

// convertUserMessage converts a user message to Anthropic format.
func convertUserMessage(msg providers.Message) *anthropic.MessageParam {
	if !msg.IsMultiModal() {
		m := anthropic.NewUserMessage(anthropic.NewTextBlock(msg.ContentString()))
		return &m
	}

	content := make([]anthropic.ContentBlockParamUnion, 0)
	for _, part := range msg.ContentParts() {
		switch part.Type {
		case "text":
			content = append(content, anthropic.NewTextBlock(part.Text))
		case "image_url":
			if part.ImageURL != nil {
				content = append(content, convertImagePart(part.ImageURL))
			}
		}
	}
	m := anthropic.NewUserMessage(content...)
	return &m
}

// thinkingBudget returns the token budget for the given reasoning effort.
// Returns the budget and true if the effort level is supported, or 0 and false otherwise.
func thinkingBudget(effort providers.ReasoningEffort) (int64, bool) {
	switch effort {
	case providers.ReasoningEffortLow:
		return 1024, true
	case providers.ReasoningEffortMedium:
		return 4096, true
	case providers.ReasoningEffortHigh:
		return 16384, true
	default:
		return 0, false
	}
}

// ConvertError converts an Anthropic SDK error to a unified error type.
// Implements providers.ErrorConverter.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	// Extract the Anthropic API error type from the error chain.
	// If it's not an API error (e.g., network error), wrap as generic provider error.
	var apiErr *anthropic.Error
	if !stderrors.As(err, &apiErr) {
		return errors.NewProviderError(providerName, err)
	}

	// Classify by HTTP status code.
	switch apiErr.StatusCode {
	case 401:
		return errors.NewAuthenticationError(providerName, err)
	case 429:
		return errors.NewRateLimitError(providerName, err)
	case 404:
		return errors.NewModelNotFoundError(providerName, err)
	case 400:
		// Anthropic uses 400 for various client errors.
		// Check the raw JSON for context length indicators.
		rawJSON := apiErr.RawJSON()
		if strings.Contains(rawJSON, errorPatternContextLength) || strings.Contains(rawJSON, errorPatternToken) {
			return errors.NewContextLengthError(providerName, err)
		}
		return errors.NewInvalidRequestError(providerName, err)
	case 403:
		// Forbidden - could be content filter or permission issue.
		rawJSON := apiErr.RawJSON()
		if strings.Contains(rawJSON, errorPatternContent) || strings.Contains(rawJSON, errorPatternSafety) {
			return errors.NewContentFilterError(providerName, err)
		}
		return errors.NewAuthenticationError(providerName, err)
	default:
		return errors.NewProviderError(providerName, err)
	}
}
