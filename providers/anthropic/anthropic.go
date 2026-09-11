// Package anthropic provides an Anthropic provider implementation for any-llm.
package anthropic

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"maps"
	"net/http"
	"reflect"
	"strconv"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/shared"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	defaultMaxTokens     = 4096
	envAPIKey            = "ANTHROPIC_API_KEY"
	envBaseURL           = "ANTHROPIC_BASE_URL"
	providerName         = "anthropic"
	thinkingBudgetLow    = 1024
	thinkingBudgetMedium = 4096
	thinkingBudgetHigh   = 16384
)

// Anthropic content block types.
const (
	blockTypeText     = "text"
	blockTypeThinking = "thinking"
	blockTypeToolUse  = "tool_use"
	toolTypeFunction  = "function"
)

// Anthropic error response patterns (checked in raw JSON).
const (
	errorPatternContextLength = "context_length"
	errorPatternToken         = "token"
)

// Anthropic delta types.
const (
	deltaTypeInputJSON = "input_json_delta"
	deltaTypeText      = "text_delta"
	deltaTypeThinking  = "thinking_delta"
)

// Anthropic streaming event types.
const (
	eventContentBlockDelta = "content_block_delta"
	eventContentBlockStart = "content_block_start"
	eventMessageDelta      = "message_delta"
	eventMessageStart      = "message_start"
	eventMessageStop       = "message_stop"
)

// Anthropic stop reasons.
const (
	stopReasonEndTurn      = "end_turn"
	stopReasonMaxTokens    = "max_tokens"
	stopReasonStopSequence = "stop_sequence"
	stopReasonToolUse      = "tool_use"
)

// Response format types.
const (
	responseFormatJSONObject = "json_object"
	responseFormatJSONSchema = "json_schema"
)

var errStreamEndedBeforeMessageStop = stderrors.New("stream ended before message_stop")

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
	messageID  string
	model      string
	inputUsage int64
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

	baseURL, err := cfg.ResolveBaseURL(envBaseURL, "")
	if err != nil {
		return nil, fmt.Errorf("resolve base URL: %w", err)
	}

	clientOpts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(cfg.HTTPClient()),
	}

	if baseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(baseURL))
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
		CompletionImage:     true,
		CompletionPDF:       true,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           false,
		ListModels:          false,
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

	resp, err := p.client.Messages.New(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	completion, err := convertResponse(resp)
	if err != nil {
		return nil, err
	}

	return completion, nil
}

// convertParams converts providers.CompletionParams to Anthropic request parameters.
func (p *Provider) convertParams(params providers.CompletionParams) (anthropic.MessageNewParams, error) {
	messages, system, err := convertMessages(params.Messages)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}

	maxTokens := int64(defaultMaxTokens)
	if params.MaxTokens != nil {
		maxTokens = int64(*params.MaxTokens)
	}

	req := anthropic.MessageNewParams{
		Model:     params.Model,
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
			converted, convertErr := convertTool(tool)
			if convertErr != nil {
				return anthropic.MessageNewParams{}, convertErr
			}
			tools = append(tools, converted)
		}
		req.Tools = tools
	}

	// Parallel control belongs inside tool_choice even when callers leave the
	// choice at its default. With no tools, retain the service's none default.
	// https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use
	if params.ToolChoice != nil || (params.ParallelToolCalls != nil && len(params.Tools) > 0) {
		req.ToolChoice = convertToolChoice(params.ToolChoice, params.ParallelToolCalls)
	}

	applyResponseFormat(&req, params.ResponseFormat)

	err = applyThinking(&req, params.ReasoningEffort, params.Model)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}

	return req, nil
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

		stream := p.client.Messages.NewStreaming(ctx, req)
		defer func() { _ = stream.Close() }()

		var (
			state       streamState
			accumulated anthropic.Message
		)

		messageStopped := false

		for stream.Next() {
			event := stream.Current()
			if accumulateErr := accumulated.Accumulate(event); accumulateErr != nil {
				errs <- errors.NewProviderError(providerName, fmt.Errorf("accumulate stream: %w", accumulateErr))

				return
			}

			var chunk *providers.ChatCompletionChunk

			switch event.Type {
			case eventMessageStart:
				chunk = new(state.handleMessageStart(event.AsMessageStart()))

			case eventContentBlockStart:
				chunk = state.handleContentBlockStart(event.AsContentBlockStart())

			case eventContentBlockDelta:
				chunk = state.handleContentBlockDelta(event.AsContentBlockDelta(), &accumulated)

			case eventMessageDelta:
				chunk = new(state.handleMessageDelta(event.AsMessageDelta()))

			case eventMessageStop:
				messageStopped = true

				chunk, err = state.handleMessageStop(&accumulated)
				if err != nil {
					errs <- err

					return
				}
			}

			if chunk != nil && !sendChunk(ctx, chunks, *chunk) {
				errs <- ctx.Err()

				return
			}
		}

		if err := stream.Err(); err != nil {
			errs <- p.ConvertError(err)

			return
		}

		if !messageStopped {
			errs <- errors.NewProviderError(providerName, errStreamEndedBeforeMessageStop)
		}
	}()

	return chunks, errs
}

func (s *streamState) handleMessageStop(message *anthropic.Message) (*providers.ChatCompletionChunk, error) {
	reasoning, err := reasoningFromContent(message.Content)
	if err != nil {
		return nil, err
	}

	if reasoning == nil {
		return nil, nil
	}

	chunk := s.chunk(providers.ChunkDelta{Reasoning: &providers.Reasoning{
		ProviderRaw: reasoning.ProviderRaw,
		Provider:    reasoning.Provider,
	}})

	return &chunk, nil
}

func sendChunk(
	ctx context.Context,
	chunks chan<- providers.ChatCompletionChunk,
	chunk providers.ChatCompletionChunk,
) bool {
	select {
	case chunks <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
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
func (s *streamState) handleContentBlockDelta(
	event anthropic.ContentBlockDeltaEvent,
	message *anthropic.Message,
) *providers.ChatCompletionChunk {
	switch event.Delta.Type {
	case deltaTypeText:
		return s.handleTextDelta(event.Delta.Text)
	case deltaTypeThinking:
		return s.handleThinkingDelta(event.Delta.Thinking)
	case deltaTypeInputJSON:
		return s.handleInputJSONDelta(message.Content[event.Index].ID, event.Delta.PartialJSON)
	default:
		return nil
	}
}

// handleContentBlockStart processes a content_block_start event.
func (s *streamState) handleContentBlockStart(event anthropic.ContentBlockStartEvent) *providers.ChatCompletionChunk {
	if event.ContentBlock.Type != blockTypeToolUse {
		return nil
	}

	chunk := s.chunk(providers.ChunkDelta{ToolCalls: []providers.ToolCall{{
		ID:       event.ContentBlock.ID,
		Type:     toolTypeFunction,
		Function: providers.FunctionCall{Name: event.ContentBlock.Name},
	}}})

	return &chunk
}

// handleInputJSONDelta processes a tool input JSON delta and returns a chunk if applicable.
func (s *streamState) handleInputJSONDelta(toolID, partialJSON string) *providers.ChatCompletionChunk {
	if toolID == "" {
		return nil
	}

	// Consumers concatenate partial_json fragments; sending accumulated input
	// would duplicate every earlier fragment. Read only the indexed block identity.
	// https://platform.claude.com/docs/en/build-with-claude/streaming#input-json-delta
	chunk := s.chunk(providers.ChunkDelta{ToolCalls: []providers.ToolCall{{
		ID:       toolID,
		Type:     toolTypeFunction,
		Function: providers.FunctionCall{Arguments: partialJSON},
	}}})
	return &chunk
}

// handleMessageDelta processes the message_delta event carrying the finish reason and usage.
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
	s.model = event.Message.Model
	s.inputUsage = event.Message.Usage.InputTokens

	return s.chunk(providers.ChunkDelta{Role: providers.RoleAssistant})
}

// handleThinkingDelta processes a thinking delta and returns a chunk.
func (s *streamState) handleThinkingDelta(thinking string) *providers.ChatCompletionChunk {
	chunk := s.chunk(providers.ChunkDelta{
		Reasoning: &providers.Reasoning{Content: thinking},
	})
	return &chunk
}

// handleTextDelta processes a text delta and returns a chunk.
func (s *streamState) handleTextDelta(text string) *providers.ChatCompletionChunk {
	chunk := s.chunk(providers.ChunkDelta{Content: text})
	return &chunk
}

// applyThinking maps normalized reasoning controls according to each model's
// supported thinking mode and effort controls.
func applyThinking(req *anthropic.MessageNewParams, effort providers.ReasoningEffort, model string) error {
	switch effort {
	case "", providers.ReasoningEffortAuto:
		// Auto preserves the model's default. Anthropic enables adaptive thinking
		// by default for some current models but not others.
		// https://platform.claude.com/docs/en/build-with-claude/thinking
		return nil
	case providers.ReasoningEffortNone:
		// Always-on-thinking models reject disabled thinking. Let that error
		// reach the caller rather than silently enabling thinking for explicit none.
		// https://platform.claude.com/docs/en/build-with-claude/thinking
		req.Thinking = anthropic.ThinkingConfigParamUnion{
			OfDisabled: new(anthropic.NewThinkingConfigDisabledParam()),
		}

		return nil
	case providers.ReasoningEffortMinimal, providers.ReasoningEffortLow, providers.ReasoningEffortMedium,
		providers.ReasoningEffortHigh, providers.ReasoningEffortXHigh, providers.ReasoningEffortMax:
	default:
		return errors.NewUnsupportedParamError(providerName, "reasoning_effort="+string(effort))
	}

	if supportsAdaptiveThinking(model) {
		if effort == providers.ReasoningEffortMinimal {
			effort = providers.ReasoningEffortLow
		}
		req.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: new(anthropic.ThinkingConfigAdaptiveParam),
		}
		req.OutputConfig.Effort = anthropic.OutputConfigEffort(effort)

		return nil
	}

	budget, ok := legacyThinkingBudget(effort)
	if !ok {
		return errors.NewUnsupportedParamError(providerName, "reasoning_effort="+string(effort))
	}
	req.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
	req.MaxTokens = max(req.MaxTokens, budget*2)

	if modelFamily(model, "claude-opus-4-5") {
		req.OutputConfig.Effort = anthropic.OutputConfigEffort(effort)
	}

	return nil
}

// The support table distinguishes adaptive models from extended-thinking-only
// models. Effort alone does not enable thinking.
// https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting
func supportsAdaptiveThinking(model string) bool {
	for _, family := range []string{
		"claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-7", "claude-opus-4-8",
		"claude-opus-5", "claude-sonnet-5", "claude-mythos-5", "claude-mythos-5-1",
		"claude-fable-5", "claude-fable-5-1", "claude-mythos-preview",
	} {
		if modelFamily(model, family) {
			return true
		}
	}

	return false
}

func modelFamily(model, family string) bool {
	if model == family {
		return true
	}

	date, ok := strings.CutPrefix(model, family+"-")
	if !ok || len(date) != 8 {
		return false
	}
	for _, digit := range date {
		if digit < '0' || digit > '9' {
			return false
		}
	}

	return true
}

func legacyThinkingBudget(effort providers.ReasoningEffort) (int64, bool) {
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

// applyResponseFormat configures structured output on the request if applicable.
func applyResponseFormat(req *anthropic.MessageNewParams, format *providers.ResponseFormat) {
	if format == nil || format.JSONSchema == nil {
		return
	}
	switch format.Type {
	case responseFormatJSONSchema:
		// JSONOutputFormatParam only carries Schema and Type; Name, Description, and Strict
		// from providers.JSONSchema are not supported by the Anthropic API.
		req.OutputConfig = anthropic.OutputConfigParam{
			Format: anthropic.JSONOutputFormatParam{
				Schema: format.JSONSchema.Schema,
			},
		}
	case responseFormatJSONObject:
		// Anthropic requires a schema for structured output; json_object without a schema
		// is not supported. No-op to preserve forward compatibility.
	}
}

// convertAssistantMessage converts an assistant message to Anthropic format.
func convertAssistantMessage(msg providers.Message) (*anthropic.MessageParam, error) {
	if msg.Reasoning != nil && msg.Reasoning.Provider == providerName && len(msg.Reasoning.ProviderRaw) > 0 {
		content, err := replayAssistantContent(msg)
		if err != nil {
			return nil, err
		}

		m := anthropic.NewAssistantMessage(content...)

		return &m, nil
	}

	if len(msg.ToolCalls) == 0 {
		m := anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.ContentString()))

		return &m, nil
	}

	content := make([]anthropic.ContentBlockParamUnion, 0)
	if msg.ContentString() != "" {
		content = append(content, anthropic.NewTextBlock(msg.ContentString()))
	}

	for _, tc := range msg.ToolCalls {
		toolCall, err := convertToolCall(tc)
		if err != nil {
			return nil, err
		}

		content = append(content, toolCall)
	}

	m := anthropic.NewAssistantMessage(content...)

	return &m, nil
}

func replayAssistantContent(msg providers.Message) ([]anthropic.ContentBlockParamUnion, error) {
	var rawBlocks []json.RawMessage
	if err := json.Unmarshal(msg.Reasoning.ProviderRaw, &rawBlocks); err != nil {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("invalid reasoning provider_raw: %w", err))
	}

	blocks := make([]anthropic.ContentBlockUnion, len(rawBlocks))
	for i, raw := range rawBlocks {
		if err := validateReplayBlock(raw); err != nil {
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("invalid reasoning provider_raw block %d: %w", i, err),
			)
		}

		if err := json.Unmarshal(raw, &blocks[i]); err != nil {
			return nil, errors.NewInvalidRequestError(
				providerName,
				fmt.Errorf("invalid reasoning provider_raw block %d: %w", i, err),
			)
		}
	}

	projectedContent, projectedReasoning, projectedTools, hasReplayBlocks := projectContent(blocks)

	content, ok := msg.Content.(string)
	if msg.Content == nil {
		content = ""
		ok = true
	}

	if !ok || !hasReplayBlocks || content != projectedContent ||
		msg.Reasoning.Content != projectedReasoning || !replayToolCallsEqual(msg.ToolCalls, projectedTools) {
		return nil, errors.NewInvalidRequestError(
			providerName,
			stderrors.New("reasoning provider_raw does not match assistant content, reasoning, and tool calls"),
		)
	}

	params := make([]anthropic.ContentBlockParamUnion, len(blocks))
	for i, raw := range rawBlocks {
		params[i] = param.Override[anthropic.ContentBlockParamUnion](raw)
	}

	return params, nil
}

func replayToolCallsEqual(actual, projected []providers.ToolCall) bool {
	if len(actual) != len(projected) {
		return false
	}

	for i := range actual {
		if actual[i].ID != projected[i].ID || actual[i].Type != projected[i].Type ||
			actual[i].Function.Name != projected[i].Function.Name ||
			!reflect.DeepEqual(actual[i].Extra, projected[i].Extra) ||
			!equalJSON(actual[i].Function.Arguments, projected[i].Function.Arguments) {
			return false
		}
	}

	return true
}

func equalJSON(left, right string) bool {
	decode := func(value string) (any, bool) {
		if !json.Valid([]byte(value)) {
			return nil, false
		}

		decoder := json.NewDecoder(strings.NewReader(value))
		decoder.UseNumber()

		var decoded any
		if err := decoder.Decode(&decoded); err != nil {
			return nil, false
		}

		return decoded, true
	}

	decodedLeft, leftOK := decode(left)
	decodedRight, rightOK := decode(right)

	return leftOK && rightOK && reflect.DeepEqual(decodedLeft, decodedRight)
}

func validateReplayBlock(raw json.RawMessage) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}

	blockType, err := requiredJSONString(fields, "type")
	if err != nil {
		return err
	}

	switch blockType {
	case blockTypeText:
		_, err = requiredJSONString(fields, "text")
	case blockTypeThinking:
		if _, err = requiredJSONString(fields, "thinking"); err == nil {
			_, err = requiredJSONString(fields, "signature")
		}
	case "redacted_thinking":
		_, err = requiredJSONString(fields, "data")
	case blockTypeToolUse:
		if _, err = requiredJSONString(fields, "id"); err == nil {
			_, err = requiredJSONString(fields, "name")
		}

		if err == nil {
			if input, ok := fields["input"]; !ok || !json.Valid(input) {
				err = stderrors.New("field input is missing or invalid")
			}
		}
	default:
		return fmt.Errorf("unsupported content block type %q", blockType)
	}

	return err
}

func requiredJSONString(fields map[string]json.RawMessage, name string) (string, error) {
	raw, ok := fields[name]
	if !ok {
		return "", fmt.Errorf("field %s is missing", name)
	}

	var value *string
	if err := json.Unmarshal(raw, &value); err != nil || value == nil {
		return "", fmt.Errorf("field %s must be a string", name)
	}

	return *value, nil
}

// convertImagePart converts an image URL to Anthropic format.
func convertImagePart(img *providers.ImageURL) (anthropic.ContentBlockParamUnion, error) {
	if img == nil {
		return anthropic.ContentBlockParamUnion{}, errors.NewInvalidRequestError(
			providerName, stderrors.New("image content is missing image_url"),
		)
	}

	if img.Detail != "" {
		// Anthropic image blocks have no OpenAI-style detail control.
		// https://platform.claude.com/docs/en/api/messages/create#body-messages-content-source
		return anthropic.ContentBlockParamUnion{}, errors.NewUnsupportedParamError(providerName, "image_url.detail")
	}

	dataURL, ok := strings.CutPrefix(img.URL, "data:")
	if !ok {
		return anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: img.URL}), nil
	}

	header, data, ok := strings.Cut(dataURL, ",")
	if !ok || !strings.HasSuffix(header, ";base64") {
		return anthropic.ContentBlockParamUnion{}, errors.NewInvalidRequestError(
			providerName, stderrors.New("invalid base64 image data URL"),
		)
	}

	// Validate only enough to choose the source variant; the service validates
	// the media type and payload. Never send a malformed data URL as a remote URL.
	mediaType, _, _ := strings.Cut(header, ";")

	return anthropic.NewImageBlockBase64(mediaType, data), nil
}

// convertMessage converts a single message to Anthropic format.
func convertMessage(msg providers.Message) (*anthropic.MessageParam, error) {
	switch msg.Role {
	case providers.RoleUser:
		return convertUserMessage(msg)
	case providers.RoleAssistant:
		return convertAssistantMessage(msg)
	case providers.RoleTool:
		return convertToolMessage(msg)
	default:
		return nil, errors.NewInvalidRequestError(
			providerName,
			fmt.Errorf("unsupported message role %q", msg.Role),
		)
	}
}

// convertMessages converts providers messages to Anthropic format.
// Returns the messages and the combined system message.
func convertMessages(messages []providers.Message) ([]anthropic.MessageParam, string, error) {
	result := make([]anthropic.MessageParam, 0, len(messages))
	var systemParts []string

	for _, msg := range messages {
		if msg.Role == providers.RoleSystem {
			// Native positional system roles have model and placement restrictions.
			// Preserve normalized system instructions in the compatible top-level field.
			// https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages
			systemParts = append(systemParts, msg.ContentString())
			continue
		}

		converted, err := convertMessage(msg)
		if err != nil {
			return nil, "", err
		}

		result = append(result, *converted)
	}

	return result, strings.Join(systemParts, "\n"), nil
}

// convertResponse converts an Anthropic response to providers format.
func convertResponse(resp *anthropic.Message) (*providers.ChatCompletion, error) {
	content, _, toolCalls, _ := projectContent(resp.Content)

	reasoning, err := reasoningFromContent(resp.Content)
	if err != nil {
		return nil, errors.NewProviderError(providerName, err)
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
		Model:  resp.Model,
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
	}, nil
}

func reasoningFromContent(blocks []anthropic.ContentBlockUnion) (*providers.Reasoning, error) {
	_, reasoning, _, hasReplayBlocks := projectContent(blocks)
	if !hasReplayBlocks {
		return nil, nil
	}

	raw, err := marshalContentSnapshot(blocks)
	if err != nil {
		return nil, err
	}

	return &providers.Reasoning{Content: reasoning, ProviderRaw: raw, Provider: providerName}, nil
}

func marshalContentSnapshot(blocks []anthropic.ContentBlockUnion) (json.RawMessage, error) {
	rawBlocks := make([]json.RawMessage, len(blocks))
	for i, block := range blocks {
		raw := json.RawMessage(block.RawJSON())
		if !json.Valid(raw) {
			return nil, fmt.Errorf("content block %d has no valid raw JSON", i)
		}

		rawBlocks[i] = raw
	}

	return json.Marshal(rawBlocks)
}

func projectContent(blocks []anthropic.ContentBlockUnion) (string, string, []providers.ToolCall, bool) {
	var (
		content   strings.Builder
		reasoning strings.Builder
		toolCalls []providers.ToolCall
	)

	hasReplayBlocks := false

	for _, block := range blocks {
		switch block.Type {
		case blockTypeText:
			content.WriteString(block.Text)
		case blockTypeThinking:
			hasReplayBlocks = true

			reasoning.WriteString(block.Thinking)
		case "redacted_thinking":
			hasReplayBlocks = true
		case blockTypeToolUse:
			inputJSON := ""

			if block.Input != nil {
				if inputBytes, err := json.Marshal(block.Input); err == nil {
					inputJSON = string(inputBytes)
				}
			}

			toolCalls = append(toolCalls, providers.ToolCall{
				ID: block.ID, Type: toolTypeFunction,
				Function: providers.FunctionCall{Name: block.Name, Arguments: inputJSON},
			})
		}
	}

	return content.String(), reasoning.String(), toolCalls, hasReplayBlocks
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
func convertTool(tool providers.Tool) (anthropic.ToolUnionParam, error) {
	parameters := tool.Function.Parameters
	if required, ok := parameters["required"]; ok {
		if _, err := toStringSlice(required); err != nil {
			return anthropic.ToolUnionParam{}, fmt.Errorf(
				"tool %s: invalid required field: %w",
				tool.Function.Name,
				err,
			)
		}
	}

	if _, ok := parameters["type"]; !ok {
		withType := map[string]any{"type": "object"}
		maps.Copy(withType, parameters)
		parameters = withType
	}

	encoded, err := json.Marshal(parameters)
	if err != nil {
		return anthropic.ToolUnionParam{}, fmt.Errorf("tool %s: encode input schema: %w", tool.Function.Name, err)
	}

	// Preserve the complete schema rather than selecting named SDK fields.
	// Raw JSON also keeps json.Number numeric with SDK encoders that quote it.
	// https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools
	// https://pkg.go.dev/github.com/anthropics/anthropic-sdk-go/packages/param#Override
	return anthropic.ToolUnionParam{
		OfTool: &anthropic.ToolParam{
			Name:        tool.Function.Name,
			Description: anthropic.String(tool.Function.Description),
			InputSchema: param.Override[anthropic.ToolInputSchemaParam](json.RawMessage(encoded)),
		},
	}, nil
}

// convertToolCall converts a tool call to Anthropic content block format.
func convertToolCall(toolCall providers.ToolCall) (anthropic.ContentBlockParamUnion, error) {
	var input map[string]json.RawMessage
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil || input == nil {
		return anthropic.ContentBlockParamUnion{}, errors.NewInvalidRequestError(
			providerName,
			stderrors.New("tool input must be a JSON object"),
		)
	}

	return anthropic.ContentBlockParamUnion{
		OfToolUse: &anthropic.ToolUseBlockParam{
			Type: blockTypeToolUse,
			ID:   toolCall.ID,
			Name: toolCall.Function.Name,
			// RawMessage preserves JSON integers that map[string]any would round
			// through float64 after validating the required object shape above.
			// https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
			Input: json.RawMessage(toolCall.Function.Arguments),
		},
	}, nil
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
func convertToolMessage(msg providers.Message) (*anthropic.MessageParam, error) {
	if msg.ToolCallID == "" {
		return nil, errors.NewInvalidRequestError(
			providerName,
			stderrors.New("tool result requires tool_call_id"),
		)
	}

	// Anthropic uses is_error to let Claude recover from client tool failures.
	// https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls#handling-errors-with-is_error
	m := anthropic.NewUserMessage(
		anthropic.NewToolResultBlock(msg.ToolCallID, msg.ContentString(), msg.ToolResultIsError),
	)

	return &m, nil
}

// convertUserMessage converts a user message to Anthropic format.
func convertUserMessage(msg providers.Message) (*anthropic.MessageParam, error) {
	if !msg.IsMultiModal() {
		m := anthropic.NewUserMessage(anthropic.NewTextBlock(msg.ContentString()))

		return &m, nil
	}

	content := make([]anthropic.ContentBlockParamUnion, 0)
	for _, part := range msg.ContentParts() {
		switch part.Type {
		case blockTypeText:
			content = append(content, anthropic.NewTextBlock(part.Text))
		case "image_url":
			image, err := convertImagePart(part.ImageURL)
			if err != nil {
				return nil, err
			}

			content = append(content, image)
		default:
			return nil, errors.NewUnsupportedParamError(providerName, "messages.content.type="+part.Type)
		}
	}
	m := anthropic.NewUserMessage(content...)

	return &m, nil
}

// toStringSlice converts a value to []string.
// Accepts []string (returned as-is) or []any (each element must be string).
func toStringSlice(v any) ([]string, error) {
	switch typed := v.(type) {
	case []string:
		return typed, nil
	case []any:
		result := make([]string, len(typed))
		for i, elem := range typed {
			s, ok := elem.(string)
			if !ok {
				return nil, fmt.Errorf("element %d: expected string, got %T", i, elem)
			}
			result[i] = s
		}
		return result, nil
	default:
		return nil, fmt.Errorf("expected []string or []any, got %T", v)
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
	apiErr, ok := stderrors.AsType[*anthropic.Error](err)
	if !ok {
		return errors.NewProviderError(providerName, err)
	}

	// The documented error schema provides no context-overflow discriminator.
	// Preserve the historical message classifier for invalid requests because
	// callers use ErrContextLength to trigger history trimming.
	// https://platform.claude.com/docs/en/api/errors#error-shapes
	switch apiErr.Type() {
	case shared.ErrorTypeAuthenticationError:
		return errors.NewAuthenticationError(providerName, err)
	case shared.ErrorTypeBillingError:
		return errors.NewInsufficientFundsError(providerName, err)
	case shared.ErrorTypeRateLimitError:
		return anthropicRateLimitError(apiErr, err)
	case shared.ErrorTypeInvalidRequestError:
		return anthropicInvalidRequestError(apiErr, err)
	case shared.ErrorTypeNotFoundError:
		// Preserve the historical classification even though 404 can name other resources.
		return errors.NewModelNotFoundError(providerName, err)
	case shared.ErrorTypePermissionError,
		shared.ErrorTypeTimeoutError,
		shared.ErrorTypeOverloadedError,
		shared.ErrorTypeAPIError:
		return errors.NewProviderError(providerName, err)
	}

	// Preserve status-based compatibility when the body is missing or unknown.
	// Permission failures and byte-size limits remain generic provider errors.
	switch apiErr.StatusCode {
	case http.StatusUnauthorized:
		return errors.NewAuthenticationError(providerName, err)
	case http.StatusPaymentRequired:
		return errors.NewInsufficientFundsError(providerName, err)
	case http.StatusNotFound:
		return errors.NewModelNotFoundError(providerName, err)
	case http.StatusTooManyRequests:
		return anthropicRateLimitError(apiErr, err)
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return anthropicInvalidRequestError(apiErr, err)
	default:
		return errors.NewProviderError(providerName, err)
	}
}

func anthropicInvalidRequestError(apiErr *anthropic.Error, original error) error {
	// Check the raw JSON for context length indicators.
	rawJSON := apiErr.RawJSON()
	if strings.Contains(rawJSON, errorPatternContextLength) || strings.Contains(rawJSON, errorPatternToken) {
		return errors.NewContextLengthError(providerName, original)
	}

	return errors.NewInvalidRequestError(providerName, original)
}

func anthropicRateLimitError(apiErr *anthropic.Error, original error) *errors.RateLimitError {
	converted := errors.NewRateLimitError(providerName, original)
	if apiErr.Response == nil {
		return converted
	}

	// Anthropic omits Retry-After for spend-cap 429 responses. Preserve zero
	// when the documented integer-seconds header is absent or invalid.
	// https://platform.claude.com/docs/en/api/rate-limits#response-headers
	seconds, err := strconv.Atoi(apiErr.Response.Header.Get("Retry-After"))
	if err == nil && seconds >= 0 {
		converted.RetryAfter = seconds
	}

	return converted
}
