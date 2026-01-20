// Package anthropic provides an Anthropic provider implementation for any-llm.
package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const (
	providerName     = "anthropic"
	envAPIKey        = "ANTHROPIC_API_KEY"
	defaultMaxTokens = 4096
)

// Anthropic streaming event types.
const (
	eventMessageStart      = "message_start"
	eventContentBlockStart = "content_block_start"
	eventContentBlockDelta = "content_block_delta"
	eventMessageDelta      = "message_delta"
)

// Anthropic content block types.
const (
	blockTypeText     = "text"
	blockTypeThinking = "thinking"
	blockTypeToolUse  = "tool_use"
)

// Anthropic delta types.
const (
	deltaTypeText      = "text_delta"
	deltaTypeThinking  = "thinking_delta"
	deltaTypeInputJSON = "input_json_delta"
)

// Anthropic stop reasons.
const (
	stopReasonEndTurn      = "end_turn"
	stopReasonMaxTokens    = "max_tokens"
	stopReasonToolUse      = "tool_use"
	stopReasonStopSequence = "stop_sequence"
)

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

// Provider implements the providers.Provider interface for Anthropic.
type Provider struct {
	client *anthropic.Client
	config *config.Config
}

// Ensure Provider implements the required interfaces.
var (
	_ providers.Provider           = (*Provider)(nil)
	_ providers.CapabilityProvider = (*Provider)(nil)
)

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

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
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
func (p *Provider) Completion(ctx context.Context, params providers.CompletionParams) (*providers.ChatCompletion, error) {
	req := p.convertParams(params)

	resp, err := p.client.Messages.New(ctx, req)
	if err != nil {
		return nil, errors.Convert(providerName, err)
	}

	return convertResponse(resp), nil
}

// CompletionStream performs a streaming chat completion request.
func (p *Provider) CompletionStream(ctx context.Context, params providers.CompletionParams) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		req := p.convertParams(params)

		stream := p.client.Messages.NewStreaming(ctx, req)

		// Track accumulated state for streaming.
		var (
			messageID        string
			model            string
			contentBuilder   strings.Builder
			reasoningBuilder strings.Builder
			toolCalls        []providers.ToolCall
			currentToolIdx   = -1
			inputUsage       int64
		)

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case eventMessageStart:
				e := event.AsMessageStart()
				messageID = e.Message.ID
				model = string(e.Message.Model)
				inputUsage = e.Message.Usage.InputTokens

				// Send initial chunk.
				chunks <- providers.ChatCompletionChunk{
					ID:     messageID,
					Object: "chat.completion.chunk",
					Model:  model,
					Choices: []providers.ChunkChoice{{
						Index: 0,
						Delta: providers.ChunkDelta{
							Role: providers.RoleAssistant,
						},
					}},
				}

			case eventContentBlockStart:
				e := event.AsContentBlockStart()
				switch e.ContentBlock.Type {
				case blockTypeThinking:
					// Reasoning block started.
				case blockTypeToolUse:
					currentToolIdx++
					toolCalls = append(toolCalls, providers.ToolCall{
						ID:   e.ContentBlock.ID,
						Type: "function",
						Function: providers.FunctionCall{
							Name: e.ContentBlock.Name,
						},
					})
				}

			case eventContentBlockDelta:
				e := event.AsContentBlockDelta()
				switch e.Delta.Type {
				case deltaTypeText:
					text := e.Delta.Text
					contentBuilder.WriteString(text)
					chunks <- providers.ChatCompletionChunk{
						ID:     messageID,
						Object: "chat.completion.chunk",
						Model:  model,
						Choices: []providers.ChunkChoice{{
							Index: 0,
							Delta: providers.ChunkDelta{
								Content: text,
							},
						}},
					}

				case deltaTypeThinking:
					thinking := e.Delta.Thinking
					reasoningBuilder.WriteString(thinking)
					chunks <- providers.ChatCompletionChunk{
						ID:     messageID,
						Object: "chat.completion.chunk",
						Model:  model,
						Choices: []providers.ChunkChoice{{
							Index: 0,
							Delta: providers.ChunkDelta{
								Reasoning: &providers.Reasoning{
									Content: thinking,
								},
							},
						}},
					}

				case deltaTypeInputJSON:
					if currentToolIdx >= 0 && currentToolIdx < len(toolCalls) {
						toolCalls[currentToolIdx].Function.Arguments += e.Delta.PartialJSON
						chunks <- providers.ChatCompletionChunk{
							ID:     messageID,
							Object: "chat.completion.chunk",
							Model:  model,
							Choices: []providers.ChunkChoice{{
								Index: 0,
								Delta: providers.ChunkDelta{
									ToolCalls: []providers.ToolCall{toolCalls[currentToolIdx]},
								},
							}},
						}
					}
				}

			case eventMessageDelta:
				e := event.AsMessageDelta()
				finishReason := convertStopReason(string(e.Delta.StopReason))
				chunks <- providers.ChatCompletionChunk{
					ID:     messageID,
					Object: "chat.completion.chunk",
					Model:  model,
					Choices: []providers.ChunkChoice{{
						Index:        0,
						FinishReason: finishReason,
					}},
					Usage: &providers.Usage{
						PromptTokens:     int(inputUsage),
						CompletionTokens: int(e.Usage.OutputTokens),
						TotalTokens:      int(inputUsage + e.Usage.OutputTokens),
					},
				}
			}
		}

		if err := stream.Err(); err != nil {
			errs <- errors.Convert(providerName, err)
		}
	}()

	return chunks, errs
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

	// Handle reasoning/thinking.
	if params.ReasoningEffort != "" && params.ReasoningEffort != providers.ReasoningEffortNone {
		if budget, ok := thinkingBudget(params.ReasoningEffort); ok {
			req.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
			// Increase max tokens to accommodate thinking.
			if maxTokens < budget*2 {
				req.MaxTokens = budget * 2
			}
		}
	}

	return req
}

// convertMessages converts providers messages to Anthropic format.
// Returns the messages and the combined system message.
func convertMessages(messages []providers.Message) ([]anthropic.MessageParam, string) {
	result := make([]anthropic.MessageParam, 0, len(messages))
	var systemParts []string

	for _, msg := range messages {
		switch msg.Role {
		case providers.RoleSystem:
			systemParts = append(systemParts, msg.GetContentString())

		case providers.RoleUser:
			if msg.IsMultiModal() {
				content := make([]anthropic.ContentBlockParamUnion, 0)
				for _, part := range msg.GetContentParts() {
					if part.Type == "text" {
						content = append(content, anthropic.NewTextBlock(part.Text))
					} else if part.Type == "image_url" && part.ImageURL != nil {
						content = append(content, convertImagePart(part.ImageURL))
					}
				}
				result = append(result, anthropic.NewUserMessage(content...))
			} else {
				result = append(result, anthropic.NewUserMessage(
					anthropic.NewTextBlock(msg.GetContentString()),
				))
			}

		case providers.RoleAssistant:
			if len(msg.ToolCalls) > 0 {
				content := make([]anthropic.ContentBlockParamUnion, 0)
				if msg.GetContentString() != "" {
					content = append(content, anthropic.NewTextBlock(msg.GetContentString()))
				}
				for _, tc := range msg.ToolCalls {
					// Parse arguments JSON to map.
					var input map[string]interface{}
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &input) // Ignore error: use empty map on failure.
					content = append(content, anthropic.ContentBlockParamUnion{
						OfToolUse: &anthropic.ToolUseBlockParam{
							Type:  "tool_use",
							ID:    tc.ID,
							Name:  tc.Function.Name,
							Input: input,
						},
					})
				}
				result = append(result, anthropic.NewAssistantMessage(content...))
			} else {
				result = append(result, anthropic.NewAssistantMessage(
					anthropic.NewTextBlock(msg.GetContentString()),
				))
			}

		case providers.RoleTool:
			result = append(result, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(msg.ToolCallID, msg.GetContentString(), false),
			))
		}
	}

	return result, strings.Join(systemParts, "\n")
}

// convertImagePart converts an image URL to Anthropic format.
func convertImagePart(img *providers.ImageURL) anthropic.ContentBlockParamUnion {
	url := img.URL

	// Check if it's a base64 data URL.
	if strings.HasPrefix(url, "data:") {
		// Parse data URL: data:image/jpeg;base64,<data>
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

// convertTool converts a providers.Tool to Anthropic format.
func convertTool(tool providers.Tool) anthropic.ToolUnionParam {
	inputSchema := anthropic.ToolInputSchemaParam{
		Type: "object",
	}
	if props, ok := tool.Function.Parameters["properties"]; ok {
		inputSchema.Properties = props
	}
	if req, ok := tool.Function.Parameters["required"]; ok {
		if reqArr, ok := req.([]interface{}); ok {
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
			// Anthropic has "none" now.
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
