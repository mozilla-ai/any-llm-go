// Package anthropic provides an Anthropic provider implementation for any-llm.
package anthropic

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

const (
	providerName     = "anthropic"
	envAPIKey        = "ANTHROPIC_API_KEY"
	defaultMaxTokens = 4096
)

// Reasoning effort to thinking budget tokens mapping.
var reasoningEffortToBudget = map[anyllm.ReasoningEffort]int64{
	anyllm.ReasoningEffortLow:    1024,
	anyllm.ReasoningEffortMedium: 4096,
	anyllm.ReasoningEffortHigh:   16384,
}

// Provider implements the anyllm.Provider interface for Anthropic.
type Provider struct {
	client *anthropic.Client
	config *anyllm.Config
}

// Ensure Provider implements the required interfaces.
var (
	_ anyllm.Provider           = (*Provider)(nil)
	_ anyllm.CapabilityProvider = (*Provider)(nil)
)

// New creates a new Anthropic provider.
func New(opts ...anyllm.Option) (*Provider, error) {
	cfg := anyllm.DefaultConfig()
	cfg.ApplyOptions(opts...)

	apiKey := cfg.GetAPIKeyFromEnv(envAPIKey)
	if apiKey == "" {
		return nil, anyllm.NewMissingAPIKeyError(providerName, envAPIKey)
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
func (p *Provider) Capabilities() anyllm.ProviderCapabilities {
	return anyllm.ProviderCapabilities{
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
func (p *Provider) Completion(ctx context.Context, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
	req := p.convertParams(params)

	resp, err := p.client.Messages.New(ctx, req)
	if err != nil {
		return nil, anyllm.ConvertError(providerName, err)
	}

	return convertResponse(resp), nil
}

// CompletionStream performs a streaming chat completion request.
func (p *Provider) CompletionStream(ctx context.Context, params anyllm.CompletionParams) (<-chan anyllm.ChatCompletionChunk, <-chan error) {
	chunks := make(chan anyllm.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		req := p.convertParams(params)

		stream := p.client.Messages.NewStreaming(ctx, req)

		// Track accumulated state for streaming
		var (
			messageID        string
			model            string
			contentBuilder   strings.Builder
			reasoningBuilder strings.Builder
			toolCalls        []anyllm.ToolCall
			currentToolIdx   = -1
			inputUsage       int64
		)

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "message_start":
				e := event.AsMessageStart()
				messageID = e.Message.ID
				model = string(e.Message.Model)
				inputUsage = e.Message.Usage.InputTokens

				// Send initial chunk
				chunks <- anyllm.ChatCompletionChunk{
					ID:     messageID,
					Object: "chat.completion.chunk",
					Model:  model,
					Choices: []anyllm.ChunkChoice{{
						Index: 0,
						Delta: anyllm.ChunkDelta{
							Role: anyllm.RoleAssistant,
						},
					}},
				}

			case "content_block_start":
				e := event.AsContentBlockStart()
				switch e.ContentBlock.Type {
				case "thinking":
					// Reasoning block started
				case "tool_use":
					currentToolIdx++
					toolCalls = append(toolCalls, anyllm.ToolCall{
						ID:   e.ContentBlock.ID,
						Type: "function",
						Function: anyllm.FunctionCall{
							Name: e.ContentBlock.Name,
						},
					})
				}

			case "content_block_delta":
				e := event.AsContentBlockDelta()
				switch e.Delta.Type {
				case "text_delta":
					text := e.Delta.Text
					contentBuilder.WriteString(text)
					chunks <- anyllm.ChatCompletionChunk{
						ID:     messageID,
						Object: "chat.completion.chunk",
						Model:  model,
						Choices: []anyllm.ChunkChoice{{
							Index: 0,
							Delta: anyllm.ChunkDelta{
								Content: text,
							},
						}},
					}

				case "thinking_delta":
					thinking := e.Delta.Thinking
					reasoningBuilder.WriteString(thinking)
					chunks <- anyllm.ChatCompletionChunk{
						ID:     messageID,
						Object: "chat.completion.chunk",
						Model:  model,
						Choices: []anyllm.ChunkChoice{{
							Index: 0,
							Delta: anyllm.ChunkDelta{
								Reasoning: &anyllm.Reasoning{
									Content: thinking,
								},
							},
						}},
					}

				case "input_json_delta":
					if currentToolIdx >= 0 && currentToolIdx < len(toolCalls) {
						toolCalls[currentToolIdx].Function.Arguments += e.Delta.PartialJSON
						chunks <- anyllm.ChatCompletionChunk{
							ID:     messageID,
							Object: "chat.completion.chunk",
							Model:  model,
							Choices: []anyllm.ChunkChoice{{
								Index: 0,
								Delta: anyllm.ChunkDelta{
									ToolCalls: []anyllm.ToolCall{toolCalls[currentToolIdx]},
								},
							}},
						}
					}
				}

			case "message_delta":
				e := event.AsMessageDelta()
				finishReason := convertStopReason(string(e.Delta.StopReason))
				chunks <- anyllm.ChatCompletionChunk{
					ID:     messageID,
					Object: "chat.completion.chunk",
					Model:  model,
					Choices: []anyllm.ChunkChoice{{
						Index:        0,
						FinishReason: finishReason,
					}},
					Usage: &anyllm.Usage{
						PromptTokens:     int(inputUsage),
						CompletionTokens: int(e.Usage.OutputTokens),
						TotalTokens:      int(inputUsage + e.Usage.OutputTokens),
					},
				}
			}
		}

		if err := stream.Err(); err != nil {
			errs <- anyllm.ConvertError(providerName, err)
		}
	}()

	return chunks, errs
}

// convertParams converts anyllm.CompletionParams to Anthropic request parameters.
func (p *Provider) convertParams(params anyllm.CompletionParams) anthropic.MessageNewParams {
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

	// Handle reasoning/thinking
	if params.ReasoningEffort != "" && params.ReasoningEffort != anyllm.ReasoningEffortNone {
		if budget, ok := reasoningEffortToBudget[params.ReasoningEffort]; ok {
			req.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
			// Increase max tokens to accommodate thinking
			if maxTokens < budget*2 {
				req.MaxTokens = budget * 2
			}
		}
	}

	return req
}

// convertMessages converts anyllm messages to Anthropic format.
// Returns the messages and the combined system message.
func convertMessages(messages []anyllm.Message) ([]anthropic.MessageParam, string) {
	result := make([]anthropic.MessageParam, 0, len(messages))
	var systemParts []string

	for _, msg := range messages {
		switch msg.Role {
		case anyllm.RoleSystem:
			systemParts = append(systemParts, msg.GetContentString())

		case anyllm.RoleUser:
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

		case anyllm.RoleAssistant:
			if len(msg.ToolCalls) > 0 {
				content := make([]anthropic.ContentBlockParamUnion, 0)
				if msg.GetContentString() != "" {
					content = append(content, anthropic.NewTextBlock(msg.GetContentString()))
				}
				for _, tc := range msg.ToolCalls {
					// Parse arguments JSON to map
					var input map[string]interface{}
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &input) // Ignore error: use empty map on failure
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

		case anyllm.RoleTool:
			result = append(result, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(msg.ToolCallID, msg.GetContentString(), false),
			))
		}
	}

	return result, strings.Join(systemParts, "\n")
}

// convertImagePart converts an image URL to Anthropic format.
func convertImagePart(img *anyllm.ImageURL) anthropic.ContentBlockParamUnion {
	url := img.URL

	// Check if it's a base64 data URL
	if strings.HasPrefix(url, "data:") {
		// Parse data URL: data:image/jpeg;base64,<data>
		parts := strings.SplitN(url, ",", 2)
		if len(parts) == 2 {
			// Extract media type from the first part
			mediaTypePart := strings.TrimPrefix(parts[0], "data:")
			mediaType := strings.Split(mediaTypePart, ";")[0]
			data := parts[1]

			return anthropic.NewImageBlockBase64(mediaType, data)
		}
	}

	// Regular URL
	return anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: url})
}

// convertTool converts an anyllm.Tool to Anthropic format.
func convertTool(tool anyllm.Tool) anthropic.ToolUnionParam {
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

// convertToolChoice converts anyllm tool choice to Anthropic format.
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
			// Anthropic has "none" now
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
	case anyllm.ToolChoice:
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

// convertResponse converts an Anthropic response to anyllm format.
func convertResponse(resp *anthropic.Message) *anyllm.ChatCompletion {
	var content string
	var reasoning *anyllm.Reasoning
	var toolCalls []anyllm.ToolCall

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content += block.Text
		case "thinking":
			reasoning = &anyllm.Reasoning{
				Content: block.Thinking,
			}
		case "tool_use":
			inputJSON := ""
			if block.Input != nil {
				if inputBytes, err := json.Marshal(block.Input); err == nil {
					inputJSON = string(inputBytes)
				}
			}
			toolCalls = append(toolCalls, anyllm.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: anyllm.FunctionCall{
					Name:      block.Name,
					Arguments: inputJSON,
				},
			})
		}
	}

	message := anyllm.Message{
		Role:      anyllm.RoleAssistant,
		Content:   content,
		ToolCalls: toolCalls,
		Reasoning: reasoning,
	}

	finishReason := convertStopReason(string(resp.StopReason))

	return &anyllm.ChatCompletion{
		ID:     resp.ID,
		Object: "chat.completion",
		Model:  string(resp.Model),
		Choices: []anyllm.Choice{{
			Index:        0,
			Message:      message,
			FinishReason: finishReason,
		}},
		Usage: &anyllm.Usage{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
	}
}

// convertStopReason converts Anthropic stop reason to OpenAI finish reason.
func convertStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return anyllm.FinishReasonStop
	case "max_tokens":
		return anyllm.FinishReasonLength
	case "tool_use":
		return anyllm.FinishReasonToolCalls
	case "stop_sequence":
		return anyllm.FinishReasonStop
	default:
		return anyllm.FinishReasonStop
	}
}

// init registers the Anthropic provider.
func init() {
	anyllm.Register(providerName, func(opts ...anyllm.Option) (anyllm.Provider, error) {
		return New(opts...)
	})
}
