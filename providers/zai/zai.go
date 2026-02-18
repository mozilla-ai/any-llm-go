// Package zai provides a z.ai provider implementation for any-llm.
package zai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	defaultBaseURL = "https://api.z.ai/api/paas/v4/"
	envAPIKey      = "ZAI_API_KEY"
	providerName   = "zai"
	dataURIPrefix  = "data:image/"
	base64Prefix   = "base64,"
)

// Object type constants.
const (
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectList                = "list"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for z.ai.
type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// New creates a new z.ai provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	apiKey := cfg.ResolveAPIKey(envAPIKey)
	if apiKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	baseURL := defaultBaseURL
	if cfg.BaseURL != "" {
		baseURL = cfg.BaseURL
	}

	return &Provider{
		apiKey:     apiKey,
		baseURL:    baseURL,
		httpClient: cfg.HTTPClient(),
	}, nil
}

// Capabilities returns the provider's capabilities.
func (p *Provider) Capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     false,
		CompletionPDF:       false,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           false,
		ListModels:          true,
	}
}

// Completion performs a chat completion request.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	reqBody, err := p.createRequest(params, false)
	if err != nil {
		return nil, err
	}

	resp, err := p.doRequest(ctx, "POST", "chat/completions", reqBody)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Printf("zai: failed to close response body: %v", err)
		}
	}()

	var zaiResult zaiChatCompletion
	if err := json.NewDecoder(resp.Body).Decode(&zaiResult); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return zaiResult.toProviderCompletion(), nil
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

		reqBody, err := p.createRequest(params, true)
		if err != nil {
			select {
			case errs <- err:
			case <-ctx.Done():
			}
			return
		}

		resp, err := p.doRequest(ctx, "POST", "chat/completions", reqBody)
		if err != nil {
			select {
			case errs <- err:
			case <-ctx.Done():
			}
			return
		}
		defer func() {
			if err := resp.Body.Close(); err != nil {
				log.Printf("zai: failed to close response body: %v", err)
			}
		}()

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			if !bytes.HasPrefix(line, []byte("data: ")) {
				continue
			}

			data := bytes.TrimPrefix(line, []byte("data: "))
			if string(data) == "[DONE]" {
				return
			}

			var zaiChunk zaiChatCompletionChunk
			if err := json.Unmarshal(data, &zaiChunk); err != nil {
				continue
			}

			select {
			case chunks <- zaiChunk.toProviderChunk():
			case <-ctx.Done():
				return
			}
		}

		if err := scanner.Err(); err != nil {
			select {
			case errs <- fmt.Errorf("reading stream: %w", err):
			case <-ctx.Done():
			}
		}
	}()

	return chunks, errs
}

// ConvertError converts z.ai errors to unified error types.
// Implements providers.ErrorConverter.
func (p *Provider) ConvertError(err error) error {
	return errors.NewProviderError(providerName, err)
}

// ListModels returns a list of available models.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.doRequest(ctx, "GET", "models", nil)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			log.Printf("zai: failed to close response body: %v", err)
		}
	}()

	var apiResp struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("decoding list models: %w", err)
	}

	models := make([]providers.Model, len(apiResp.Data))
	for i, m := range apiResp.Data {
		models[i] = providers.Model{
			ID:      m.ID,
			Object:  m.Object,
			Created: m.Created,
			OwnedBy: m.OwnedBy,
		}
	}

	return &providers.ModelsResponse{
		Object: apiResp.Object,
		Data:   models,
	}, nil
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// createRequest converts providers.CompletionParams to a z.ai chat request.
func (p *Provider) createRequest(params providers.CompletionParams, stream bool) (*chatRequest, error) {
	req := &chatRequest{
		Model:       params.Model,
		Stream:      stream,
		Tools:       params.Tools,
		ToolChoice:  params.ToolChoice,
		Temperature: params.Temperature,
		TopP:        params.TopP,
		MaxTokens:   params.MaxTokens,
		Stop:        params.Stop,
		UserID:      params.User,
	}

	if params.ReasoningEffort != "" && params.ReasoningEffort != providers.ReasoningEffortNone {
		req.Thinking = &thinkingParam{Type: "enabled"}
	}

	msgs := make([]messageParam, len(params.Messages))
	for i, m := range params.Messages {
		mp := messageParam{
			Message: providers.Message{
				Role:       m.Role,
				Name:       m.Name,
				ToolCalls:  m.ToolCalls,
				ToolCallID: m.ToolCallID,
				Content:    m.Content,
			},
		}

		if m.Reasoning != nil {
			mp.ReasoningContent = m.Reasoning.Content
		}

		if parts, ok := m.Content.([]providers.ContentPart); ok {
			newParts := make([]contentPart, len(parts))
			for j, part := range parts {
				newParts[j] = contentPart{
					Type: part.Type,
					Text: part.Text,
				}
				if part.ImageURL != nil {
					// Strip Data URI prefix for z.ai image format.
					url := part.ImageURL.URL
					if strings.HasPrefix(url, dataURIPrefix) {
						if idx := strings.Index(url, base64Prefix); idx != -1 {
							url = url[idx+len(base64Prefix):]
						}
					}
					newParts[j].ImageURL = map[string]string{
						"url": url,
					}
				}
			}
			mp.Content = newParts
		}
		msgs[i] = mp
	}
	req.Messages = msgs

	return req, nil
}

// doRequest sends an HTTP request to the z.ai API.
func (p *Provider) doRequest(ctx context.Context, method, endpoint string, body any) (*http.Response, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshaling request: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	fullURL, err := url.JoinPath(p.baseURL, endpoint)
	if err != nil {
		return nil, fmt.Errorf("joining url path: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, method, fullURL, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)
	req.Header.Set("Accept-Language", "en-US,en")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	if resp.StatusCode >= 400 {
		defer func() {
			if err := resp.Body.Close(); err != nil {
				log.Printf("zai: failed to close response body: %v", err)
			}
		}()
		return nil, p.handleErrorResponse(resp)
	}

	return resp, nil
}

// handleErrorResponse parses an error response from the z.ai API.
func (p *Provider) handleErrorResponse(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)
	msg := fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(body))

	type errorResp struct {
		Error struct {
			Message string `json:"message"`
			Code    any    `json:"code"`
		} `json:"error"`
	}
	var e errorResp
	if json.Unmarshal(body, &e) == nil && e.Error.Message != "" {
		msg = fmt.Sprintf("z.ai error: %s (code: %v)", e.Error.Message, e.Error.Code)
	}

	switch resp.StatusCode {
	case 401:
		return errors.NewAuthenticationError(providerName, fmt.Errorf("%s", msg))
	case 429:
		return errors.NewRateLimitError(providerName, fmt.Errorf("%s", msg))
	case 404:
		return errors.NewModelNotFoundError(providerName, fmt.Errorf("%s", msg))
	case 400:
		return errors.NewInvalidRequestError(providerName, fmt.Errorf("%s", msg))
	default:
		return errors.NewProviderError(providerName, fmt.Errorf("%s", msg))
	}
}

// z.ai response types.

// zaiChatCompletion represents a z.ai chat completion response.
type zaiChatCompletion struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []zaiChoice      `json:"choices"`
	Usage   *providers.Usage `json:"usage,omitempty"`
}

// zaiChoice represents a choice in a z.ai chat completion response.
type zaiChoice struct {
	Index        int        `json:"index"`
	Message      zaiMessage `json:"message"`
	FinishReason string     `json:"finish_reason,omitempty"`
}

// zaiMessage represents a message in a z.ai chat completion response.
type zaiMessage struct {
	providers.Message
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// toProviderCompletion converts a z.ai response to the unified ChatCompletion format.
func (z *zaiChatCompletion) toProviderCompletion() *providers.ChatCompletion {
	choices := make([]providers.Choice, len(z.Choices))
	for i, c := range z.Choices {
		msg := c.Message.Message
		if c.Message.ReasoningContent != "" {
			msg.Reasoning = &providers.Reasoning{
				Content: c.Message.ReasoningContent,
			}
		}
		choices[i] = providers.Choice{
			Index:        c.Index,
			Message:      msg,
			FinishReason: c.FinishReason,
		}
	}
	return &providers.ChatCompletion{
		ID:      z.ID,
		Object:  z.Object,
		Created: z.Created,
		Model:   z.Model,
		Choices: choices,
		Usage:   z.Usage,
	}
}

// zaiChatCompletionChunk represents a z.ai streaming chunk.
type zaiChatCompletionChunk struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []zaiChunkChoice `json:"choices"`
	Usage   *providers.Usage `json:"usage,omitempty"`
}

// zaiChunkChoice represents a choice in a z.ai streaming chunk.
type zaiChunkChoice struct {
	Index        int           `json:"index"`
	Delta        zaiChunkDelta `json:"delta"`
	FinishReason string        `json:"finish_reason,omitempty"`
}

// zaiChunkDelta represents delta content in a z.ai streaming chunk.
type zaiChunkDelta struct {
	providers.ChunkDelta
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// toProviderChunk converts a z.ai streaming chunk to the unified ChatCompletionChunk format.
func (z *zaiChatCompletionChunk) toProviderChunk() providers.ChatCompletionChunk {
	choices := make([]providers.ChunkChoice, len(z.Choices))
	for i, c := range z.Choices {
		delta := c.Delta.ChunkDelta
		if c.Delta.ReasoningContent != "" {
			delta.Reasoning = &providers.Reasoning{
				Content: c.Delta.ReasoningContent,
			}
		}
		choices[i] = providers.ChunkChoice{
			Index:        c.Index,
			Delta:        delta,
			FinishReason: c.FinishReason,
		}
	}
	return providers.ChatCompletionChunk{
		ID:      z.ID,
		Object:  z.Object,
		Created: z.Created,
		Model:   z.Model,
		Choices: choices,
		Usage:   z.Usage,
	}
}

// z.ai request types.

// chatRequest represents a z.ai chat completion request body.
type chatRequest struct {
	Model       string           `json:"model"`
	Messages    []messageParam   `json:"messages"`
	Stream      bool             `json:"stream,omitempty"`
	Thinking    *thinkingParam   `json:"thinking,omitempty"`
	Tools       []providers.Tool `json:"tools,omitempty"`
	ToolChoice  any              `json:"tool_choice,omitempty"`
	Temperature *float64         `json:"temperature,omitempty"`
	TopP        *float64         `json:"top_p,omitempty"`
	MaxTokens   *int             `json:"max_tokens,omitempty"`
	Stop        []string         `json:"stop,omitempty"`
	UserID      string           `json:"user_id,omitempty"`
}

// thinkingParam represents the thinking configuration for z.ai.
type thinkingParam struct {
	Type string `json:"type"`
}

// messageParam represents a message in a z.ai chat request.
type messageParam struct {
	providers.Message
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// contentPart represents a multimodal content part in a z.ai message.
type contentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL any    `json:"image_url,omitempty"`
}
