package anyllm

import (
	"encoding/json"
)

// Message roles
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Finish reasons
const (
	FinishReasonStop          = "stop"
	FinishReasonLength        = "length"
	FinishReasonToolCalls     = "tool_calls"
	FinishReasonContentFilter = "content_filter"
)

// ReasoningEffort levels for extended thinking
type ReasoningEffort string

const (
	ReasoningEffortNone   ReasoningEffort = "none"
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
	ReasoningEffortAuto   ReasoningEffort = "auto"
)

// CompletionParams represents normalized parameters for chat completion requests.
// This is the internal representation used across all providers.
type CompletionParams struct {
	Model             string          `json:"model"`
	Messages          []Message       `json:"messages"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	MaxTokens         *int            `json:"max_tokens,omitempty"`
	Stop              []string        `json:"stop,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	Tools             []Tool          `json:"tools,omitempty"`
	ToolChoice        any             `json:"tool_choice,omitempty"` // string or ToolChoice
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	ResponseFormat    *ResponseFormat `json:"response_format,omitempty"`
	ReasoningEffort   ReasoningEffort `json:"reasoning_effort,omitempty"`
	Seed              *int            `json:"seed,omitempty"`
	User              string          `json:"user,omitempty"`
	Extra             map[string]any  `json:"-"` // Provider-specific extra parameters
}

// Message represents a chat message in OpenAI format.
type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"` // string or []ContentPart
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Reasoning  *Reasoning `json:"reasoning,omitempty"`
}

// ContentPart represents a part of a multi-modal message.
type ContentPart struct {
	Type     string    `json:"type"` // "text", "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image URL in a message.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"` // "auto", "low", "high"
}

// Reasoning represents extended thinking/reasoning content.
type Reasoning struct {
	Content string `json:"content,omitempty"`
}

// Tool represents a tool/function that can be called.
type Tool struct {
	Type     string   `json:"type"` // "function"
	Function Function `json:"function"`
}

// Function represents a function definition for tool calling.
type Function struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ToolCall represents a tool call made by the assistant.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall represents the function being called.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// ToolChoice represents a specific tool choice.
type ToolChoice struct {
	Type     string              `json:"type"` // "function"
	Function *ToolChoiceFunction `json:"function,omitempty"`
}

// ToolChoiceFunction specifies which function to call.
type ToolChoiceFunction struct {
	Name string `json:"name"`
}

// ResponseFormat specifies the format of the response.
type ResponseFormat struct {
	Type       string      `json:"type"` // "text", "json_object", "json_schema"
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// JSONSchema for structured output.
type JSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
	Strict      *bool          `json:"strict,omitempty"`
}

// ChatCompletion represents a chat completion response in OpenAI format.
type ChatCompletion struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// Choice represents a completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`
}

// ChatCompletionChunk represents a streaming chunk in OpenAI format.
type ChatCompletionChunk struct {
	ID                string        `json:"id"`
	Object            string        `json:"object"`
	Created           int64         `json:"created"`
	Model             string        `json:"model"`
	Choices           []ChunkChoice `json:"choices"`
	Usage             *Usage        `json:"usage,omitempty"`
	SystemFingerprint string        `json:"system_fingerprint,omitempty"`
}

// ChunkChoice represents a choice in a streaming chunk.
type ChunkChoice struct {
	Index        int        `json:"index"`
	Delta        ChunkDelta `json:"delta"`
	FinishReason string     `json:"finish_reason,omitempty"`
}

// ChunkDelta represents the delta content in a streaming chunk.
type ChunkDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Reasoning *Reasoning `json:"reasoning,omitempty"`
}

// EmbeddingParams represents parameters for embedding requests.
type EmbeddingParams struct {
	Model          string `json:"model"`
	Input          any    `json:"input"`                     // string or []string
	EncodingFormat string `json:"encoding_format,omitempty"` // "float" or "base64"
	Dimensions     *int   `json:"dimensions,omitempty"`
	User           string `json:"user,omitempty"`
}

// EmbeddingResponse represents an embedding response in OpenAI format.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  *EmbeddingUsage `json:"usage,omitempty"`
}

// EmbeddingData represents a single embedding.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingUsage represents token usage for embeddings.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// Model represents a model from the list models API.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse represents a list models response.
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// GetContentString extracts string content from a message.
// Returns empty string if content is not a string.
func (m *Message) GetContentString() string {
	if s, ok := m.Content.(string); ok {
		return s
	}
	return ""
}

// GetContentParts extracts content parts from a message.
// Returns nil if content is not a slice of content parts.
func (m *Message) GetContentParts() []ContentPart {
	if m.Content == nil {
		return nil
	}

	// Handle []ContentPart directly
	if parts, ok := m.Content.([]ContentPart); ok {
		return parts
	}

	// Handle []any (from JSON unmarshaling)
	if parts, ok := m.Content.([]any); ok {
		result := make([]ContentPart, 0, len(parts))
		for _, p := range parts {
			if partMap, ok := p.(map[string]any); ok {
				var part ContentPart
				if b, err := json.Marshal(partMap); err == nil {
					if err := json.Unmarshal(b, &part); err == nil {
						result = append(result, part)
					}
				}
			}
		}
		return result
	}

	return nil
}

// IsMultiModal returns true if the message contains multi-modal content.
func (m *Message) IsMultiModal() bool {
	return m.GetContentParts() != nil
}
