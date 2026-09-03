// Package providers defines the core provider interface and related types.
package providers

import (
	"context"
	"encoding/json"
)

// Finish reasons.
const (
	FinishReasonContentFilter = "content_filter"
	FinishReasonLength        = "length"
	FinishReasonStop          = "stop"
	FinishReasonToolCalls     = "tool_calls"
)

// Reasoning effort levels for extended thinking.
const (
	ReasoningEffortAuto    ReasoningEffort = "auto"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMax     ReasoningEffort = "max"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
)

// Message roles.
const (
	RoleAssistant = "assistant"
	RoleSystem    = "system"
	RoleTool      = "tool"
	RoleUser      = "user"
)

// CapabilityProvider is an optional interface for providers to report capabilities.
type CapabilityProvider interface {
	Provider
	Capabilities() Capabilities
}

// EmbeddingProvider is an optional interface for providers that support embeddings.
type EmbeddingProvider interface {
	Provider
	Embedding(ctx context.Context, params EmbeddingParams) (*EmbeddingResponse, error)
}

// ModerationProvider is an optional interface for providers that support
// content moderation. Use errors.As with *errors.UnsupportedOperationError to detect
// providers that do not support moderation.
type ModerationProvider interface {
	Provider
	Moderation(ctx context.Context, params ModerationParams) (*ModerationResponse, error)
}

// ErrorConverter converts provider-specific SDK errors to unified error types.
// This interface ensures all providers implement consistent error handling.
type ErrorConverter interface {
	// ConvertError takes a provider SDK error and returns a normalized error type
	// from the errors package (e.g., *errors.RateLimitError, *errors.AuthenticationError).
	ConvertError(err error) error
}

// ModelLister is an optional interface for providers that support listing models.
type ModelLister interface {
	Provider
	ListModels(ctx context.Context) (*ModelsResponse, error)
}

// RerankProvider is an optional interface for providers that support document reranking.
type RerankProvider interface {
	Provider
	Rerank(ctx context.Context, params RerankParams) (*RerankResponse, error)
}

// Provider is the core interface that all LLM providers must implement.
type Provider interface {
	// Name returns the provider's identifier (e.g., "openai", "anthropic").
	Name() string

	// Completion performs a chat completion request.
	Completion(ctx context.Context, params CompletionParams) (*ChatCompletion, error)

	// CompletionStream performs a streaming chat completion request.
	CompletionStream(ctx context.Context, params CompletionParams) (<-chan ChatCompletionChunk, <-chan error)
}

// ProviderData holds provider-specific metadata keyed by field name.
type ProviderData map[string]any

// ReasoningEffort levels for extended thinking.
type ReasoningEffort string

// Capabilities describes what features a provider supports.
type Capabilities struct {
	Completion          bool
	CompletionImage     bool
	CompletionPDF       bool
	CompletionReasoning bool
	CompletionStreaming bool
	CompletionTools     bool
	Embedding           bool
	ListModels          bool
	Moderation          bool
	Rerank              bool
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

// Choice represents a completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason,omitempty"`
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

// CompletionParams represents normalized parameters for chat completion requests.
type CompletionParams struct {
	Model             string          `json:"model"`
	Messages          []Message       `json:"messages"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	MaxTokens         *int            `json:"max_tokens,omitempty"`
	Stop              []string        `json:"stop,omitempty"`
	Stream            bool            `json:"stream,omitempty"`
	StreamOptions     *StreamOptions  `json:"stream_options,omitempty"`
	Tools             []Tool          `json:"tools,omitempty"`
	ToolChoice        any             `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	ResponseFormat    *ResponseFormat `json:"response_format,omitempty"`
	ReasoningEffort   ReasoningEffort `json:"reasoning_effort,omitempty"`
	Seed              *int            `json:"seed,omitempty"`
	User              string          `json:"user,omitempty"`
	Extra             map[string]any  `json:"-"`
}

// ContentPart represents a part of a multi-modal message.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// EmbeddingData represents a single embedding.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingParams represents parameters for embedding requests.
type EmbeddingParams struct {
	Model          string `json:"model"`
	Input          any    `json:"input"`
	EncodingFormat string `json:"encoding_format,omitempty"`
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

// EmbeddingUsage represents token usage for embeddings.
type EmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// ModerationParams is the request payload for POST /v1/moderations.
// Input accepts string | []string | []ContentPart.
type ModerationParams struct {
	// IncludeRaw is sent as a ?include_raw=true query param; it is
	// intentionally excluded from the JSON body.
	IncludeRaw bool   `json:"-"`
	Input      any    `json:"input"`
	Model      string `json:"model"`
	User       string `json:"user,omitempty"`
}

// ModerationResult is one result entry in a moderation response.
type ModerationResult struct {
	CategoryAppliedInputTypes map[string][]string `json:"category_applied_input_types,omitempty"`
	CategoryScores            map[string]float64  `json:"category_scores,omitempty"`
	Categories                map[string]bool     `json:"categories,omitempty"`
	Flagged                   bool                `json:"flagged"`
	ProviderRaw               json.RawMessage     `json:"provider_raw,omitempty"`
}

// ModerationResponse is the OpenAI-compatible moderation response.
type ModerationResponse struct {
	ID      string             `json:"id"`
	Model   string             `json:"model"`
	Results []ModerationResult `json:"results"`
}

// RerankMeta contains provider-specific billing metadata.
type RerankMeta struct {
	BilledUnits map[string]float64 `json:"billed_units,omitempty"`
	Tokens      map[string]int     `json:"tokens,omitempty"`
}

// RerankParams represents parameters for a rerank request.
type RerankParams struct {
	Documents       []string `json:"documents"`
	MaxTokensPerDoc *int     `json:"max_tokens_per_doc,omitempty"`
	Model           string   `json:"model"`
	Query           string   `json:"query"`
	TopN            *int     `json:"top_n,omitempty"`
	User            string   `json:"user,omitempty"`
}

// RerankResponse represents a rerank response.
type RerankResponse struct {
	ID      string         `json:"id"`
	Meta    *RerankMeta    `json:"meta,omitempty"`
	Results []RerankResult `json:"results"`
	Usage   *RerankUsage   `json:"usage,omitempty"`
}

// RerankResult represents a single reranked document score.
type RerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

// RerankUsage contains normalized token usage.
type RerankUsage struct {
	TotalTokens *int `json:"total_tokens,omitempty"`
}

// Function represents a function definition for tool calling.
type Function struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// FunctionCall represents the function being called.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ImageURL represents an image URL in a message.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// JSONSchema for structured output.
type JSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema"`
	Strict      *bool          `json:"strict,omitempty"`
}

// Message represents a chat message in OpenAI format.
type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Reasoning  *Reasoning `json:"reasoning,omitempty"`
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

// Reasoning represents extended thinking/reasoning content.
type Reasoning struct {
	Content string `json:"content,omitempty"`
}

// ResponseFormat specifies the format of the response.
type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// StreamOptions contains options for streaming responses.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// Tool represents a tool/function that can be called.
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// ToolCall represents a tool call made by the assistant.
type ToolCall struct {
	// Extra holds provider-specific metadata for round-tripping across
	// multi-turn conversations. Keyed by provider name (e.g. "gemini").
	// Excluded from JSON; callers preserve this through their own storage.
	Extra    map[string]ProviderData `json:"-"`
	Function FunctionCall            `json:"function"`
	ID       string                  `json:"id"`
	Type     string                  `json:"type"`
}

// ToolChoice represents a specific tool choice.
type ToolChoice struct {
	Type     string              `json:"type"`
	Function *ToolChoiceFunction `json:"function,omitempty"`
}

// ToolChoiceFunction specifies which function to call.
type ToolChoiceFunction struct {
	Name string `json:"name"`
}

// Usage represents token usage information.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`
}

// ContentParts extracts content parts from a message.
func (m *Message) ContentParts() []ContentPart {
	if m.Content == nil {
		return nil
	}

	if parts, ok := m.Content.([]ContentPart); ok {
		return parts
	}

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

// ContentString extracts string content from a message.
func (m *Message) ContentString() string {
	if s, ok := m.Content.(string); ok {
		return s
	}
	return ""
}

// IsMultiModal returns true if the message contains multi-modal content.
func (m *Message) IsMultiModal() bool {
	return m.ContentParts() != nil
}

// Batch represents a batch job.
type Batch struct {
	CompletedAt      *int64              `json:"completed_at,omitempty"`
	CompletionWindow string              `json:"completion_window"`
	CreatedAt        int64               `json:"created_at"`
	Endpoint         string              `json:"endpoint"`
	ErrorFileID      string              `json:"error_file_id,omitempty"`
	ID               string              `json:"id"`
	InProgressAt     *int64              `json:"in_progress_at,omitempty"`
	InputFileID      string              `json:"input_file_id,omitempty"`
	Metadata         map[string]string   `json:"metadata,omitempty"`
	Object           string              `json:"object"`
	OutputFileID     string              `json:"output_file_id,omitempty"`
	Provider         string              `json:"provider,omitempty"`
	RequestCounts    *BatchRequestCounts `json:"request_counts,omitempty"`
	Status           BatchStatus         `json:"status"`
}

// BatchStatus represents the status of a batch job.
type BatchStatus string

const (
	BatchStatusCancelled  BatchStatus = "cancelled"
	BatchStatusCancelling BatchStatus = "cancelling"
	BatchStatusCompleted  BatchStatus = "completed"
	BatchStatusExpired    BatchStatus = "expired"
	BatchStatusFailed     BatchStatus = "failed"
	BatchStatusFinalizing BatchStatus = "finalizing"
	BatchStatusInProgress BatchStatus = "in_progress"
	BatchStatusValidating BatchStatus = "validating"
)

// BatchRequestCounts tracks request counts for a batch job.
type BatchRequestCounts struct {
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
	Total     int `json:"total"`
}

// BatchRequestItem is a single request within a batch.
type BatchRequestItem struct {
	Body     map[string]any `json:"body"`
	CustomID string         `json:"custom_id"`
}

// CreateBatchParams are parameters for creating a batch job.
type CreateBatchParams struct {
	CompletionWindow string             `json:"completion_window,omitempty"`
	Metadata         map[string]string  `json:"metadata,omitempty"`
	Model            string             `json:"model"`
	Requests         []BatchRequestItem `json:"requests"`
}

// ListBatchesOptions are options for listing batch jobs.
type ListBatchesOptions struct {
	After string
	Limit *int
}

// BatchResult contains the results of a completed batch job.
type BatchResult struct {
	Results []BatchResultItem `json:"results"`
}

// BatchResultItem is the result of a single request within a batch.
type BatchResultItem struct {
	CustomID string            `json:"custom_id"`
	Error    *BatchResultError `json:"error,omitempty"`
	Result   *ChatCompletion   `json:"result,omitempty"`
}

// BatchResultError is an error for a single request within a batch.
type BatchResultError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// BatchProvider is an optional interface for providers that support
// batch operations. Use a type assertion to check if a provider
// supports batch:
//
//	if bp, ok := provider.(BatchProvider); ok {
//	    batch, err := bp.CreateBatch(ctx, params)
//	}
type BatchProvider interface {
	Provider
	CreateBatch(ctx context.Context, params CreateBatchParams) (*Batch, error)
	RetrieveBatch(ctx context.Context, batchID string, provider string) (*Batch, error)
	CancelBatch(ctx context.Context, batchID string, provider string) (*Batch, error)
	ListBatches(ctx context.Context, provider string, opts ListBatchesOptions) ([]Batch, error)
	RetrieveBatchResults(ctx context.Context, batchID string, provider string) (*BatchResult, error)
}
