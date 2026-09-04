// Package anyllm provides a unified interface for interacting with LLM providers.
//
// This package re-exports common types and configuration options from subpackages,
// allowing most use cases to work with just two imports:
//
//	import (
//	    anyllm "github.com/mozilla-ai/any-llm-go"
//	    "github.com/mozilla-ai/any-llm-go/providers/openai"
//	)
//
//	provider, err := openai.New(anyllm.WithAPIKey("sk-..."))
//	response, err := provider.Completion(ctx, anyllm.CompletionParams{
//	    Model: "gpt-4o-mini",
//	    Messages: []anyllm.Message{
//	        {Role: anyllm.RoleUser, Content: "Hello!"},
//	    },
//	})
package anyllm

import (
	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Message roles.
const (
	RoleAssistant = providers.RoleAssistant
	RoleSystem    = providers.RoleSystem
	RoleTool      = providers.RoleTool
	RoleUser      = providers.RoleUser
)

// Finish reasons.
const (
	FinishReasonContentFilter = providers.FinishReasonContentFilter
	FinishReasonLength        = providers.FinishReasonLength
	FinishReasonStop          = providers.FinishReasonStop
	FinishReasonToolCalls     = providers.FinishReasonToolCalls
)

// Batch status constants.
const (
	BatchStatusCancelled  = providers.BatchStatusCancelled
	BatchStatusCancelling = providers.BatchStatusCancelling
	BatchStatusCompleted  = providers.BatchStatusCompleted
	BatchStatusExpired    = providers.BatchStatusExpired
	BatchStatusFailed     = providers.BatchStatusFailed
	BatchStatusFinalizing = providers.BatchStatusFinalizing
	BatchStatusInProgress = providers.BatchStatusInProgress
	BatchStatusValidating = providers.BatchStatusValidating
)

// ReasoningEffort levels.
const (
	ReasoningEffortAuto    = providers.ReasoningEffortAuto
	ReasoningEffortHigh    = providers.ReasoningEffortHigh
	ReasoningEffortLow     = providers.ReasoningEffortLow
	ReasoningEffortMax     = providers.ReasoningEffortMax
	ReasoningEffortMedium  = providers.ReasoningEffortMedium
	ReasoningEffortMinimal = providers.ReasoningEffortMinimal
	ReasoningEffortNone    = providers.ReasoningEffortNone
	ReasoningEffortXHigh   = providers.ReasoningEffortXHigh
)

// Provider types.
type (
	BatchProvider      = providers.BatchProvider
	Capabilities       = providers.Capabilities
	CapabilityProvider = providers.CapabilityProvider
	EmbeddingProvider  = providers.EmbeddingProvider
	ModelLister        = providers.ModelLister
	ModerationProvider = providers.ModerationProvider
	Provider           = providers.Provider
	RerankProvider     = providers.RerankProvider
)

// Batch types.
type (
	Batch              = providers.Batch
	BatchRequestCounts = providers.BatchRequestCounts
	BatchRequestItem   = providers.BatchRequestItem
	BatchResult        = providers.BatchResult
	BatchResultError   = providers.BatchResultError
	BatchResultItem    = providers.BatchResultItem
	CreateBatchParams  = providers.CreateBatchParams
	ListBatchesOptions = providers.ListBatchesOptions
)

// Request/Response types.
type (
	ChatCompletion             = providers.ChatCompletion
	ChatCompletionChunk        = providers.ChatCompletionChunk
	ChatCompletionLogprobs     = providers.ChatCompletionLogprobs
	ChatCompletionTokenLogprob = providers.ChatCompletionTokenLogprob
	ChatCompletionTopLogprob   = providers.ChatCompletionTopLogprob
	Choice                     = providers.Choice
	ChunkChoice                = providers.ChunkChoice
	ChunkDelta                 = providers.ChunkDelta
	CompletionParams           = providers.CompletionParams
	EmbeddingParams            = providers.EmbeddingParams
	EmbeddingResponse          = providers.EmbeddingResponse
	ModelsResponse             = providers.ModelsResponse
	ModerationParams           = providers.ModerationParams
	ModerationResponse         = providers.ModerationResponse
	ModerationResult           = providers.ModerationResult
)

// Message types.
type (
	ContentPart = providers.ContentPart
	ImageURL    = providers.ImageURL
	Message     = providers.Message
	Reasoning   = providers.Reasoning
)

// Tool types.
type (
	Function           = providers.Function
	FunctionCall       = providers.FunctionCall
	Tool               = providers.Tool
	ToolCall           = providers.ToolCall
	ToolChoice         = providers.ToolChoice
	ToolChoiceFunction = providers.ToolChoiceFunction
)

// Response format types.
type (
	JSONSchema     = providers.JSONSchema
	ResponseFormat = providers.ResponseFormat
	StreamOptions  = providers.StreamOptions
)

// Rerank types.
type (
	RerankMeta     = providers.RerankMeta
	RerankParams   = providers.RerankParams
	RerankResponse = providers.RerankResponse
	RerankResult   = providers.RerankResult
	RerankUsage    = providers.RerankUsage
)

// Usage and model types.
type (
	BatchStatus     = providers.BatchStatus
	EmbeddingData   = providers.EmbeddingData
	EmbeddingUsage  = providers.EmbeddingUsage
	Model           = providers.Model
	ReasoningEffort = providers.ReasoningEffort
	Usage           = providers.Usage
)

// Config types.
type (
	Config = config.Config
	Option = config.Option
)

// Configuration options.
var (
	NewConfig      = config.New
	WithAPIKey     = config.WithAPIKey
	WithBaseURL    = config.WithBaseURL
	WithExtra      = config.WithExtra
	WithHeader     = config.WithHeader
	WithHTTPClient = config.WithHTTPClient
	WithTimeout    = config.WithTimeout
)

// Sentinel errors for type checking with errors.Is().
var (
	ErrAuthentication      = errors.ErrAuthentication
	ErrContentFilter       = errors.ErrContentFilter
	ErrContextLength       = errors.ErrContextLength
	ErrInsufficientFunds   = errors.ErrInsufficientFunds
	ErrInvalidRequest      = errors.ErrInvalidRequest
	ErrMissingAPIKey       = errors.ErrMissingAPIKey
	ErrModelNotFound       = errors.ErrModelNotFound
	ErrProvider            = errors.ErrProvider
	ErrRateLimit           = errors.ErrRateLimit
	ErrUnsupported         = errors.ErrUnsupported
	ErrUnsupportedParam    = errors.ErrUnsupportedParam
	ErrUnsupportedProvider = errors.ErrUnsupportedProvider
)

// Error types.
type (
	AuthenticationError       = errors.AuthenticationError
	BaseError                 = errors.BaseError
	ContentFilterError        = errors.ContentFilterError
	ContextLengthError        = errors.ContextLengthError
	InsufficientFundsError    = errors.InsufficientFundsError
	InvalidRequestError       = errors.InvalidRequestError
	MissingAPIKeyError        = errors.MissingAPIKeyError
	ModelNotFoundError        = errors.ModelNotFoundError
	ProviderError             = errors.ProviderError
	RateLimitError            = errors.RateLimitError
	UnsupportedOperationError = errors.UnsupportedOperationError
	UnsupportedParamError     = errors.UnsupportedParamError
	UnsupportedProviderError  = errors.UnsupportedProviderError
)
