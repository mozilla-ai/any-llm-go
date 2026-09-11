package providers

import (
	"context"
	"encoding/json"
)

// ResponsesProvider is an optional interface for providers that expose the
// portable Responses request and result contract.
type ResponsesProvider interface {
	Provider
	Responses(ctx context.Context, params ResponsesParams) (*ResponsesResult, error)
}

// ResponsesInputRole is a message role accepted by the portable Responses input.
type ResponsesInputRole string

const (
	ResponsesInputRoleAssistant ResponsesInputRole = "assistant"
	ResponsesInputRoleDeveloper ResponsesInputRole = "developer"
	ResponsesInputRoleSystem    ResponsesInputRole = "system"
	ResponsesInputRoleUser      ResponsesInputRole = "user"
)

// ResponsesInputItem is a text message sent through the portable Responses API.
type ResponsesInputItem struct {
	Content string             `json:"content"`
	Role    ResponsesInputRole `json:"role"`
}

// ResponsesParams contains the portable subset of a Responses request.
// OpenAI permits Model and Input to be omitted. Provider implementations may
// enforce stricter documented requirements.
// https://developers.openai.com/api/reference/resources/responses/methods/create
type ResponsesParams struct {
	Input           []ResponsesInputItem `json:"input,omitempty"`
	Instructions    *string              `json:"instructions,omitempty"`
	MaxOutputTokens *int                 `json:"max_output_tokens,omitempty"`
	Model           string               `json:"model,omitempty"`
	ReasoningEffort ReasoningEffort      `json:"reasoning_effort,omitempty"`
}

// ResponsesResult is a portable Responses result. ProviderRaw retains fields
// outside the normalized subset.
type ResponsesResult struct {
	ID                string                      `json:"id"`
	Model             string                      `json:"model"`
	Status            string                      `json:"status,omitempty"`
	Error             *ResponsesError             `json:"error,omitempty"`
	IncompleteDetails *ResponsesIncompleteDetails `json:"incomplete_details,omitempty"`
	OutputText        string                      `json:"output_text"`
	OutputItems       []ResponsesOutputItem       `json:"output_items,omitempty"`
	Usage             *ResponsesUsage             `json:"usage,omitempty"`
	ProviderRaw       json.RawMessage             `json:"provider_raw,omitempty"`
}

// ResponsesError describes a response whose generation status is failed.
// ProviderRaw retains provider-specific error details.
type ResponsesError struct {
	Code        string          `json:"code"`
	Message     string          `json:"message"`
	ProviderRaw json.RawMessage `json:"provider_raw,omitempty"`
}

// ResponsesIncompleteDetails describes why generation stopped before completion.
// Providers can add reasons, so Reason intentionally remains an open string.
type ResponsesIncompleteDetails struct {
	Reason      string          `json:"reason"`
	ProviderRaw json.RawMessage `json:"provider_raw,omitempty"`
}

// ResponsesOutputItem contains the portable fields for one Responses output
// item. ProviderRaw retains the complete item, including unknown item types.
type ResponsesOutputItem struct {
	Type        string                   `json:"type"`
	ID          string                   `json:"id,omitempty"`
	Status      string                   `json:"status,omitempty"`
	Content     []ResponsesOutputContent `json:"content,omitempty"`
	Name        string                   `json:"name,omitempty"`
	CallID      string                   `json:"call_id,omitempty"`
	Arguments   string                   `json:"arguments,omitempty"`
	Summary     []string                 `json:"summary,omitempty"`
	ProviderRaw json.RawMessage          `json:"provider_raw,omitempty"`
}

// ResponsesOutputContent contains one portable message content part.
type ResponsesOutputContent struct {
	Type        string          `json:"type"`
	Text        string          `json:"text,omitempty"`
	Refusal     string          `json:"refusal,omitempty"`
	ProviderRaw json.RawMessage `json:"provider_raw,omitempty"`
}

// ResponsesUsage contains token usage reported by a Responses result.
type ResponsesUsage struct {
	InputTokens     int `json:"input_tokens"`
	OutputTokens    int `json:"output_tokens"`
	TotalTokens     int `json:"total_tokens"`
	CachedTokens    int `json:"cached_tokens,omitempty"`
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}
