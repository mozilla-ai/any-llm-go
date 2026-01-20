package errors

import (
	stderrors "errors"
	"fmt"
)

// Error codes used in BaseError.Code field.
const (
	CodeRateLimit           = "rate_limit"
	CodeAuthError           = "auth_error"
	CodeInvalidRequest      = "invalid_request"
	CodeContextLength       = "context_length_exceeded"
	CodeContentFilter       = "content_filter"
	CodeModelNotFound       = "model_not_found"
	CodeProviderError       = "provider_error"
	CodeMissingAPIKey       = "missing_api_key"
	CodeUnsupportedProvider = "unsupported_provider"
	CodeUnsupportedParam    = "unsupported_parameter"
)

// Sentinel errors for type checking with errors.Is().
var (
	ErrRateLimit           = stderrors.New("rate limit exceeded")
	ErrAuthentication      = stderrors.New("authentication failed")
	ErrInvalidRequest      = stderrors.New("invalid request")
	ErrContextLength       = stderrors.New("context length exceeded")
	ErrContentFilter       = stderrors.New("content filtered")
	ErrModelNotFound       = stderrors.New("model not found")
	ErrProvider            = stderrors.New("provider error")
	ErrMissingAPIKey       = stderrors.New("missing API key")
	ErrUnsupportedProvider = stderrors.New("unsupported provider")
	ErrUnsupportedParam    = stderrors.New("unsupported parameter")
)

// BaseError is the base error type for all any-llm errors.
// It wraps the original error and includes provider context.
type BaseError struct {
	// Code is a short error code (e.g., "rate_limit", "auth_error").
	Code string

	// Provider is the name of the provider that returned the error.
	Provider string

	// Err is the underlying error (original provider error).
	Err error

	// sentinel is the sentinel error for errors.Is() matching.
	sentinel error
}

// Error implements the error interface.
func (e *BaseError) Error() string {
	msg := ""
	if e.Err != nil {
		msg = e.Err.Error()
	}
	if e.Provider != "" {
		return fmt.Sprintf("[%s] %s: %s", e.Provider, e.Code, msg)
	}
	return fmt.Sprintf("%s: %s", e.Code, msg)
}

// Is allows checking error types with errors.Is().
func (e *BaseError) Is(target error) bool {
	return e.sentinel != nil && target == e.sentinel
}

// Unwrap returns the underlying error for errors.Is() and errors.As().
func (e *BaseError) Unwrap() error {
	return e.Err
}

// RateLimitError is returned when the API rate limit is exceeded.
type RateLimitError struct {
	BaseError
	RetryAfter int // Seconds until retry is allowed, if known
}

// AuthenticationError is returned when authentication fails.
type AuthenticationError struct {
	BaseError
}

// InvalidRequestError is returned when the request is malformed.
type InvalidRequestError struct {
	BaseError
}

// ContextLengthError is returned when the context exceeds the model's limit.
type ContextLengthError struct {
	BaseError
}

// ContentFilterError is returned when content is blocked by safety filters.
type ContentFilterError struct {
	BaseError
}

// ModelNotFoundError is returned when the requested model doesn't exist.
type ModelNotFoundError struct {
	BaseError
}

// ProviderError is returned for general provider-side errors.
type ProviderError struct {
	BaseError
	StatusCode int
}

// MissingAPIKeyError is returned when no API key is provided.
type MissingAPIKeyError struct {
	BaseError
	EnvVar string // The environment variable that should contain the key
}

// UnsupportedProviderError is returned when the provider is not supported.
type UnsupportedProviderError struct {
	BaseError
}

// UnsupportedParamError is returned when a parameter is not supported.
type UnsupportedParamError struct {
	BaseError
	Param string // The unsupported parameter name
}

// NewRateLimitError creates a new RateLimitError.
func NewRateLimitError(provider string, err error) *RateLimitError {
	return &RateLimitError{
		BaseError: BaseError{
			Code:     CodeRateLimit,
			Provider: provider,
			Err:      err,
			sentinel: ErrRateLimit,
		},
	}
}

// NewAuthenticationError creates a new AuthenticationError.
func NewAuthenticationError(provider string, err error) *AuthenticationError {
	return &AuthenticationError{
		BaseError: BaseError{
			Code:     CodeAuthError,
			Provider: provider,
			Err:      err,
			sentinel: ErrAuthentication,
		},
	}
}

// NewInvalidRequestError creates a new InvalidRequestError.
func NewInvalidRequestError(provider string, err error) *InvalidRequestError {
	return &InvalidRequestError{
		BaseError: BaseError{
			Code:     CodeInvalidRequest,
			Provider: provider,
			Err:      err,
			sentinel: ErrInvalidRequest,
		},
	}
}

// NewContextLengthError creates a new ContextLengthError.
func NewContextLengthError(provider string, err error) *ContextLengthError {
	return &ContextLengthError{
		BaseError: BaseError{
			Code:     CodeContextLength,
			Provider: provider,
			Err:      err,
			sentinel: ErrContextLength,
		},
	}
}

// NewContentFilterError creates a new ContentFilterError.
func NewContentFilterError(provider string, err error) *ContentFilterError {
	return &ContentFilterError{
		BaseError: BaseError{
			Code:     CodeContentFilter,
			Provider: provider,
			Err:      err,
			sentinel: ErrContentFilter,
		},
	}
}

// NewModelNotFoundError creates a new ModelNotFoundError.
func NewModelNotFoundError(provider string, err error) *ModelNotFoundError {
	return &ModelNotFoundError{
		BaseError: BaseError{
			Code:     CodeModelNotFound,
			Provider: provider,
			Err:      err,
			sentinel: ErrModelNotFound,
		},
	}
}

// NewProviderError creates a new ProviderError.
func NewProviderError(provider string, err error) *ProviderError {
	return &ProviderError{
		BaseError: BaseError{
			Code:     CodeProviderError,
			Provider: provider,
			Err:      err,
			sentinel: ErrProvider,
		},
	}
}

// NewMissingAPIKeyError creates a new MissingAPIKeyError.
func NewMissingAPIKeyError(provider string, envVar string) *MissingAPIKeyError {
	return &MissingAPIKeyError{
		BaseError: BaseError{
			Code:     CodeMissingAPIKey,
			Provider: provider,
			Err: fmt.Errorf(
				"API key not provided. Set %s environment variable or pass WithAPIKey option",
				envVar,
			),
			sentinel: ErrMissingAPIKey,
		},
		EnvVar: envVar,
	}
}

// NewUnsupportedProviderError creates a new UnsupportedProviderError.
func NewUnsupportedProviderError(provider string) *UnsupportedProviderError {
	return &UnsupportedProviderError{
		BaseError: BaseError{
			Code:     CodeUnsupportedProvider,
			Provider: provider,
			Err:      fmt.Errorf("provider %q is not supported", provider),
			sentinel: ErrUnsupportedProvider,
		},
	}
}

// NewUnsupportedParamError creates a new UnsupportedParamError.
func NewUnsupportedParamError(provider string, param string) *UnsupportedParamError {
	return &UnsupportedParamError{
		BaseError: BaseError{
			Code:     CodeUnsupportedParam,
			Provider: provider,
			Err:      fmt.Errorf("parameter %q is not supported by provider %s", param, provider),
			sentinel: ErrUnsupportedParam,
		},
		Param: param,
	}
}
