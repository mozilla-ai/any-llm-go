package errors

import (
	stderrors "errors"
	"fmt"
	"regexp"
	"strings"
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

	// Message is a human-readable error message.
	Message string

	// Provider is the name of the provider that returned the error.
	Provider string

	// Err is the underlying error (original provider error).
	Err error
}

// Error implements the error interface.
func (e *BaseError) Error() string {
	if e.Provider != "" {
		return fmt.Sprintf("[%s] %s: %s", e.Provider, e.Code, e.Message)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
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
func NewRateLimitError(provider, message string, err error) *RateLimitError {
	return &RateLimitError{
		BaseError: BaseError{
			Code:     "rate_limit",
			Message:  message,
			Provider: provider,
			Err:      err,
		},
	}
}

// NewAuthenticationError creates a new AuthenticationError.
func NewAuthenticationError(provider, message string, err error) *AuthenticationError {
	return &AuthenticationError{
		BaseError: BaseError{
			Code:     "auth_error",
			Message:  message,
			Provider: provider,
			Err:      err,
		},
	}
}

// NewInvalidRequestError creates a new InvalidRequestError.
func NewInvalidRequestError(provider, message string, err error) *InvalidRequestError {
	return &InvalidRequestError{
		BaseError: BaseError{
			Code:     "invalid_request",
			Message:  message,
			Provider: provider,
			Err:      err,
		},
	}
}

// NewContextLengthError creates a new ContextLengthError.
func NewContextLengthError(provider, message string, err error) *ContextLengthError {
	return &ContextLengthError{
		BaseError: BaseError{
			Code:     "context_length_exceeded",
			Message:  message,
			Provider: provider,
			Err:      err,
		},
	}
}

// NewContentFilterError creates a new ContentFilterError.
func NewContentFilterError(provider, message string, err error) *ContentFilterError {
	return &ContentFilterError{
		BaseError: BaseError{
			Code:     "content_filter",
			Message:  message,
			Provider: provider,
			Err:      err,
		},
	}
}

// NewModelNotFoundError creates a new ModelNotFoundError.
func NewModelNotFoundError(provider, message string, err error) *ModelNotFoundError {
	return &ModelNotFoundError{
		BaseError: BaseError{
			Code:     "model_not_found",
			Message:  message,
			Provider: provider,
			Err:      err,
		},
	}
}

// NewProviderError creates a new ProviderError.
func NewProviderError(provider, message string, statusCode int, err error) *ProviderError {
	return &ProviderError{
		BaseError: BaseError{
			Code:     "provider_error",
			Message:  message,
			Provider: provider,
			Err:      err,
		},
		StatusCode: statusCode,
	}
}

// NewMissingAPIKeyError creates a new MissingAPIKeyError.
func NewMissingAPIKeyError(provider, envVar string) *MissingAPIKeyError {
	return &MissingAPIKeyError{
		BaseError: BaseError{
			Code: "missing_api_key",
			Message: fmt.Sprintf(
				"API key not provided. Set %s environment variable or pass WithAPIKey option",
				envVar,
			),
			Provider: provider,
			Err:      ErrMissingAPIKey,
		},
		EnvVar: envVar,
	}
}

// NewUnsupportedProviderError creates a new UnsupportedProviderError.
func NewUnsupportedProviderError(provider string) *UnsupportedProviderError {
	return &UnsupportedProviderError{
		BaseError: BaseError{
			Code:     "unsupported_provider",
			Message:  fmt.Sprintf("provider %q is not supported", provider),
			Provider: provider,
			Err:      ErrUnsupportedProvider,
		},
	}
}

// NewUnsupportedParamError creates a new UnsupportedParamError.
func NewUnsupportedParamError(provider, param string) *UnsupportedParamError {
	return &UnsupportedParamError{
		BaseError: BaseError{
			Code:     "unsupported_parameter",
			Message:  fmt.Sprintf("parameter %q is not supported by provider %s", param, provider),
			Provider: provider,
			Err:      ErrUnsupportedParam,
		},
		Param: param,
	}
}

// Error type detection patterns (used for converting provider errors).
var (
	rateLimitPatterns = []string{
		"rate limit",
		"rate_limit",
		"ratelimit",
		"too many requests",
		"429",
		"quota exceeded",
		"throttl",
	}

	authPatterns = []string{
		"invalid api key",
		"invalid_api_key",
		"authentication",
		"unauthorized",
		"401",
		"api key",
		"apikey",
		"permission denied",
		"access denied",
	}

	contextLengthPatterns = []string{
		"context length",
		"context_length",
		"maximum context",
		"token limit",
		"too long",
		"exceeds the model",
		"max_tokens",
	}

	contentFilterPatterns = []string{
		"content filter",
		"content_filter",
		"safety",
		"blocked",
		"harmful",
		"policy violation",
	}

	modelNotFoundPatterns = []string{
		"model not found",
		"model_not_found",
		"does not exist",
		"no such model",
		"invalid model",
		"404",
	}
)

// Convert attempts to convert a provider error to an any-llm error type.
// If the error cannot be classified, it returns a generic ProviderError.
func Convert(provider string, err error) error {
	if err == nil {
		return nil
	}

	errStr := strings.ToLower(err.Error())

	// Check for rate limit errors
	for _, pattern := range rateLimitPatterns {
		if strings.Contains(errStr, pattern) {
			return NewRateLimitError(provider, err.Error(), err)
		}
	}

	// Check for authentication errors
	for _, pattern := range authPatterns {
		if strings.Contains(errStr, pattern) {
			return NewAuthenticationError(provider, err.Error(), err)
		}
	}

	// Check for context length errors
	for _, pattern := range contextLengthPatterns {
		if strings.Contains(errStr, pattern) {
			return NewContextLengthError(provider, err.Error(), err)
		}
	}

	// Check for content filter errors
	for _, pattern := range contentFilterPatterns {
		if strings.Contains(errStr, pattern) {
			return NewContentFilterError(provider, err.Error(), err)
		}
	}

	// Check for model not found errors
	for _, pattern := range modelNotFoundPatterns {
		if strings.Contains(errStr, pattern) {
			return NewModelNotFoundError(provider, err.Error(), err)
		}
	}

	// Default to generic provider error
	return NewProviderError(provider, err.Error(), 0, err)
}

// ConvertWithRegex uses regex patterns for more precise error detection.
func ConvertWithRegex(provider string, err error, patterns map[*regexp.Regexp]func(string, string, error) error) error {
	if err == nil {
		return nil
	}

	errStr := err.Error()

	for pattern, constructor := range patterns {
		if pattern.MatchString(errStr) {
			return constructor(provider, errStr, err)
		}
	}

	return Convert(provider, err)
}

// Is allows checking error types with errors.Is().
func (e *RateLimitError) Is(target error) bool {
	return target == ErrRateLimit
}

func (e *AuthenticationError) Is(target error) bool {
	return target == ErrAuthentication
}

func (e *InvalidRequestError) Is(target error) bool {
	return target == ErrInvalidRequest
}

func (e *ContextLengthError) Is(target error) bool {
	return target == ErrContextLength
}

func (e *ContentFilterError) Is(target error) bool {
	return target == ErrContentFilter
}

func (e *ModelNotFoundError) Is(target error) bool {
	return target == ErrModelNotFound
}

func (e *ProviderError) Is(target error) bool {
	return target == ErrProvider
}

func (e *MissingAPIKeyError) Is(target error) bool {
	return target == ErrMissingAPIKey
}

func (e *UnsupportedProviderError) Is(target error) bool {
	return target == ErrUnsupportedProvider
}

func (e *UnsupportedParamError) Is(target error) bool {
	return target == ErrUnsupportedParam
}
