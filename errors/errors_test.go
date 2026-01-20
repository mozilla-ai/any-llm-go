package errors

import (
	stderrors "errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestErrorIs(t *testing.T) {
	t.Parallel()

	originalErr := stderrors.New("original error")

	tests := []struct {
		name      string
		err       error
		target    error
		wantMatch bool
	}{
		{
			name:      "RateLimitError matches ErrRateLimit",
			err:       NewRateLimitError("openai", originalErr),
			target:    ErrRateLimit,
			wantMatch: true,
		},
		{
			name:      "RateLimitError does not match ErrAuthentication",
			err:       NewRateLimitError("openai", originalErr),
			target:    ErrAuthentication,
			wantMatch: false,
		},
		{
			name:      "AuthenticationError matches ErrAuthentication",
			err:       NewAuthenticationError("anthropic", originalErr),
			target:    ErrAuthentication,
			wantMatch: true,
		},
		{
			name:      "InvalidRequestError matches ErrInvalidRequest",
			err:       NewInvalidRequestError("openai", originalErr),
			target:    ErrInvalidRequest,
			wantMatch: true,
		},
		{
			name:      "ContextLengthError matches ErrContextLength",
			err:       NewContextLengthError("openai", originalErr),
			target:    ErrContextLength,
			wantMatch: true,
		},
		{
			name:      "ContentFilterError matches ErrContentFilter",
			err:       NewContentFilterError("anthropic", originalErr),
			target:    ErrContentFilter,
			wantMatch: true,
		},
		{
			name:      "ModelNotFoundError matches ErrModelNotFound",
			err:       NewModelNotFoundError("openai", originalErr),
			target:    ErrModelNotFound,
			wantMatch: true,
		},
		{
			name:      "ProviderError matches ErrProvider",
			err:       NewProviderError("openai", originalErr),
			target:    ErrProvider,
			wantMatch: true,
		},
		{
			name:      "MissingAPIKeyError matches ErrMissingAPIKey",
			err:       NewMissingAPIKeyError("openai", "OPENAI_API_KEY"),
			target:    ErrMissingAPIKey,
			wantMatch: true,
		},
		{
			name:      "UnsupportedProviderError matches ErrUnsupportedProvider",
			err:       NewUnsupportedProviderError("unknown"),
			target:    ErrUnsupportedProvider,
			wantMatch: true,
		},
		{
			name:      "UnsupportedParamError matches ErrUnsupportedParam",
			err:       NewUnsupportedParamError("openai", "unsupported_field"),
			target:    ErrUnsupportedParam,
			wantMatch: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := stderrors.Is(tc.err, tc.target)
			require.Equal(t, tc.wantMatch, got)
		})
	}
}

func TestErrorMessage(t *testing.T) {
	t.Parallel()

	originalErr := stderrors.New("something went wrong")

	tests := []struct {
		name        string
		err         error
		wantContain []string
	}{
		{
			name:        "RateLimitError includes provider and code",
			err:         NewRateLimitError("openai", originalErr),
			wantContain: []string{"[openai]", "rate_limit", "something went wrong"},
		},
		{
			name:        "AuthenticationError includes provider and code",
			err:         NewAuthenticationError("anthropic", originalErr),
			wantContain: []string{"[anthropic]", "auth_error", "something went wrong"},
		},
		{
			name:        "MissingAPIKeyError includes env var hint",
			err:         NewMissingAPIKeyError("openai", "OPENAI_API_KEY"),
			wantContain: []string{"[openai]", "missing_api_key", "OPENAI_API_KEY"},
		},
		{
			name:        "UnsupportedProviderError includes provider name",
			err:         NewUnsupportedProviderError("unknown_provider"),
			wantContain: []string{"[unknown_provider]", "unsupported_provider"},
		},
		{
			name:        "UnsupportedParamError includes param name",
			err:         NewUnsupportedParamError("openai", "bad_param"),
			wantContain: []string{"[openai]", "unsupported_parameter", "bad_param"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			msg := tc.err.Error()
			for _, want := range tc.wantContain {
				require.Contains(t, msg, want)
			}
		})
	}
}

func TestErrorCodes(t *testing.T) {
	t.Parallel()

	t.Run("RateLimitError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewRateLimitError("openai", nil)
		require.Equal(t, CodeRateLimit, err.Code)
	})

	t.Run("AuthenticationError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewAuthenticationError("openai", nil)
		require.Equal(t, CodeAuthError, err.Code)
	})

	t.Run("InvalidRequestError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewInvalidRequestError("openai", nil)
		require.Equal(t, CodeInvalidRequest, err.Code)
	})

	t.Run("ContextLengthError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewContextLengthError("openai", nil)
		require.Equal(t, CodeContextLength, err.Code)
	})

	t.Run("ContentFilterError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewContentFilterError("openai", nil)
		require.Equal(t, CodeContentFilter, err.Code)
	})

	t.Run("ModelNotFoundError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewModelNotFoundError("openai", nil)
		require.Equal(t, CodeModelNotFound, err.Code)
	})

	t.Run("ProviderError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewProviderError("openai", nil)
		require.Equal(t, CodeProviderError, err.Code)
	})

	t.Run("MissingAPIKeyError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewMissingAPIKeyError("openai", "OPENAI_API_KEY")
		require.Equal(t, CodeMissingAPIKey, err.Code)
	})

	t.Run("UnsupportedProviderError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewUnsupportedProviderError("unknown")
		require.Equal(t, CodeUnsupportedProvider, err.Code)
	})

	t.Run("UnsupportedParamError has correct code", func(t *testing.T) {
		t.Parallel()
		err := NewUnsupportedParamError("openai", "param")
		require.Equal(t, CodeUnsupportedParam, err.Code)
	})
}

func TestErrorAs(t *testing.T) {
	t.Parallel()

	t.Run("can extract RateLimitError with RetryAfter", func(t *testing.T) {
		t.Parallel()

		err := NewRateLimitError("openai", stderrors.New("rate limited"))
		err.RetryAfter = 30

		var rateErr *RateLimitError
		require.True(t, stderrors.As(err, &rateErr))
		require.Equal(t, 30, rateErr.RetryAfter)
		require.Equal(t, "openai", rateErr.Provider)
	})

	t.Run("can extract MissingAPIKeyError with EnvVar", func(t *testing.T) {
		t.Parallel()

		err := NewMissingAPIKeyError("anthropic", "ANTHROPIC_API_KEY")

		var keyErr *MissingAPIKeyError
		require.True(t, stderrors.As(err, &keyErr))
		require.Equal(t, "ANTHROPIC_API_KEY", keyErr.EnvVar)
		require.Equal(t, "anthropic", keyErr.Provider)
	})

	t.Run("can extract UnsupportedParamError with Param", func(t *testing.T) {
		t.Parallel()

		err := NewUnsupportedParamError("openai", "frequency_penalty")

		var paramErr *UnsupportedParamError
		require.True(t, stderrors.As(err, &paramErr))
		require.Equal(t, "frequency_penalty", paramErr.Param)
		require.Equal(t, "openai", paramErr.Provider)
	})
}
