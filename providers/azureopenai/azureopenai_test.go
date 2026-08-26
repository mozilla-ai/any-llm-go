package azureopenai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
)

func TestNew(t *testing.T) {
	t.Run("creates provider with API key and endpoint", func(t *testing.T) {
		t.Parallel()

		provider, err := New(
			config.WithAPIKey("test-key"),
			config.WithBaseURL("https://example.openai.azure.com"),
		)
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, providerName, provider.Name())
	})

	t.Run("creates provider from environment variables", func(t *testing.T) {
		t.Setenv(envAPIKey, "env-key")
		t.Setenv(envBaseURL, "https://example.openai.azure.com")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv(envAPIKey, "")

		provider, err := New(config.WithBaseURL("https://example.openai.azure.com"))
		require.Nil(t, provider)
		require.Error(t, err)

		var missingKeyErr *errors.MissingAPIKeyError
		require.ErrorAs(t, err, &missingKeyErr)
		require.Equal(t, providerName, missingKeyErr.Provider)
		require.Equal(t, envAPIKey, missingKeyErr.EnvVar)
	})

	t.Run("returns error when endpoint is missing", func(t *testing.T) {
		t.Setenv(envAPIKey, "")
		t.Setenv(envBaseURL, "")

		provider, err := New(config.WithAPIKey("test-key"))
		require.Nil(t, provider)
		require.Error(t, err)
		require.Contains(t, err.Error(), "endpoint is required")
	})
}

func TestCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL("https://example.openai.azure.com"),
	)
	require.NoError(t, err)

	caps := provider.Capabilities()
	require.True(t, caps.Completion)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.True(t, caps.Embedding)
	require.True(t, caps.ListModels)
	require.True(t, caps.Responses)
}

func TestResolveAPIVersion(t *testing.T) {
	t.Run("defaults to preview", func(t *testing.T) {
		t.Setenv(envAPIVersion, "")

		cfg, err := config.New()
		require.NoError(t, err)
		require.Equal(t, defaultAPIVersion, resolveAPIVersion(cfg))
	})

	t.Run("uses environment variable", func(t *testing.T) {
		t.Setenv(envAPIVersion, "2025-04-01-preview")

		cfg, err := config.New()
		require.NoError(t, err)
		require.Equal(t, "2025-04-01-preview", resolveAPIVersion(cfg))
	})

	t.Run("uses extra config over environment", func(t *testing.T) {
		t.Setenv(envAPIVersion, "2024-10-21")

		cfg, err := config.New(config.WithExtra(extraAPIVersion, "2025-04-01-preview"))
		require.NoError(t, err)
		require.Equal(t, "2025-04-01-preview", resolveAPIVersion(cfg))
	})
}
