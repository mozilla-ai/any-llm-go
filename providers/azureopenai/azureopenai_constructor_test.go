package azureopenai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
)

func TestNew(t *testing.T) {
	t.Parallel()

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL("https://example.openai.azure.com"),
	)
	require.NoError(t, err)
	require.Equal(t, providerName, provider.Name())
	require.Equal(t, capabilities(), provider.Capabilities())
}

func TestNewFromEnvironment(t *testing.T) {
	t.Setenv(envAPIKey, "env-key")
	t.Setenv(envBaseURL, "https://example.openai.azure.com")

	provider, err := New()
	require.NoError(t, err)
	require.NotNil(t, provider)
}

func TestNewAppliesOptionsOnce(t *testing.T) {
	t.Parallel()

	applied := 0
	option := func(cfg *config.Config) error {
		err := config.WithAPIKey("test-key")(cfg)
		if err != nil {
			return err
		}

		applied++

		return config.WithBaseURL("https://example.openai.azure.com")(cfg)
	}

	provider, err := New(option)
	require.NoError(t, err)
	require.NotNil(t, provider)
	require.Equal(t, 1, applied)
}

func TestNewRequiresAPIKey(t *testing.T) {
	t.Setenv(envAPIKey, "")

	provider, err := New(config.WithBaseURL("https://example.openai.azure.com"))
	require.Nil(t, provider)

	var missingKeyErr *errors.MissingAPIKeyError
	require.ErrorAs(t, err, &missingKeyErr)
	require.Equal(t, providerName, missingKeyErr.Provider)
	require.Equal(t, envAPIKey, missingKeyErr.EnvVar)
}

func TestNewRequiresEndpoint(t *testing.T) {
	t.Setenv(envBaseURL, "")

	provider, err := New(config.WithAPIKey("test-key"))
	require.Nil(t, provider)
	require.ErrorContains(t, err, "endpoint is required")
}

func TestNewRequiresHTTPS(t *testing.T) {
	t.Parallel()

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL("http://example.openai.azure.com"),
	)
	require.Nil(t, provider)
	require.ErrorContains(t, err, "must use HTTPS")
}

func TestNewRejectsEndpointURLMetadata(t *testing.T) {
	t.Parallel()

	for _, endpoint := range []string{
		"https://user@example.openai.azure.com",
		"https://example.openai.azure.com?api-version=v1",
		"https://example.openai.azure.com#fragment",
	} {
		provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(endpoint))
		require.Nil(t, provider)
		require.ErrorContains(t, err, "must not contain")
	}
}

func TestNewRejectsAuthenticationHeaderOverrides(t *testing.T) {
	t.Parallel()

	for _, header := range []string{"Api-Key", "Authorization"} {
		t.Run(header, func(t *testing.T) {
			t.Parallel()

			provider, err := New(
				config.WithAPIKey("test-key"),
				config.WithBaseURL("https://example.openai.azure.com"),
				config.WithHeader(header, "other-credential"),
			)
			require.Nil(t, provider)
			require.ErrorContains(t, err, "authentication headers")
		})
	}
}

func TestNewRejectsUnsupportedExtra(t *testing.T) {
	t.Parallel()

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL("https://example.openai.azure.com"),
		config.WithExtra("api_version", "2025-04-01-preview"),
	)
	require.Nil(t, provider)
	require.ErrorContains(t, err, "does not support extra options")
}

func TestV1BaseURL(t *testing.T) {
	t.Parallel()

	require.Equal(t, "https://example.openai.azure.com/openai/v1/", v1BaseURL("https://example.openai.azure.com/"))
	require.Equal(
		t,
		"https://example.services.ai.azure.com/openai/v1/",
		v1BaseURL("https://example.services.ai.azure.com/openai/v1/"),
	)
}
