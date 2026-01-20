package platform

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestNew(t *testing.T) {
	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("ANY_LLM_KEY", "")

		provider, err := New()

		require.Nil(t, provider)
		require.Error(t, err)
		require.ErrorIs(t, err, errors.ErrMissingAPIKey)
	})

	t.Run("creates provider with API key from env", func(t *testing.T) {
		t.Setenv("ANY_LLM_KEY", "ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI=")

		provider, err := New()

		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, "platform", provider.Name())
	})

	t.Run("creates provider with explicit API key", func(t *testing.T) {
		t.Setenv("ANY_LLM_KEY", "")

		provider, err := New(config.WithAPIKey("ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI="))

		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

func TestProvider_Name(t *testing.T) {
	t.Setenv("ANY_LLM_KEY", "ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI=")

	provider, err := New()
	require.NoError(t, err)

	require.Equal(t, "platform", provider.Name())
}

func TestProvider_Capabilities(t *testing.T) {
	t.Setenv("ANY_LLM_KEY", "ANY.v1.test.fingerprint-dGVzdHByaXZhdGVrZXkxMjM0NTY3ODkwMTI=")

	provider, err := New()
	require.NoError(t, err)

	caps := provider.Capabilities()

	require.True(t, caps.Completion)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionReasoning)
	require.True(t, caps.Embedding)
}

func TestParseModelString(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input        string
		wantProvider string
		wantModel    string
	}{
		{"openai:gpt-4o-mini", "openai", "gpt-4o-mini"},
		{"anthropic:claude-3-5-haiku-latest", "anthropic", "claude-3-5-haiku-latest"},
		{"gpt-4o-mini", "", "gpt-4o-mini"},
		{"provider:model:with:colons", "provider", "model:with:colons"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			t.Parallel()

			provider, model := parseModelString(tc.input)
			require.Equal(t, tc.wantProvider, provider)
			require.Equal(t, tc.wantModel, model)
		})
	}
}

// Integration tests - require actual platform connection and ANY_LLM_KEY

func TestIntegrationOpenAICompletion(t *testing.T) {
	t.Parallel()

	if os.Getenv("ANY_LLM_KEY") == "" {
		t.Skip("ANY_LLM_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	// Request a completion through the platform, which will:
	// 1. Authenticate with the platform using ANY_LLM_KEY
	// 2. Get the decrypted OpenAI API key from the platform
	// 3. Create an OpenAI provider and delegate the request
	// 4. Report usage metrics back to the platform
	response, err := provider.Completion(ctx, providers.CompletionParams{
		Model: "openai:gpt-4o-mini",
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Say 'hello' and nothing else."},
		},
	})
	require.NoError(t, err)

	require.NotNil(t, response)
	require.NotEmpty(t, response.Choices)
	require.NotEmpty(t, response.Choices[0].Message.Content)

	content, ok := response.Choices[0].Message.Content.(string)
	require.True(t, ok, "Content should be a string")
	require.True(t, strings.Contains(strings.ToLower(content), "hello"))

	// Verify usage was tracked
	require.NotNil(t, response.Usage)
	require.Greater(t, response.Usage.TotalTokens, 0)

	t.Logf("Response: %s", content)
	t.Logf("Tokens used: %d", response.Usage.TotalTokens)

	// Wait a bit for the usage event goroutine to complete
	time.Sleep(2 * time.Second)
}

func TestIntegrationOpenAIStreaming(t *testing.T) {
	t.Parallel()

	if os.Getenv("ANY_LLM_KEY") == "" {
		t.Skip("ANY_LLM_KEY not set")
	}

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()

	chunks, errs := provider.CompletionStream(ctx, providers.CompletionParams{
		Model: "openai:gpt-4o-mini",
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Count from 1 to 5, one number per line."},
		},
		Stream: true,
	})

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	require.Greater(t, chunkCount, 0, "Should have received chunks")
	require.NotEmpty(t, content.String(), "Should have received content")

	t.Logf("Received %d chunks", chunkCount)
	t.Logf("Content: %s", content.String())

	// Wait a bit for the usage event goroutine to complete
	time.Sleep(2 * time.Second)
}
