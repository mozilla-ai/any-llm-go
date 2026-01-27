package llamafile

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Test constants.
const (
	testLlamafileAvailabilityTimeout = 5 * time.Second

	// Expected object types in API responses.
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectList                = "list"
)

func TestNew(t *testing.T) {
	// Note: Not using t.Parallel() here because child test uses t.Setenv.

	t.Run("creates provider with default settings", func(t *testing.T) {
		t.Parallel()

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
		require.Equal(t, providerName, provider.Name())
	})

	t.Run("creates provider with custom base URL", func(t *testing.T) {
		t.Parallel()

		provider, err := New(config.WithBaseURL("http://localhost:8081/v1"))
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("creates provider from LLAMAFILE_BASE_URL environment variable", func(t *testing.T) {
		t.Setenv("LLAMAFILE_BASE_URL", "http://custom-host:8080/v1")

		provider, err := New()
		require.NoError(t, err)
		require.NotNil(t, provider)
	})
}

func TestCapabilities(t *testing.T) {
	t.Parallel()

	provider, err := New()
	require.NoError(t, err)

	caps := provider.Capabilities()

	require.True(t, caps.Completion)
	require.True(t, caps.CompletionStreaming)
	require.False(t, caps.CompletionReasoning)
	require.True(t, caps.CompletionImage)
	require.False(t, caps.CompletionPDF)
	require.True(t, caps.Embedding)
	require.True(t, caps.ListModels)
}

func TestProviderName(t *testing.T) {
	t.Parallel()

	provider, err := New()
	require.NoError(t, err)
	require.Equal(t, "llamafile", provider.Name())
}

// Integration tests - only run if Llamafile is available.

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()
	skipIfLlamafileUnavailable(t)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.SimpleMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Equal(t, objectChatCompletion, resp.Object)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
	require.Equal(t, providers.RoleAssistant, resp.Choices[0].Message.Role)
}

func TestIntegrationCompletionWithSystemMessage(t *testing.T) {
	t.Parallel()
	skipIfLlamafileUnavailable(t)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.MessagesWithSystem(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.Content)
}

func TestIntegrationCompletionStream(t *testing.T) {
	t.Parallel()
	skipIfLlamafileUnavailable(t)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.SimpleMessages(),
		Stream:   true,
	}

	chunks, errs := provider.CompletionStream(ctx, params)

	var content strings.Builder
	chunkCount := 0

	for chunk := range chunks {
		chunkCount++
		require.Equal(t, objectChatCompletionChunk, chunk.Object)
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	err = <-errs
	require.NoError(t, err)

	require.Greater(t, chunkCount, 0)
	require.NotEmpty(t, content.String())
}

func TestIntegrationListModels(t *testing.T) {
	t.Parallel()
	skipIfLlamafileUnavailable(t)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	resp, err := provider.ListModels(ctx)
	require.NoError(t, err)

	require.Equal(t, objectList, resp.Object)
	// Llamafile typically has at least one model loaded.
	require.NotEmpty(t, resp.Data)
}

func TestIntegrationConversation(t *testing.T) {
	t.Parallel()
	skipIfLlamafileUnavailable(t)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.CompletionParams{
		Model:    testutil.TestModel(providerName),
		Messages: testutil.ConversationMessages(),
	}

	resp, err := provider.Completion(ctx, params)
	require.NoError(t, err)

	require.NotEmpty(t, resp.ID)
	require.Len(t, resp.Choices, 1)

	// The model should remember the name "Alice".
	contentStr, ok := resp.Choices[0].Message.Content.(string)
	require.True(t, ok)
	require.Contains(t, strings.ToLower(contentStr), "alice")
}

func TestIntegrationEmbedding(t *testing.T) {
	t.Parallel()
	skipIfLlamafileUnavailable(t)

	provider, err := New()
	require.NoError(t, err)

	ctx := context.Background()
	params := providers.EmbeddingParams{
		Model: testutil.EmbeddingModel(providerName),
		Input: "Hello, world!",
	}

	resp, err := provider.Embedding(ctx, params)
	if err != nil {
		// Embedding model may not be available.
		t.Skipf("Embedding not available: %v", err)
	}

	require.Equal(t, objectList, resp.Object)
	require.NotEmpty(t, resp.Data)
	require.NotEmpty(t, resp.Data[0].Embedding)
}

// skipIfLlamafileUnavailable skips the test if Llamafile is not running.
func skipIfLlamafileUnavailable(t *testing.T) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), testLlamafileAvailabilityTimeout)
	defer cancel()

	provider, err := New()
	if err != nil {
		t.Skip("Llamafile not available")
	}

	if _, err = provider.ListModels(ctx); err != nil {
		t.Skip("Llamafile not available")
	}
}
