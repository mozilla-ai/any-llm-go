package llamacpp

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/internal/testutil"
)

const (
	// testModel is the name typically returned by /v1/models when only one model is loaded.
	testModel = "llama.cpp"

	// testLlamacppAvailabilityTimeout is how long we wait to check if the server is alive.
	testLlamacppAvailabilityTimeout = 5 * time.Second
)

// TestNew checks that the provider can be created with defaults or custom options.
func TestNew(t *testing.T) {
	t.Parallel()

	t.Run("creates provider with defaults", func(t *testing.T) {
		p, err := New()
		require.NoError(t, err)
		require.NotNil(t, p)
	})

	t.Run("creates provider with custom options", func(t *testing.T) {
		p, err := New(anyllm.WithBaseURL("http://custom:8080"))
		require.NoError(t, err)
		require.NotNil(t, p)
	})
}

// TestProviderName verifies the provider returns the correct identifier.
func TestProviderName(t *testing.T) {
	t.Parallel()

	p, err := New()
	require.NoError(t, err)
	require.Equal(t, providerName, p.Name())
}

// TestCapabilities confirms the provider advertises the expected feature set.
func TestCapabilities(t *testing.T) {
	t.Parallel()

	p, err := New()
	require.NoError(t, err)

	caps := p.Capabilities()
	require.True(t, caps.Completion)
	require.True(t, caps.CompletionStreaming)
	require.True(t, caps.CompletionTools)
	require.True(t, caps.Embedding)
	require.True(t, caps.ListModels)
}

// TestIntegration runs real calls against a live llama.cpp server.
//
// Skipped automatically if no server is responding on the default port.
func TestIntegration(t *testing.T) {
	t.Parallel()

	skipIfLlamacppUnavailable(t)

	p, err := New(anyllm.WithTimeout(30 * time.Second))
	require.NoError(t, err)

	ctx := context.Background()

	t.Run("ListModels returns at least one model", func(t *testing.T) {
		t.Parallel()

		models, err := p.ListModels(ctx)
		require.NoError(t, err)
		require.NotEmpty(t, models.Data)
	})

	t.Run("Completion generates text", func(t *testing.T) {
		t.Parallel()

		resp, err := p.Completion(ctx, anyllm.CompletionParams{
			Model:    testModel,
			Messages: testutil.MessagesWithSystem(),
		})
		require.NoError(t, err)
		require.Len(t, resp.Choices, 1)
		require.NotEmpty(t, resp.Choices[0].Message.Content)
	})

	t.Run("Embedding generates vectors", func(t *testing.T) {
		t.Parallel()

		resp, err := p.Embedding(ctx, anyllm.EmbeddingParams{
			Model: testModel,
			Input: []string{"test"},
		})
		require.NoError(t, err)
		require.NotEmpty(t, resp.Data)
	})
}

// skipIfLlamacppUnavailable skips the test if no functional llama.cpp server is detected.
//
// It makes a quick /v1/models call to check. Waits up to 5 seconds.
func skipIfLlamacppUnavailable(t *testing.T) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), testLlamacppAvailabilityTimeout)
	defer cancel()

	p, err := New()
	if err != nil {
		t.Skip("llamacpp not available: failed to create provider")
	}

	if _, err = p.ListModels(ctx); err != nil {
		t.Skipf("llamacpp not available: server not responding at %s (error: %v)", defaultBaseURL, err)
	}
}
