package azureopenai_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/azureopenai"
)

func TestIntegrationCompletion(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"} {
		if os.Getenv(name) == "" {
			t.Skipf("%s not set", name)
		}
	}

	provider, err := azureopenai.New()
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()

	completion, err := provider.Completion(ctx, providers.CompletionParams{
		Model: os.Getenv("AZURE_OPENAI_DEPLOYMENT"),
		Messages: []providers.Message{
			{Role: providers.RoleUser, Content: "Reply with OK."},
		},
	})
	require.NoError(t, err)
	require.NotEmpty(t, completion.ID)
	require.Len(t, completion.Choices, 1)
	require.Equal(t, providers.RoleAssistant, completion.Choices[0].Message.Role)
	require.NotEmpty(t, completion.Choices[0].Message.Content)
}
