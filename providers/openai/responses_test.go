package openai

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestConvertResponsesParams(t *testing.T) {
	t.Parallel()

	t.Run("converts input items and instructions", func(t *testing.T) {
		t.Parallel()

		params := providers.ResponsesParams{
			Model:        "gpt-4o-mini",
			Instructions: "you are a shell assistant",
			Input: []providers.ResponsesInputItem{
				{Role: providers.RoleUser, Content: "list files"},
				{Role: providers.RoleAssistant, Content: "=ls"},
			},
		}

		req, err := convertResponsesParams(params)
		require.NoError(t, err)
		require.Equal(t, "gpt-4o-mini", string(req.Model))
		require.Equal(t, "you are a shell assistant", req.Instructions.Value)
		require.Len(t, req.Input.OfInputItemList, 2)
	})

	t.Run("sets reasoning effort", func(t *testing.T) {
		t.Parallel()

		params := providers.ResponsesParams{
			Model:     "o3-mini",
			Reasoning: providers.ReasoningEffortMedium,
			Input: []providers.ResponsesInputItem{
				{Role: providers.RoleUser, Content: "think"},
			},
		}

		req, err := convertResponsesParams(params)
		require.NoError(t, err)
		require.Equal(t, "medium", string(req.Reasoning.Effort))
	})

	t.Run("rejects unknown roles", func(t *testing.T) {
		t.Parallel()

		params := providers.ResponsesParams{
			Model: "gpt-4o-mini",
			Input: []providers.ResponsesInputItem{
				{Role: "moderator", Content: "nope"},
			},
		}

		_, err := convertResponsesParams(params)
		require.Error(t, err)
		require.Contains(t, err.Error(), "unsupported responses role")
	})
}

func TestValidateResponsesParams(t *testing.T) {
	t.Parallel()

	t.Run("requires model", func(t *testing.T) {
		t.Parallel()
		err := validateResponsesParams(providers.ResponsesParams{
			Input: []providers.ResponsesInputItem{{Role: providers.RoleUser, Content: "hi"}},
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "model is required")
	})

	t.Run("requires input", func(t *testing.T) {
		t.Parallel()
		err := validateResponsesParams(providers.ResponsesParams{Model: "gpt-4o-mini"})
		require.Error(t, err)
		require.Contains(t, err.Error(), "at least one input item is required")
	})
}

func TestCapabilitiesIncludesResponses(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"))
	require.NoError(t, err)
	require.True(t, provider.Capabilities().Responses)
}
