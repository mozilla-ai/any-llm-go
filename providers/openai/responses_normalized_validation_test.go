package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	anyerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestConvertResponsesParamsPreservesOptionalFieldsAndReasoning(t *testing.T) {
	t.Parallel()

	req, err := convertResponsesParams("test-provider", providers.ResponsesParams{})
	require.NoError(t, err)
	body, err := json.Marshal(req)
	require.NoError(t, err)
	require.JSONEq(t, `{}`, string(body))

	req, err = convertResponsesParams("test-provider", providers.ResponsesParams{
		ReasoningEffort: providers.ReasoningEffortAuto,
	})
	require.NoError(t, err)
	body, err = json.Marshal(req)
	require.NoError(t, err)
	require.JSONEq(t, `{}`, string(body))

	for _, effort := range []providers.ReasoningEffort{
		providers.ReasoningEffortNone,
		providers.ReasoningEffortMinimal,
		providers.ReasoningEffortLow,
		providers.ReasoningEffortMedium,
		providers.ReasoningEffortHigh,
		providers.ReasoningEffortXHigh,
		providers.ReasoningEffortMax,
	} {
		t.Run(string(effort), func(t *testing.T) {
			t.Parallel()

			params, convertErr := convertResponsesParams("test-provider", providers.ResponsesParams{
				ReasoningEffort: effort,
			})
			require.NoError(t, convertErr)
			encoded, marshalErr := json.Marshal(params)
			require.NoError(t, marshalErr)
			require.JSONEq(t, `{"reasoning":{"effort":"`+string(effort)+`"}}`, string(encoded))
		})
	}
}

func TestResponsesRejectsUnsupportedOrInvalidPortableRequests(t *testing.T) {
	t.Parallel()

	unsupported, err := NewCompatible(CompatibleConfig{Name: "unsupported"})
	require.NoError(t, err)
	result, err := unsupported.Responses(t.Context(), providers.ResponsesParams{})
	require.Nil(t, result)
	require.ErrorIs(t, err, anyerrors.ErrUnsupported)

	provider, err := NewCompatible(CompatibleConfig{
		Capabilities: providers.Capabilities{Responses: true},
		Name:         "test-provider",
	})
	require.NoError(t, err)

	result, err = provider.Responses(t.Context(), providers.ResponsesParams{
		Input: []providers.ResponsesInputItem{{Role: "unknown"}},
	})
	require.Nil(t, result)
	require.ErrorIs(t, err, anyerrors.ErrInvalidRequest)

	result, err = provider.Responses(t.Context(), providers.ResponsesParams{
		ReasoningEffort: providers.ReasoningEffort("turbo"),
	})
	require.Nil(t, result)
	require.ErrorIs(t, err, anyerrors.ErrUnsupportedParam)
}

func TestResponsesPreservesTerminalStatusDetails(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name              string
		fixture           string
		wantStatus        string
		wantError         *providers.ResponsesError
		wantIncomplete    *providers.ResponsesIncompleteDetails
		wantProviderError bool
	}{
		{
			name:       "empty completed output is valid",
			fixture:    `{"id":"resp_123","object":"response","model":"gpt-5.6-sol","status":"completed","output":[]}`,
			wantStatus: "completed",
		},
		{
			name:       "failed response carries its API error",
			fixture:    `{"id":"resp_123","object":"response","model":"gpt-5.6-sol","status":"failed","error":{"code":"server_error","message":"generation failed","trace_id":"trace_1"},"output":[]}`,
			wantStatus: "failed",
			wantError: &providers.ResponsesError{
				Code:    "server_error",
				Message: "generation failed",
			},
		},
		{
			name:           "incomplete response carries its reason",
			fixture:        `{"id":"resp_123","object":"response","model":"gpt-5.6-sol","status":"incomplete","incomplete_details":{"reason":"max_output_tokens","future_detail":true},"output":[]}`,
			wantStatus:     "incomplete",
			wantIncomplete: &providers.ResponsesIncompleteDetails{Reason: "max_output_tokens"},
		},
		{
			name:       "cancelled response remains inspectable",
			fixture:    `{"id":"resp_123","object":"response","model":"gpt-5.6-sol","status":"cancelled","output":[]}`,
			wantStatus: "cancelled",
		},
		{
			name:       "omitted optional status remains inspectable",
			fixture:    `{"id":"resp_123","object":"response","model":"gpt-5.6-sol","output":[]}`,
			wantStatus: "",
		},
		{
			name:              "missing response identity and output is malformed",
			fixture:           `{"id":"resp_123","model":"gpt-5.6-sol"}`,
			wantProviderError: true,
		},
		{
			name:              "wrong object type is malformed",
			fixture:           `{"id":"resp_123","object":"chat.completion","model":"gpt-5.6-sol","output":[]}`,
			wantProviderError: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, writeErr := io.WriteString(w, tc.fixture)
				require.NoError(t, writeErr)
			}))
			t.Cleanup(server.Close)

			provider, err := NewCompatible(CompatibleConfig{
				Capabilities:   providers.Capabilities{Responses: true},
				DefaultAPIKey:  "test-key",
				DefaultBaseURL: server.URL + "/v1",
				Name:           "test-provider",
			})
			require.NoError(t, err)

			result, err := provider.Responses(t.Context(), providers.ResponsesParams{})
			if tc.wantProviderError {
				require.Nil(t, result)
				require.ErrorIs(t, err, anyerrors.ErrProvider)
				return
			}

			require.NoError(t, err)
			require.Equal(t, tc.wantStatus, result.Status)
			require.Empty(t, result.OutputText)
			require.Empty(t, result.OutputItems)
			require.Nil(t, result.Usage)
			if tc.wantError == nil {
				require.Nil(t, result.Error)
			} else {
				require.NotNil(t, result.Error)
				require.Equal(t, tc.wantError.Code, result.Error.Code)
				require.Equal(t, tc.wantError.Message, result.Error.Message)
				require.Contains(t, string(result.Error.ProviderRaw), `"trace_id":"trace_1"`)
			}
			if tc.wantIncomplete == nil {
				require.Nil(t, result.IncompleteDetails)
			} else {
				require.NotNil(t, result.IncompleteDetails)
				require.Equal(t, tc.wantIncomplete.Reason, result.IncompleteDetails.Reason)
				require.Contains(t, string(result.IncompleteDetails.ProviderRaw), `"future_detail":true`)
			}
		})
	}
}
