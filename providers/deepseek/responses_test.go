package deepseek

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	openaisdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	anyerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const deepSeekResponseFixture = `{
  "id":"resp_deepseek",
  "object":"response",
  "created_at":1,
  "status":"completed",
  "error":null,
  "incomplete_details":null,
  "model":"deepseek-v4-flash",
  "output":[{
    "type":"message",
    "id":"msg_1",
    "status":"completed",
    "role":"assistant",
    "content":[{"type":"output_text","text":"hello","annotations":[],"logprobs":[]}]
  }],
  "usage":{
    "input_tokens":8,
    "input_tokens_details":{"cached_tokens":3},
    "output_tokens":5,
    "output_tokens_details":{"reasoning_tokens":2},
    "total_tokens":13
  },
  "store":false,
  "previous_response_id":null,
  "parallel_tool_calls":true
}`

func TestResponsesUsesDeepSeekWireAndNormalizesUsage(t *testing.T) {
	t.Parallel()

	var request json.RawMessage
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/responses", r.URL.Path)
		require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		w.Header().Set("Content-Type", "application/json")
		_, err := io.WriteString(w, deepSeekResponseFixture)
		require.NoError(t, err)
	}))
	t.Cleanup(server.Close)

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	result, err := provider.Responses(t.Context(), providers.ResponsesParams{
		Input: []providers.ResponsesInputItem{{
			Role:    providers.ResponsesInputRoleUser,
			Content: "hello",
		}},
		Instructions:    new(""),
		MaxOutputTokens: new(128),
		Model:           "deepseek-v4-flash",
		ReasoningEffort: providers.ReasoningEffortMinimal,
	})
	require.NoError(t, err)

	require.JSONEq(t, `{
      "input":[{"role":"user","content":"hello"}],
      "instructions":"",
      "max_output_tokens":128,
      "model":"deepseek-v4-flash",
      "reasoning":{"effort":"minimal"}
    }`, string(request))
	require.Equal(t, "deepseek-v4-flash", result.Model)
	require.Equal(t, "completed", result.Status)
	require.Equal(t, "hello", result.OutputText)
	require.Equal(t, &providers.ResponsesUsage{
		InputTokens:     8,
		OutputTokens:    5,
		TotalTokens:     13,
		CachedTokens:    3,
		ReasoningTokens: 2,
	}, result.Usage)
	require.Contains(t, string(result.ProviderRaw), `"store":false`)
}

func TestResponsesRejectsMissingDeepSeekRequirements(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		requests.Add(1)
	}))
	t.Cleanup(server.Close)

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	for _, tc := range []struct {
		name   string
		params providers.ResponsesParams
		field  string
	}{
		{
			name: "model",
			params: providers.ResponsesParams{
				Input: []providers.ResponsesInputItem{{Role: providers.ResponsesInputRoleUser}},
			},
			field: "model",
		},
		{
			name:   "input and instructions",
			params: providers.ResponsesParams{Model: "deepseek-v4-flash"},
			field:  "input or instructions",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, responseErr := provider.Responses(t.Context(), tc.params)
			require.Nil(t, result)
			require.ErrorIs(t, responseErr, anyerrors.ErrInvalidRequest)
			require.Contains(t, responseErr.Error(), tc.field)
			require.Contains(t, responseErr.Error(), "[deepseek]")
		})
	}
	require.Zero(t, requests.Load(), "invalid requests must fail before transport")
}

func TestStreamResponseUsesDeepSeekSemanticEvents(t *testing.T) {
	t.Parallel()

	var compactResponse bytes.Buffer
	require.NoError(t, json.Compact(&compactResponse, []byte(deepSeekResponseFixture)))
	completedResponse := compactResponse.String()
	inProgressResponse := strings.Replace(
		completedResponse,
		`"status":"completed"`,
		`"status":"in_progress"`,
		1,
	)
	var request map[string]json.RawMessage
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/responses", r.URL.Path)
		require.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := io.WriteString(w,
			"data: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":"+
				inProgressResponse+"}\n\n"+
				"data: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":"+
				completedResponse+"}\n\n",
		)
		require.NoError(t, err)
	}))
	t.Cleanup(server.Close)

	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL(server.URL),
	)
	require.NoError(t, err)

	events, errs := provider.StreamResponse(t.Context(), responses.ResponseNewParams{
		Input: responses.ResponseNewParamsInputUnion{OfString: openaisdk.String("hello")},
		Model: shared.ResponsesModel("deepseek-v4-flash"),
	})
	var eventTypes []string
	for event := range events {
		eventTypes = append(eventTypes, event.Type)
	}
	require.NoError(t, <-errs)
	require.Equal(t, []string{"response.created", "response.completed"}, eventTypes)
	require.JSONEq(t, `true`, string(request["stream"]))
}
