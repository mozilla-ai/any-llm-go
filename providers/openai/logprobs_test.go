package openai

import (
	"encoding/json"
	"testing"

	oaisdk "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"

	llmerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestConvertParamsPreservesLogprobControls(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name        string
		logprobs    *bool
		topLogprobs *int
	}{
		{name: "omitted"},
		{name: "explicit false", logprobs: new(false)},
		{name: "zero alternatives", logprobs: new(true), topLogprobs: new(0)},
		{name: "twenty alternatives", logprobs: new(true), topLogprobs: new(20)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			body, err := json.Marshal(convertParams(providers.CompletionParams{
				Model:       "gpt-5.6",
				Messages:    []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
				Logprobs:    tc.logprobs,
				TopLogprobs: tc.topLogprobs,
			}))
			require.NoError(t, err)

			var wire struct {
				Logprobs    *bool  `json:"logprobs"`
				TopLogprobs *int64 `json:"top_logprobs"`
			}
			require.NoError(t, json.Unmarshal(body, &wire))
			require.Equal(t, tc.logprobs, wire.Logprobs)
			if tc.topLogprobs == nil {
				require.Nil(t, wire.TopLogprobs)
			} else {
				require.Equal(t, int64(*tc.topLogprobs), *wire.TopLogprobs)
			}
		})
	}
}

func TestValidateCompletionParamsRejectsInvalidLogprobControls(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name        string
		logprobs    *bool
		topLogprobs int
	}{
		{name: "logprobs omitted", topLogprobs: 1},
		{name: "logprobs false", logprobs: new(false), topLogprobs: 1},
		{name: "negative alternatives", logprobs: new(true), topLogprobs: -1},
		{name: "too many alternatives", logprobs: new(true), topLogprobs: 21},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := validateCompletionParams(providers.CompletionParams{
				Model:       "gpt-5.6",
				Messages:    []providers.Message{{Role: providers.RoleUser, Content: "hello"}},
				Logprobs:    tc.logprobs,
				TopLogprobs: &tc.topLogprobs,
			})
			require.ErrorIs(t, err, llmerrors.ErrInvalidRequest)
		})
	}
}

func TestConvertResponsePreservesLogprobs(t *testing.T) {
	t.Parallel()

	var source oaisdk.ChatCompletion
	require.NoError(t, json.Unmarshal([]byte(`{
		"id":"chatcmpl-test","object":"chat.completion","created":1,"model":"gpt-5.6",
		"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"A"},
		"logprobs":{"content":[{"token":"A","bytes":null,"logprob":-0.1,"top_logprobs":[
			{"token":"B","bytes":[66],"logprob":-9999.0}
		]}],"refusal":[{"token":"no","bytes":[110,111],"logprob":-0.2,"top_logprobs":[]}]}}]
	}`), &source))

	result := convertResponse(&source)
	require.Equal(t, &providers.ChatCompletionLogprobs{
		Content: []providers.ChatCompletionTokenLogprob{{
			Token: "A", Logprob: -0.1,
			TopLogprobs: []providers.ChatCompletionTopLogprob{{Token: "B", Bytes: []int{66}, Logprob: -9999}},
		}},
		Refusal: []providers.ChatCompletionTokenLogprob{{
			Token: "no", Bytes: []int{110, 111}, Logprob: -0.2,
			TopLogprobs: []providers.ChatCompletionTopLogprob{},
		}},
	}, result.Choices[0].Logprobs)
}

func TestConvertChunkPreservesLogprobs(t *testing.T) {
	t.Parallel()

	var source oaisdk.ChatCompletionChunk
	require.NoError(t, json.Unmarshal([]byte(`{
		"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-5.6",
		"choices":[{"index":0,"finish_reason":null,"delta":{"content":"A"},
		"logprobs":{"content":[{"token":"A","bytes":[65],"logprob":-0.1,"top_logprobs":[]}]}}]
	}`), &source))

	result := convertChunk(&source)
	require.Equal(t, &providers.ChatCompletionLogprobs{
		Content: []providers.ChatCompletionTokenLogprob{{
			Token: "A", Bytes: []int{65}, Logprob: -0.1,
			TopLogprobs: []providers.ChatCompletionTopLogprob{},
		}},
	}, result.Choices[0].Logprobs)
}
