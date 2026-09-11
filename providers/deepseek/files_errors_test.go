package deepseek

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	llmerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

func TestFileOperationsRejectInvalidDeepSeekParamsBeforeTransport(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		requests.Add(1)
	}))
	t.Cleanup(server.Close)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)
	t.Cleanup(func() { require.Zero(t, requests.Load()) })

	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{name: "missing file", run: func() error {
			_, err := provider.UploadFile(t.Context(), providers.UploadFileParams{Purpose: providers.FilePurposeUserData})

			return err
		}},
		{name: "wrong purpose", run: func() error {
			_, err := provider.UploadFile(t.Context(), providers.UploadFileParams{File: strings.NewReader("x"), Purpose: "vision"})

			return err
		}},
		{name: "short expiration", run: func() error {
			_, err := provider.UploadFile(t.Context(), providers.UploadFileParams{
				File: strings.NewReader("x"), Purpose: providers.FilePurposeUserData, ExpiresAfter: new(3599),
			})

			return err
		}},
		{name: "long expiration", run: func() error {
			_, err := provider.UploadFile(t.Context(), providers.UploadFileParams{
				File: strings.NewReader("x"), Purpose: providers.FilePurposeUserData, ExpiresAfter: new(2592001),
			})

			return err
		}},
		{name: "zero list limit", run: func() error {
			_, err := provider.ListFiles(t.Context(), providers.ListFilesOptions{Limit: new(0)})

			return err
		}},
		{name: "large list limit", run: func() error {
			_, err := provider.ListFiles(t.Context(), providers.ListFilesOptions{Limit: new(1001)})

			return err
		}},
		{name: "wrong list order", run: func() error {
			_, err := provider.ListFiles(t.Context(), providers.ListFilesOptions{Order: "newest"})

			return err
		}},
		{name: "wrong list purpose", run: func() error {
			_, err := provider.ListFiles(t.Context(), providers.ListFilesOptions{Purpose: "vision"})

			return err
		}},
		{name: "missing retrieve id", run: func() error {
			_, err := provider.RetrieveFile(t.Context(), "")

			return err
		}},
		{name: "missing delete id", run: func() error {
			_, err := provider.DeleteFile(t.Context(), "")

			return err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.ErrorIs(t, tc.run(), llmerrors.ErrInvalidRequest)
		})
	}
}

func TestFileUploadHonorsCallerContext(t *testing.T) {
	t.Parallel()

	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL("http://127.0.0.1:1"))
	require.NoError(t, err)

	canceled, cancel := context.WithCancel(t.Context())
	cancel()

	_, err = provider.UploadFile(canceled, providers.UploadFileParams{
		File: strings.NewReader("x"), Purpose: providers.FilePurposeUserData,
	})
	require.ErrorIs(t, err, context.Canceled)

	timedOut, cancelTimeout := context.WithTimeout(t.Context(), time.Nanosecond)
	t.Cleanup(cancelTimeout)
	<-timedOut.Done()
	_, err = provider.UploadFile(timedOut, providers.UploadFileParams{
		File: strings.NewReader("x"), Purpose: providers.FilePurposeUserData,
	})
	require.ErrorIs(t, err, context.DeadlineExceeded)
}

func TestFileAPIErrorsUseNormalizedErrorMapping(t *testing.T) {
	t.Parallel()

	client := &http.Client{Transport: syncHTTPHandler(func(w *httptest.ResponseRecorder, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		w.Body.WriteString(`{"error":{"message":"slow down","type":"rate_limit_error"}}`)
	})}
	provider, err := New(
		config.WithAPIKey("test-key"),
		config.WithBaseURL("http://test"),
		config.WithHTTPClient(client),
	)
	require.NoError(t, err)

	_, err = provider.RetrieveFile(t.Context(), "file-api-one")
	require.ErrorIs(t, err, llmerrors.ErrRateLimit)
}

func TestFileOperationsAttributeMalformedResponses(t *testing.T) {
	t.Parallel()

	for _, operation := range []string{"upload", "retrieve", "list", "delete"} {
		for _, payload := range []string{
			`{"id":"file-api-one","object":"file"}`,
			`{"id":"file-api-one","object":"file","bytes":"invalid","deleted":"invalid"}`,
		} {
			t.Run(operation+"/"+payload, func(t *testing.T) {
				t.Parallel()

				client := &http.Client{Transport: syncHTTPHandler(func(w *httptest.ResponseRecorder, _ *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					if operation == "list" {
						w.Body.WriteString(`{"object":"list","data":[` + payload + `],"has_more":false}`)
					} else {
						w.Body.WriteString(payload)
					}
				})}
				provider, err := New(
					config.WithAPIKey("test-key"),
					config.WithBaseURL("http://test"),
					config.WithHTTPClient(client),
				)
				require.NoError(t, err)

				switch operation {
				case "upload":
					_, err = provider.UploadFile(t.Context(), providers.UploadFileParams{
						File: strings.NewReader("x"), Purpose: providers.FilePurposeUserData,
					})
				case "retrieve":
					_, err = provider.RetrieveFile(t.Context(), "file-api-one")
				case "list":
					_, err = provider.ListFiles(t.Context(), providers.ListFilesOptions{})
				case "delete":
					_, err = provider.DeleteFile(t.Context(), "file-api-one")
				}

				providerErr, ok := errors.AsType[*llmerrors.ProviderError](err)
				require.True(t, ok, "error lacks provider attribution: %v", err)
				require.Equal(t, "deepseek", providerErr.Provider)
				require.NotNil(t, errors.Unwrap(providerErr))
			})
		}
	}
}
