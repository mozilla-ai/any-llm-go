package deepseek

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

type namedFileReader struct {
	*bytes.Reader
}

func (namedFileReader) Filename() string {
	return "chart.png"
}

func TestFileLifecycleUsesDeepSeekContract(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		switch r.Method + " " + r.URL.Path {
		case "POST /files":
			require.NoError(t, r.ParseMultipartForm(1<<20))
			require.Equal(t, "user_data", r.FormValue("purpose"))
			require.Equal(t, "created_at", r.FormValue("expires_after[anchor]"))
			require.Equal(t, "3600", r.FormValue("expires_after[seconds]"))
			file, header, err := r.FormFile("file")
			require.NoError(t, err)
			defer func() { require.NoError(t, file.Close()) }()
			require.Equal(t, "chart.png", header.Filename)
			content, err := io.ReadAll(file)
			require.NoError(t, err)
			require.Equal(t, []byte("image bytes"), content)
			_, err = fmt.Fprint(w, `{
				"id":"file-api-one","object":"file","bytes":11,"created_at":1700000000,
				"filename":"chart.png","purpose":"user_data","expires_at":1700003600,
				"future_field":"ignored"
			}`)
			require.NoError(t, err)
		case "GET /files":
			require.Equal(t, "file-api-cursor", r.URL.Query().Get("after"))
			require.Equal(t, "2", r.URL.Query().Get("limit"))
			require.Equal(t, "desc", r.URL.Query().Get("order"))
			require.Equal(t, "user_data", r.URL.Query().Get("purpose"))
			_, err := fmt.Fprint(w, `{
				"object":"list","data":[{"id":"file-api-one","object":"file","bytes":11,
				"created_at":1700000000,"filename":"chart.png","purpose":"user_data"}],
				"first_id":"file-api-one","last_id":"file-api-one","has_more":false
			}`)
			require.NoError(t, err)
		case "GET /files/file-api-one":
			_, err := fmt.Fprint(w, `{
				"id":"file-api-one","object":"file","bytes":11,"created_at":1700000000,
				"filename":"chart.png","purpose":"user_data"
			}`)
			require.NoError(t, err)
		case "DELETE /files/file-api-one":
			_, err := fmt.Fprint(w, `{"id":"file-api-one","object":"file","deleted":true}`)
			require.NoError(t, err)
		default:
			http.Error(w, "unexpected request", http.StatusNotFound)
		}
	}))
	t.Cleanup(server.Close)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	uploaded, err := provider.UploadFile(t.Context(), providers.UploadFileParams{
		File:         namedFileReader{Reader: bytes.NewReader([]byte("image bytes"))},
		Purpose:      providers.FilePurposeUserData,
		ExpiresAfter: new(3600),
	})
	require.NoError(t, err)
	require.Equal(t, &providers.File{
		ID: "file-api-one", Object: "file", Bytes: 11, CreatedAt: 1700000000,
		Filename: "chart.png", Purpose: providers.FilePurposeUserData, ExpiresAt: new(int64(1700003600)),
	}, uploaded)

	files, err := provider.ListFiles(t.Context(), providers.ListFilesOptions{
		After: "file-api-cursor", Limit: new(2), Order: providers.FileOrderDesc,
		Purpose: providers.FilePurposeUserData,
	})
	require.NoError(t, err)
	require.Equal(t, "list", files.Object)
	require.Equal(t, "file-api-one", files.FirstID)
	require.Equal(t, "file-api-one", files.LastID)
	require.False(t, files.HasMore)
	require.Len(t, files.Data, 1)

	retrieved, err := provider.RetrieveFile(t.Context(), "file-api-one")
	require.NoError(t, err)
	require.Equal(t, "chart.png", retrieved.Filename)
	require.Nil(t, retrieved.ExpiresAt)

	deleted, err := provider.DeleteFile(t.Context(), "file-api-one")
	require.NoError(t, err)
	require.Equal(t, &providers.DeletedFile{ID: "file-api-one", Object: "file", Deleted: true}, deleted)
}

func TestFileUploadOmitsExpirationFieldsForPermanentFiles(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, r.ParseMultipartForm(1<<20))
		require.Empty(t, r.FormValue("expires_after[anchor]"))
		require.Empty(t, r.FormValue("expires_after[seconds]"))
		w.Header().Set("Content-Type", "application/json")
		_, err := fmt.Fprint(w, `{
			"id":"file-api-one","object":"file","bytes":1,"created_at":1700000000,
			"filename":"image.png","purpose":"user_data"
		}`)
		require.NoError(t, err)
	}))
	t.Cleanup(server.Close)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	file, err := provider.UploadFile(t.Context(), providers.UploadFileParams{
		File: strings.NewReader("x"), Purpose: providers.FilePurposeUserData,
	})
	require.NoError(t, err)
	require.Nil(t, file.ExpiresAt)
}

func TestListFilesAcceptsDocumentedLimitBoundaries(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Contains(t, []string{"1", "1000"}, r.URL.Query().Get("limit"))
		w.Header().Set("Content-Type", "application/json")
		_, err := fmt.Fprint(w, `{"object":"list","data":[],"has_more":false}`)
		require.NoError(t, err)
	}))
	t.Cleanup(server.Close)
	provider, err := New(config.WithAPIKey("test-key"), config.WithBaseURL(server.URL))
	require.NoError(t, err)

	for _, limit := range []int{1, 1000} {
		files, err := provider.ListFiles(t.Context(), providers.ListFilesOptions{Limit: new(limit)})
		require.NoError(t, err)
		require.Empty(t, files.Data)
	}
}
