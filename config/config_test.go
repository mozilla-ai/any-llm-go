package config

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestWithAPIKey(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		key     string
		wantErr bool
		wantKey string
	}{
		{
			name:    "valid key",
			key:     "sk-123456",
			wantErr: false,
			wantKey: "sk-123456",
		},
		{
			name:    "valid key with whitespace trimmed",
			key:     "  sk-123456  ",
			wantErr: false,
			wantKey: "sk-123456",
		},
		{
			name:    "empty key",
			key:     "",
			wantErr: true,
		},
		{
			name:    "whitespace only key",
			key:     "   ",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := New(WithAPIKey(tc.key))
			if tc.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.Equal(t, tc.wantKey, cfg.APIKey)
		})
	}
}

func TestWithBaseURL(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		url     string
		wantErr bool
		wantURL string
	}{
		{
			name:    "valid https URL",
			url:     "https://api.example.com",
			wantErr: false,
			wantURL: "https://api.example.com",
		},
		{
			name:    "valid http URL",
			url:     "http://localhost:8080",
			wantErr: false,
			wantURL: "http://localhost:8080",
		},
		{
			name:    "valid URL with path",
			url:     "https://api.example.com/v1",
			wantErr: false,
			wantURL: "https://api.example.com/v1",
		},
		{
			name:    "valid URL with whitespace trimmed",
			url:     "  https://api.example.com  ",
			wantErr: false,
			wantURL: "https://api.example.com",
		},
		{
			name:    "empty URL",
			url:     "",
			wantErr: true,
		},
		{
			name:    "whitespace only URL",
			url:     "   ",
			wantErr: true,
		},
		{
			name:    "URL without scheme",
			url:     "api.example.com",
			wantErr: true,
		},
		{
			name:    "URL without host",
			url:     "https://",
			wantErr: true,
		},
		{
			name:    "relative path only",
			url:     "/v1/chat",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := New(WithBaseURL(tc.url))
			if tc.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.Equal(t, tc.wantURL, cfg.BaseURL)
		})
	}
}

func TestWithTimeout(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		timeout     time.Duration
		wantErr     bool
		wantTimeout time.Duration
	}{
		{
			name:        "valid timeout",
			timeout:     30 * time.Second,
			wantErr:     false,
			wantTimeout: 30 * time.Second,
		},
		{
			name:        "one nanosecond",
			timeout:     time.Nanosecond,
			wantErr:     false,
			wantTimeout: time.Nanosecond,
		},
		{
			name:    "zero timeout",
			timeout: 0,
			wantErr: true,
		},
		{
			name:    "negative timeout",
			timeout: -1 * time.Second,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := New(WithTimeout(tc.timeout))
			if tc.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.Equal(t, tc.wantTimeout, cfg.Timeout)
		})
	}
}

func TestWithHTTPClient(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		client  *http.Client
		wantErr bool
	}{
		{
			name:    "valid client",
			client:  &http.Client{Timeout: 10 * time.Second},
			wantErr: false,
		},
		{
			name:    "nil client",
			client:  nil,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := New(WithHTTPClient(tc.client))
			if tc.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.Same(t, tc.client, cfg.HTTPClient())
		})
	}
}

func TestHTTPClientLazyCreation(t *testing.T) {
	t.Parallel()

	t.Run("uses configured timeout", func(t *testing.T) {
		t.Parallel()

		cfg, err := New(WithTimeout(45 * time.Second))
		require.NoError(t, err)

		client := cfg.HTTPClient()
		require.NotNil(t, client)
		require.Equal(t, 45*time.Second, client.Timeout)
	})

	t.Run("uses default timeout when not configured", func(t *testing.T) {
		t.Parallel()

		cfg, err := New()
		require.NoError(t, err)

		client := cfg.HTTPClient()
		require.NotNil(t, client)
		require.Equal(t, 120*time.Second, client.Timeout)
	})

	t.Run("custom client takes precedence", func(t *testing.T) {
		t.Parallel()

		customClient := &http.Client{Timeout: 5 * time.Second}
		cfg, err := New(
			WithTimeout(60*time.Second),
			WithHTTPClient(customClient),
		)
		require.NoError(t, err)

		require.Same(t, customClient, cfg.HTTPClient())
	})
}

func TestWithExtra(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		key     string
		value   any
		wantErr bool
	}{
		{
			name:    "valid string value",
			key:     "client_name",
			value:   "my-client",
			wantErr: false,
		},
		{
			name:    "valid int value",
			key:     "max_retries",
			value:   3,
			wantErr: false,
		},
		{
			name:    "key with whitespace trimmed",
			key:     "  client_name  ",
			value:   "my-client",
			wantErr: false,
		},
		{
			name:    "empty key",
			key:     "",
			value:   "value",
			wantErr: true,
		},
		{
			name:    "whitespace only key",
			key:     "   ",
			value:   "value",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := New(WithExtra(tc.key, tc.value))
			if tc.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, cfg.Extra)
		})
	}
}

func TestExtraValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		cfg       *Config
		key       string
		wantValue any
		wantOK    bool
	}{
		{
			name: "returns value when present",
			cfg: &Config{
				Extra: map[string]any{"key": "value"},
			},
			key:       "key",
			wantValue: "value",
			wantOK:    true,
		},
		{
			name: "returns false when key missing",
			cfg: &Config{
				Extra: map[string]any{"other": "value"},
			},
			key:       "nonexistent",
			wantValue: nil,
			wantOK:    false,
		},
		{
			name:      "returns false when Extra is nil",
			cfg:       &Config{},
			key:       "key",
			wantValue: nil,
			wantOK:    false,
		},
		{
			name: "returns int value",
			cfg: &Config{
				Extra: map[string]any{"count": 42},
			},
			key:       "count",
			wantValue: 42,
			wantOK:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			v, ok := tc.cfg.ExtraValue(tc.key)
			require.Equal(t, tc.wantOK, ok)
			require.Equal(t, tc.wantValue, v)
		})
	}
}

func TestNew(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		opts        []Option
		wantErr     bool
		wantTimeout time.Duration
		wantAPIKey  string
		wantBaseURL string
	}{
		{
			name:        "no options uses defaults",
			opts:        nil,
			wantErr:     false,
			wantTimeout: 120 * time.Second,
		},
		{
			name:        "nil option is skipped",
			opts:        []Option{nil, WithAPIKey("test-key"), nil},
			wantErr:     false,
			wantTimeout: 120 * time.Second,
			wantAPIKey:  "test-key",
		},
		{
			name: "multiple options applied",
			opts: []Option{
				WithAPIKey("my-key"),
				WithBaseURL("https://api.example.com"),
				WithTimeout(30 * time.Second),
			},
			wantErr:     false,
			wantTimeout: 30 * time.Second,
			wantAPIKey:  "my-key",
			wantBaseURL: "https://api.example.com",
		},
		{
			name:    "error from option propagates",
			opts:    []Option{WithAPIKey("")},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := New(tc.opts...)
			if tc.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, cfg)
			require.Equal(t, tc.wantTimeout, cfg.Timeout)
			require.Equal(t, tc.wantAPIKey, cfg.APIKey)
			require.Equal(t, tc.wantBaseURL, cfg.BaseURL)
		})
	}
}

func TestResolveAPIKey(t *testing.T) {
	// Note: Cannot use t.Parallel() with t.Setenv().

	tests := []struct {
		name       string
		configKey  string
		envVar     string
		envValue   string
		wantAPIKey string
	}{
		{
			name:       "returns config key when set",
			configKey:  "config-key",
			envVar:     "TEST_API_KEY",
			envValue:   "env-key",
			wantAPIKey: "config-key",
		},
		{
			name:       "falls back to env when config key empty",
			configKey:  "",
			envVar:     "TEST_API_KEY_FALLBACK",
			envValue:   "env-key",
			wantAPIKey: "env-key",
		},
		{
			name:       "returns empty when both empty",
			configKey:  "",
			envVar:     "TEST_API_KEY_EMPTY",
			envValue:   "",
			wantAPIKey: "",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.envValue != "" {
				t.Setenv(tc.envVar, tc.envValue)
			}

			cfg := &Config{APIKey: tc.configKey}
			result := cfg.ResolveAPIKey(tc.envVar)
			require.Equal(t, tc.wantAPIKey, result)
		})
	}
}

func TestResolveEnv(t *testing.T) {
	// Note: Cannot use t.Parallel() with t.Setenv().

	t.Run("returns trimmed env value", func(t *testing.T) {
		t.Setenv("TEST_RESOLVE_ENV", "  some-value  ")

		cfg := &Config{}
		result := cfg.ResolveEnv("TEST_RESOLVE_ENV")
		require.Equal(t, "some-value", result)
	})

	t.Run("returns empty for unset variable", func(t *testing.T) {
		cfg := &Config{}
		result := cfg.ResolveEnv("TEST_RESOLVE_ENV_UNSET")
		require.Empty(t, result)
	})

	t.Run("returns empty for empty env var name", func(t *testing.T) {
		cfg := &Config{}
		result := cfg.ResolveEnv("")
		require.Empty(t, result)
	})
}

func TestResolveBaseURL(t *testing.T) {
	// Note: Cannot use t.Parallel() with t.Setenv().

	t.Run("uses config BaseURL first", func(t *testing.T) {
		cfg := &Config{BaseURL: "https://config.example.com/v1"}
		result, err := cfg.ResolveBaseURL("", "https://default.example.com/v1")
		require.NoError(t, err)
		require.Equal(t, "https://config.example.com/v1", result)
	})

	t.Run("falls back to env var", func(t *testing.T) {
		t.Setenv("TEST_BASE_URL_RESOLVE", "https://env.example.com/v1")

		cfg := &Config{}
		result, err := cfg.ResolveBaseURL("TEST_BASE_URL_RESOLVE", "https://default.example.com/v1")
		require.NoError(t, err)
		require.Equal(t, "https://env.example.com/v1", result)
	})

	t.Run("falls back to default", func(t *testing.T) {
		cfg := &Config{}
		result, err := cfg.ResolveBaseURL("", "https://default.example.com/v1")
		require.NoError(t, err)
		require.Equal(t, "https://default.example.com/v1", result)
	})

	t.Run("returns empty when all empty", func(t *testing.T) {
		cfg := &Config{}
		result, err := cfg.ResolveBaseURL("", "")
		require.NoError(t, err)
		require.Empty(t, result)
	})

	t.Run("returns error for invalid URL", func(t *testing.T) {
		cfg := &Config{BaseURL: "://bad-url"}
		_, err := cfg.ResolveBaseURL("", "")
		require.Error(t, err)
		require.Contains(t, err.Error(), "invalid base URL")
	})

	t.Run("returns error for URL without scheme", func(t *testing.T) {
		cfg := &Config{BaseURL: "example.com/v1"}
		_, err := cfg.ResolveBaseURL("", "")
		require.Error(t, err)
		require.Contains(t, err.Error(), "must have scheme and host")
	})

	t.Run("trims whitespace from resolved URL", func(t *testing.T) {
		t.Setenv("TEST_BASE_URL_WS", "  https://env.example.com/v1  ")

		cfg := &Config{}
		result, err := cfg.ResolveBaseURL("TEST_BASE_URL_WS", "")
		require.NoError(t, err)
		require.Equal(t, "https://env.example.com/v1", result)
	})
}

func TestHTTPClientCaching(t *testing.T) {
	t.Parallel()

	cfg, err := New()
	require.NoError(t, err)

	// Get client twice - should return same instance.
	client1 := cfg.HTTPClient()
	client2 := cfg.HTTPClient()

	require.Same(t, client1, client2)
}

func TestWithHeader(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		key     string
		value   string
		wantKey string
		wantErr bool
	}{
		{
			name:    "valid header",
			key:     "cf-aig-authorization",
			value:   "Bearer token123",
			wantKey: "cf-aig-authorization",
			wantErr: false,
		},
		{
			name:    "key with whitespace trimmed",
			key:     "  X-Custom-Header  ",
			value:   "value",
			wantKey: "X-Custom-Header",
			wantErr: false,
		},
		{
			name:    "empty key",
			key:     "",
			value:   "value",
			wantErr: true,
		},
		{
			name:    "whitespace only key",
			key:     "   ",
			value:   "value",
			wantErr: true,
		},
		{
			name:    "empty value is allowed",
			key:     "X-Header",
			value:   "",
			wantKey: "X-Header",
			wantErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg, err := New(WithHeader(tc.key, tc.value))
			if tc.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, cfg.Headers)
			require.Equal(t, tc.value, cfg.Headers.Get(tc.wantKey))
		})
	}
}

func TestWithHeader_Multiple(t *testing.T) {
	t.Parallel()

	cfg, err := New(
		WithHeader("cf-aig-authorization", "Bearer token1"),
		WithHeader("X-Custom", "value2"),
	)
	require.NoError(t, err)
	require.Len(t, cfg.Headers, 2)
	require.Equal(t, "Bearer token1", cfg.Headers.Get("cf-aig-authorization"))
	require.Equal(t, "value2", cfg.Headers.Get("X-Custom"))
}

func TestWithHeader_DuplicateKeyOverwrites(t *testing.T) {
	t.Parallel()

	cfg, err := New(
		WithHeader("X-Token", "first"),
		WithHeader("X-Token", "second"),
	)
	require.NoError(t, err)

	// WithHeader uses Set semantics: the later value replaces the earlier one
	// rather than appending a second value.
	require.Len(t, cfg.Headers, 1)
	require.Equal(t, "second", cfg.Headers.Get("X-Token"))
	require.Equal(t, []string{"second"}, cfg.Headers.Values("X-Token"))
}

func TestHTTPClient_WithHeaders(t *testing.T) {
	t.Parallel()

	cfg, err := New(
		WithHeader("cf-aig-authorization", "Bearer mytoken"),
		WithHeader("X-Custom", "value"),
	)
	require.NoError(t, err)

	client := cfg.HTTPClient()
	require.NotNil(t, client)

	// The transport should be a headerTransport.
	ht, ok := client.Transport.(*headerTransport)
	require.True(t, ok, "expected headerTransport, got %T", client.Transport)
	require.Equal(t, "Bearer mytoken", ht.headers.Get("cf-aig-authorization"))
	require.Equal(t, "value", ht.headers.Get("X-Custom"))
}

func TestHTTPClient_WithHeaders_CustomClient(t *testing.T) {
	t.Parallel()

	customClient := &http.Client{Timeout: 5 * time.Second}
	cfg, err := New(
		WithHTTPClient(customClient),
		WithHeader("cf-aig-authorization", "Bearer token"),
	)
	require.NoError(t, err)

	client := cfg.HTTPClient()
	require.NotNil(t, client)

	// Should wrap the custom client's transport.
	ht, ok := client.Transport.(*headerTransport)
	require.True(t, ok, "expected headerTransport wrapping custom client")
	require.Equal(t, "Bearer token", ht.headers.Get("cf-aig-authorization"))
	require.Equal(t, 5*time.Second, client.Timeout)
}

func TestHTTPClient_NoHeaders_NoWrapping(t *testing.T) {
	t.Parallel()

	cfg, err := New()
	require.NoError(t, err)

	client := cfg.HTTPClient()
	require.NotNil(t, client)

	// Without headers, the client is left unwrapped and carries no custom transport.
	require.Nil(t, client.Transport)
}

func TestHTTPClient_Headers_EndToEnd(t *testing.T) {
	t.Parallel()

	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	cfg, err := New(
		WithHeader("cf-aig-authorization", "Bearer e2e-token"),
		WithHeader("X-Custom", "custom-value"),
	)
	require.NoError(t, err)

	req, err := http.NewRequest(http.MethodGet, srv.URL, nil)
	require.NoError(t, err)

	resp, err := cfg.HTTPClient().Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.Equal(t, "Bearer e2e-token", got.Get("cf-aig-authorization"))
	require.Equal(t, "custom-value", got.Get("X-Custom"))

	// RoundTrip must not mutate the caller's request (net/http contract).
	require.Empty(t, req.Header.Get("cf-aig-authorization"))
	require.Empty(t, req.Header.Get("X-Custom"))
}
