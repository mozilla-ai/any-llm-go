package config

import (
	"net/http"
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
			name:        "multiple options applied",
			opts:        []Option{WithAPIKey("my-key"), WithBaseURL("https://api.example.com"), WithTimeout(30 * time.Second)},
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

func TestHTTPClientCaching(t *testing.T) {
	t.Parallel()

	cfg, err := New()
	require.NoError(t, err)

	// Get client twice - should return same instance.
	client1 := cfg.HTTPClient()
	client2 := cfg.HTTPClient()

	require.Same(t, client1, client2)
}
