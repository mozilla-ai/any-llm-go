package llm

import (
	"net/http"
	"os"
	"time"
)

// Config holds the configuration for a provider.
type Config struct {
	// APIKey is the API key for authentication.
	APIKey string

	// BaseURL is the base URL for the API. If empty, the provider's default is used.
	BaseURL string

	// Timeout is the request timeout. If zero, a default timeout is used.
	Timeout time.Duration

	// HTTPClient is a custom HTTP client to use. If nil, a default client is created.
	HTTPClient *http.Client

	// Extra holds provider-specific configuration options.
	Extra map[string]any
}

// Option is a function that modifies the Config.
type Option func(*Config)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *Config) {
		c.APIKey = key
	}
}

// WithBaseURL sets the base URL.
func WithBaseURL(url string) Option {
	return func(c *Config) {
		c.BaseURL = url
	}
}

// WithTimeout sets the request timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Config) {
		c.Timeout = d
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *Config) {
		c.HTTPClient = client
	}
}

// WithExtra sets a provider-specific configuration option.
func WithExtra(key string, value any) Option {
	return func(c *Config) {
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[key] = value
	}
}

// DefaultConfig returns a Config with default values.
func DefaultConfig() *Config {
	return &Config{
		Timeout: 120 * time.Second,
	}
}

// ApplyOptions applies the given options to the config.
func (c *Config) ApplyOptions(opts ...Option) {
	for _, opt := range opts {
		opt(c)
	}
}

// GetAPIKeyFromEnv retrieves the API key from environment variable if not set.
func (c *Config) GetAPIKeyFromEnv(envVar string) string {
	if c.APIKey != "" {
		return c.APIKey
	}
	return os.Getenv(envVar)
}

// GetHTTPClient returns the configured HTTP client or creates a default one.
func (c *Config) GetHTTPClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{
		Timeout: c.Timeout,
	}
}

// GetExtra retrieves a provider-specific configuration value.
func (c *Config) GetExtra(key string) (any, bool) {
	if c.Extra == nil {
		return nil, false
	}
	v, ok := c.Extra[key]
	return v, ok
}

// GetExtraString retrieves a provider-specific string configuration value.
func (c *Config) GetExtraString(key string) string {
	v, ok := c.GetExtra(key)
	if !ok {
		return ""
	}
	s, _ := v.(string)
	return s
}

// GetExtraInt retrieves a provider-specific int configuration value.
func (c *Config) GetExtraInt(key string) int {
	v, ok := c.GetExtra(key)
	if !ok {
		return 0
	}
	i, _ := v.(int)
	return i
}

// GetExtraBool retrieves a provider-specific bool configuration value.
func (c *Config) GetExtraBool(key string) bool {
	v, ok := c.GetExtra(key)
	if !ok {
		return false
	}
	b, _ := v.(bool)
	return b
}
