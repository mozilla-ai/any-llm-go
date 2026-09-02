package config

import (
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// Config holds the configuration for a provider.
type Config struct {
	// APIKey is the API key for authentication.
	APIKey string

	// BaseURL is the base URL for the API. If empty, the provider's default is used.
	BaseURL string

	// Headers holds extra HTTP headers to include on every request.
	// Useful for proxy authentication (e.g., Cloudflare AI Gateway's
	// cf-aig-authorization header).
	Headers http.Header

	// Extra holds provider-specific configuration options.
	Extra map[string]any

	// Timeout is the request timeout. If zero, a default timeout is used.
	Timeout time.Duration

	// httpClient is a custom HTTP client. Access via HTTPClient() method which
	// handles lazy creation with the configured Timeout if not explicitly set on the client.
	httpClient     *http.Client
	httpClientOnce sync.Once
}

// Option is a function that modifies the Config.
type Option func(*Config) error

// headerTransport is an http.RoundTripper that injects a fixed set of headers
// onto every request it forwards.
type headerTransport struct {
	base    http.RoundTripper
	headers http.Header
}

// RoundTrip injects the configured headers and forwards the request.
// It operates on a clone so the caller's *http.Request is left unmodified,
// per the http.RoundTripper contract.
func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	for key, values := range t.headers {
		clone.Header.Del(key)
		for _, value := range values {
			clone.Header.Add(key, value)
		}
	}

	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}

	return base.RoundTrip(clone)
}

// New creates a Config with the given options applied.
// Note: HTTPClient is not created here by default; it is lazily created via the HTTPClient()
// method using the configured Timeout when first accessed.
func New(opts ...Option) (*Config, error) {
	// Configure default values.
	cfg := &Config{
		Timeout: 120 * time.Second,
	}

	// Apply options.
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(cfg); err != nil {
			return nil, err
		}
	}

	return cfg, nil
}

// WithAPIKey sets the API key. Whitespace is automatically trimmed.
func WithAPIKey(key string) Option {
	return func(c *Config) error {
		key = strings.TrimSpace(key)
		if key == "" {
			return fmt.Errorf("API key cannot be empty")
		}

		c.APIKey = key
		return nil
	}
}

// WithBaseURL sets the base URL.
func WithBaseURL(baseURL string) Option {
	return func(c *Config) error {
		baseURL = strings.TrimSpace(baseURL)
		if baseURL == "" {
			return fmt.Errorf("base URL cannot be empty")
		}

		parsed, err := url.Parse(baseURL)
		if err != nil {
			return fmt.Errorf("invalid base URL: %w", err)
		}

		if parsed.Scheme == "" || parsed.Host == "" {
			return fmt.Errorf("base URL must have scheme and host")
		}

		c.BaseURL = baseURL
		return nil
	}
}

// WithExtra sets extra provider-specific configuration.
// Whitespace is automatically trimmed from the key.
func WithExtra(key string, value any) Option {
	return func(c *Config) error {
		key = strings.TrimSpace(key)
		if key == "" {
			return fmt.Errorf("extra key cannot be empty")
		}

		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}

		c.Extra[key] = value
		return nil
	}
}

// WithHeader adds an HTTP header sent on every request. Whitespace is trimmed
// from the key, and repeated calls with the same key overwrite the previous value.
// Useful for proxy/gateway authentication (e.g., cf-aig-authorization for
// Cloudflare AI Gateway).
func WithHeader(key string, value string) Option {
	return func(c *Config) error {
		key = strings.TrimSpace(key)
		if key == "" {
			return fmt.Errorf("header key cannot be empty")
		}

		if c.Headers == nil {
			c.Headers = make(http.Header)
		}

		c.Headers.Set(key, value)
		return nil
	}
}

// WithHTTPClient sets a custom HTTP client.
// When a custom client is provided, the Timeout field is ignored for HTTP requests
// since the custom client manages its own timeout configuration.
// If both WithHTTPClient and WithTimeout are used, the custom client takes precedence.
func WithHTTPClient(client *http.Client) Option {
	return func(c *Config) error {
		if client == nil {
			return fmt.Errorf("HTTP client cannot be nil")
		}

		c.httpClient = client
		return nil
	}
}

// WithTimeout sets the request timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Config) error {
		if d <= 0 {
			return fmt.Errorf("timeout must be positive, got %v", d)
		}

		c.Timeout = d
		return nil
	}
}

// ExtraValue retrieves a provider-specific configuration value.
func (c *Config) ExtraValue(key string) (any, bool) {
	if c.Extra == nil {
		return nil, false
	}

	v, ok := c.Extra[key]
	return v, ok
}

// HTTPClient returns the HTTP client used for requests, creating and caching one on first use.
// When Headers are configured, the returned client wraps the base transport to inject those
// headers on every request; the base is the client from WithHTTPClient, or a lazily-created
// client using the configured Timeout.
//
// Note: with Headers set, the returned *http.Client is a distinct value from any client passed
// to WithHTTPClient — its Timeout, CheckRedirect, and Jar are copied, but the pointer differs.
func (c *Config) HTTPClient() *http.Client {
	c.httpClientOnce.Do(func() {
		if c.httpClient == nil {
			c.httpClient = &http.Client{Timeout: c.Timeout}
		}

		if len(c.Headers) > 0 {
			c.httpClient = &http.Client{
				Transport: &headerTransport{
					base:    c.httpClient.Transport,
					headers: c.Headers.Clone(),
				},
				Timeout:       c.httpClient.Timeout,
				CheckRedirect: c.httpClient.CheckRedirect,
				Jar:           c.httpClient.Jar,
			}
		}
	})

	return c.httpClient
}

// ResolveAPIKey returns the API key from config if set, otherwise falls back
// to the specified environment variable.
func (c *Config) ResolveAPIKey(envVar string) string {
	if c.APIKey != "" {
		return c.APIKey
	}

	return os.Getenv(envVar)
}

// ResolveEnv returns the value of the specified environment variable,
// trimming whitespace. Returns empty string if the variable is not set or empty.
func (c *Config) ResolveEnv(envVar string) string {
	if envVar == "" {
		return ""
	}
	return strings.TrimSpace(os.Getenv(envVar))
}

// ResolveBaseURL resolves the base URL from config, environment variable, or default value.
// It validates that the resolved URL has a scheme and host.
func (c *Config) ResolveBaseURL(envVar, defaultVal string) (string, error) {
	baseURL := c.BaseURL
	if baseURL == "" {
		baseURL = c.ResolveEnv(envVar)
	}
	if baseURL == "" {
		baseURL = defaultVal
	}

	if baseURL == "" {
		return "", nil
	}

	baseURL = strings.TrimSpace(baseURL)

	parsed, err := url.Parse(baseURL)
	if err != nil {
		return "", fmt.Errorf("invalid base URL %q: %w", baseURL, err)
	}

	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("base URL %q must have scheme and host", baseURL)
	}

	return baseURL, nil
}
