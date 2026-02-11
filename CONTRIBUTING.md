# Contributing to any-llm-go

Thank you for your interest in contributing to any-llm-go! This guide will help you get started.

## Development Setup

### Prerequisites

- **Go 1.25+** - [Download Go](https://go.dev/dl/)
- **Git** - For version control
- **API Keys** - For running integration tests (optional)
- **golangci-lint** - For linting (optional but recommended)

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mozilla-ai/any-llm-go.git
   cd any-llm-go
   ```

2. **Install dependencies:**
   ```bash
   go mod download
   ```

3. **Run tests:**
   ```bash
   make test-unit  # Unit tests only (no API keys needed)
   make test       # All tests (requires API keys for integration tests)
   ```

4. **Run linting:**
   ```bash
   make lint
   ```

### Setting Up API Keys for Integration Tests

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Project Structure

```
any-llm-go/
├── anyllm.go           # Root package - re-exports types for simple imports
├── config/config.go    # Functional options pattern for configuration
├── errors/errors.go    # Normalized error types with sentinel errors
├── providers/
│   ├── types.go        # Core interfaces and shared types
│   ├── anthropic/      # Native SDK provider (reference implementation)
│   ├── deepseek/       # OpenAI-compatible provider (with overrides)
│   ├── gemini/         # Native SDK provider
│   ├── groq/           # OpenAI-compatible provider (minimal wrapper)
│   ├── llamafile/      # OpenAI-compatible provider (local, no API key)
│   ├── mistral/        # OpenAI-compatible provider (with overrides)
│   ├── ollama/         # OpenAI-compatible provider (local, no API key)
│   └── openai/         # Native SDK provider + compatible base
│       ├── openai.go       # Native OpenAI provider
│       └── compatible.go   # Shared base for OpenAI-compatible APIs
├── internal/testutil/  # Test utilities and fixtures
├── docs/               # Documentation
└── examples/           # Example code
```

## Coding Standards

### Go Conventions

- Follow [Effective Go](https://go.dev/doc/effective_go) guidelines
- Use `gofmt` for formatting
- Run `golangci-lint` before committing

### Naming Conventions

- **Packages:** lowercase, single word (`openai`, `anthropic`)
- **Exported functions:** PascalCase (`New`, `Completion`)
- **Unexported functions:** camelCase (`convertParams`, `parseResponse`)
- **Constants:** PascalCase for exported, camelCase for unexported

### Error Handling

- Always check and handle errors
- Use sentinel errors for error categories (`ErrRateLimit`, etc.)
- Wrap errors with context using `fmt.Errorf("context: %w", err)`

### Testing

- Write unit tests for all new functionality
- Use table-driven tests where appropriate
- Use `testify/require` for assertions (not `assert`)
- Use `t.Parallel()` except when using `t.Setenv()`
- Use `t.Helper()` in test helpers
- Name test case variables `tc`, not `tt`
- Integration tests should skip gracefully when API keys are missing

## Adding a New Provider

There are two paths for adding a provider, depending on whether the provider exposes an OpenAI-compatible API.

### Path A: OpenAI-Compatible Provider (Recommended When Possible)

Many providers (Groq, DeepSeek, Mistral, Together AI, etc.) expose an OpenAI-compatible API. For these, you can build on the shared base in `providers/openai/compatible.go` and avoid reimplementing completions, streaming, tool calls, error conversion, and the rest.

**Use this path when:** the provider's API accepts OpenAI-format requests and returns OpenAI-format responses.

#### Minimal wrapper (Groq-style)

If the provider is fully OpenAI-compatible with no quirks, the entire implementation can be under 70 lines. See `providers/groq/groq.go` as the reference.

```go
package newprovider

import (
	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	defaultBaseURL = "https://api.newprovider.com/v1"
	envAPIKey      = "NEWPROVIDER_API_KEY"
	providerName   = "newprovider"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface for NewProvider.
type Provider struct {
	*openai.CompatibleProvider
}

// New creates a new NewProvider provider.
func New(opts ...config.Option) (*Provider, error) {
	base, err := openai.NewCompatible(openai.CompatibleConfig{
		APIKeyEnvVar:   envAPIKey,
		BaseURLEnvVar:  "",
		Capabilities:   capabilities(),
		DefaultAPIKey:  "",
		DefaultBaseURL: defaultBaseURL,
		Name:           providerName,
		RequireAPIKey:  true,
	}, opts...)
	if err != nil {
		return nil, err
	}

	return &Provider{CompatibleProvider: base}, nil
}

// newproviderCapabilities returns the capabilities for the provider.
func newproviderCapabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     false,
		CompletionPDF:       false,
		CompletionReasoning: false,
		CompletionStreaming: true,
		Embedding:           false,
		ListModels:          true,
	}
}
```

Key points:

- Set **all** `CompatibleConfig` fields explicitly, including empty values (`BaseURLEnvVar: ""`, `DefaultAPIKey: ""`)
- The embedded `CompatibleProvider` provides `Completion`, `CompletionStream`, `Embedding`, `ListModels`, `ConvertError`, `Capabilities`, and `Name` for free
- Only declare interface assertions for interfaces the provider actually supports (e.g., omit `EmbeddingProvider` if `Embedding` is false)

#### Wrapper with overrides (DeepSeek/Mistral-style)

Some providers are mostly compatible but have quirks: unsupported parameters, different JSON schema handling, required message patching, etc. In these cases, embed the base and override specific methods.

See `providers/deepseek/deepseek.go` or `providers/mistral/mistral.go` as references.

```go
// Completion overrides the base to handle provider-specific quirks.
func (p *Provider) Completion(ctx context.Context, params providers.CompletionParams) (*providers.ChatCompletion, error) {
	params = preprocessParams(params)
	return p.CompatibleProvider.Completion(ctx, params)
}

// CompletionStream overrides the base to handle provider-specific quirks.
func (p *Provider) CompletionStream(ctx context.Context, params providers.CompletionParams) (<-chan providers.ChatCompletionChunk, <-chan error) {
	params = preprocessParams(params)
	return p.CompatibleProvider.CompletionStream(ctx, params)
}

// preprocessParams adjusts parameters for provider-specific behavior.
func preprocessParams(params providers.CompletionParams) providers.CompletionParams {
	// Strip unsupported fields, patch messages, etc.
	return params
}
```

#### For local servers (Llamafile/Ollama-style)

For providers that run locally and don't require an API key:

```go
base, err := openai.NewCompatible(openai.CompatibleConfig{
	APIKeyEnvVar:   "",
	BaseURLEnvVar:  "NEWPROVIDER_BASE_URL",
	Capabilities:   capabilities(),
	DefaultAPIKey:  "no-key-required",
	DefaultBaseURL: "http://localhost:8080/v1",
	Name:           providerName,
	RequireAPIKey:  false,  // No API key needed for local servers.
}, opts...)
```

### Path B: Native SDK Provider

When a provider has an official Go SDK and their API is not OpenAI-compatible (e.g., Anthropic, Google Gemini), implement the provider using that SDK directly.

**Use this path when:** the provider has a dedicated Go SDK and their request/response format differs from OpenAI's.

See `providers/anthropic/anthropic.go` as the canonical reference.

```go
package newprovider

import (
	"context"
	stderrors "errors"
	"fmt"

	"github.com/newprovider/sdk-go"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// Provider configuration constants.
const (
	envAPIKey    = "NEWPROVIDER_API_KEY"
	providerName = "newprovider"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// Provider implements the providers.Provider interface using the native SDK.
type Provider struct {
	client *sdk.Client
	config *config.Config
	name   string
}

// New creates a new provider instance.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	apiKey := cfg.ResolveAPIKey(envAPIKey)
	if apiKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	client := sdk.NewClient(apiKey)

	return &Provider{
		client: client,
		config: cfg,
		name:   providerName,
	}, nil
}

func (p *Provider) Name() string {
	return p.name
}

func (p *Provider) Capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionStreaming: true,
		// ... set all fields explicitly.
	}
}

func (p *Provider) Completion(ctx context.Context, params providers.CompletionParams) (*providers.ChatCompletion, error) {
	// Convert unified params to SDK-specific request.
	req := convertParams(params)

	resp, err := p.client.Chat(ctx, req)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	// Convert SDK-specific response to unified format.
	return convertResponse(resp), nil
}

func (p *Provider) CompletionStream(ctx context.Context, params providers.CompletionParams) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errc := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errc)

		// Always use select with ctx.Done() when sending to channels.
		select {
		case chunks <- chunk:
		case <-ctx.Done():
			return
		}
	}()

	return chunks, errc
}

// ConvertError converts SDK errors to unified error types.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *sdk.Error
	if !stderrors.As(err, &apiErr) {
		return errors.NewProviderError(providerName, err)
	}

	switch apiErr.StatusCode {
	case 401:
		return errors.NewAuthenticationError(providerName, err)
	case 404:
		return errors.NewModelNotFoundError(providerName, err)
	case 429:
		return errors.NewRateLimitError(providerName, err)
	default:
		return errors.NewProviderError(providerName, err)
	}
}
```

Key requirements for native SDK providers:

- Normalize all responses to OpenAI format (`ChatCompletion`, `ChatCompletionChunk`)
- Use `errors.As` with SDK typed errors for error conversion (avoid string matching)
- Always use `select` with `ctx.Done()` in streaming goroutines
- Convert SDK-specific message formats, tool calls, and content types to the unified types

### Write Tests

Both paths follow the same testing patterns:

```go
package newprovider

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mozilla-ai/any-llm-go/config"
)

func TestNew(t *testing.T) {
	t.Parallel()

	t.Run("creates provider with API key", func(t *testing.T) {
		t.Parallel()

		provider, err := New(config.WithAPIKey("test-key"))
		require.NoError(t, err)
		require.NotNil(t, provider)
	})

	t.Run("returns error when API key is missing", func(t *testing.T) {
		t.Setenv("NEWPROVIDER_API_KEY", "")

		provider, err := New()
		require.Nil(t, provider)
		require.Error(t, err)
	})
}
```

### Update Documentation

- Add provider to `docs/providers.md` (feature matrix and details section)
- Update `README.md` supported providers table

### Requirements Checklist

- [ ] Uses official provider SDK (Path B) or OpenAI-compatible base (Path A)
- [ ] Implements `Provider` interface
- [ ] Implements `CapabilityProvider` interface
- [ ] Implements `ErrorConverter` interface
- [ ] Normalizes responses to OpenAI format
- [ ] Has unit tests with `t.Parallel()`
- [ ] Has integration tests (skipped when no API key)
- [ ] Passes `golangci-lint`
- [ ] Documentation updated

## File Organization

Within each provider file, follow this ordering:

1. Package declaration and imports
2. Constants (grouped by purpose, unexported)
3. Interface assertions (`var _ Interface = (*Type)(nil)`)
4. Types (exported first, then unexported helpers)
5. Constructor (`New()`)
6. Exported methods (alphabetically)
7. Unexported methods (alphabetically)
8. Package-level functions (alphabetically)

## Branch Naming

Use descriptive branch names:

- `feature/add-mistral-provider`
- `fix/streaming-error-handling`
- `docs/update-quickstart`
- `refactor/simplify-error-types`

## Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Examples:
```
feat(provider): add Mistral provider support

fix(anthropic): handle streaming errors correctly

docs: update quickstart guide with streaming example
```

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the coding standards
3. **Write/update tests** for your changes
4. **Run tests and linting:**
   ```bash
   make lint
   make test-unit
   ```
5. **Update documentation** if needed
6. **Submit a PR** with a clear description

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2

## Testing
How were these changes tested?

## Checklist
- [ ] Tests pass locally
- [ ] Linting passes
- [ ] Documentation updated (if needed)
```

## Getting Help

- **Issues:** Open a GitHub issue for bugs or feature requests
- **Discussions:** Use GitHub Discussions for questions

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
