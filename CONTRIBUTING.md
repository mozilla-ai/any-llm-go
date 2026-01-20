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
├── llm.go           # Public API convenience functions
├── types.go            # Core type definitions
├── provider.go         # Provider interface and helpers
├── options.go          # Configuration options
├── errors.go           # Error types
├── registry.go         # Provider registration
├── providers/
│   ├── openai/         # OpenAI provider implementation
│   │   ├── openai.go
│   │   └── openai_test.go
│   └── anthropic/      # Anthropic provider implementation
│       ├── anthropic.go
│       └── anthropic_test.go
├── internal/
│   └── testutil/       # Test utilities and fixtures
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
- **Exported functions:** PascalCase (`NewProvider`, `Completion`)
- **Unexported functions:** camelCase (`convertParams`, `parseResponse`)
- **Constants:** PascalCase for exported, camelCase for unexported

### Error Handling

- Always check and handle errors
- Use sentinel errors for error categories (`ErrRateLimit`, etc.)
- Wrap errors with context using `fmt.Errorf("context: %w", err)`

### Testing

- Write unit tests for all new functionality
- Use table-driven tests where appropriate
- Use `testify` for assertions
- Integration tests should skip gracefully when API keys are missing

## Adding a New Provider

### 1. Create the Provider Package

Create a new directory under `providers/`:

```
providers/
└── newprovider/
    ├── newprovider.go
    └── newprovider_test.go
```

### 2. Implement the Provider Interface

```go
package newprovider

import (
    "context"
    "fmt"

    "github.com/mozilla-ai/any-llm-go/config"
    "github.com/mozilla-ai/any-llm-go/errors"
    "github.com/mozilla-ai/any-llm-go/providers"
)

const (
    providerName = "newprovider"
    envAPIKey    = "NEWPROVIDER_API_KEY"
)

type Provider struct {
    client *sdk.Client
    config *config.Config
}

// Ensure interface compliance.
var (
    _ providers.CapabilityProvider = (*Provider)(nil)
    _ providers.ErrorConverter     = (*Provider)(nil)
    _ providers.Provider           = (*Provider)(nil)
)

func New(opts ...config.Option) (*Provider, error) {
    cfg, err := config.New(opts...)
    if err != nil {
        return nil, fmt.Errorf("invalid options: %w", err)
    }

    apiKey := cfg.ResolveAPIKey(envAPIKey)
    if apiKey == "" {
        return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
    }

    // Initialize the official SDK client.
    client := sdk.NewClient(apiKey)

    return &Provider{
        client: client,
        config: cfg,
    }, nil
}

func (p *Provider) Name() string {
    return providerName
}

func (p *Provider) Capabilities() providers.Capabilities {
    return providers.Capabilities{
        Completion:          true,
        CompletionStreaming: true,
        // ... other capabilities
    }
}

func (p *Provider) Completion(ctx context.Context, params providers.CompletionParams) (*providers.ChatCompletion, error) {
    req := convertParams(params)

    resp, err := p.client.Messages.New(ctx, req)
    if err != nil {
        return nil, p.ConvertError(err)
    }

    return convertResponse(resp), nil
}

func (p *Provider) CompletionStream(ctx context.Context, params providers.CompletionParams) (<-chan providers.ChatCompletionChunk, <-chan error) {
    // Implement streaming, use p.ConvertError() for errors.
}

// ConvertError converts SDK errors to unified error types.
// Implements providers.ErrorConverter.
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
    case 429:
        return errors.NewRateLimitError(providerName, err)
    case 404:
        return errors.NewModelNotFoundError(providerName, err)
    default:
        return errors.NewProviderError(providerName, err)
    }
}
```

### 3. Write Tests

```go
package newprovider

import (
    "testing"

    "github.com/stretchr/testify/require"

    "github.com/mozilla-ai/any-llm-go/config"
    "github.com/mozilla-ai/any-llm-go/internal/testutil"
)

func TestNew(t *testing.T) {
    t.Run("creates provider with API key", func(t *testing.T) {
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

// Integration tests.
func TestIntegrationCompletion(t *testing.T) {
    if testutil.SkipIfNoAPIKey("newprovider") {
        t.Skip("NEWPROVIDER_API_KEY not set")
    }

    provider, err := New()
    require.NoError(t, err)

    // Test completion...
}
```

### 4. Update Documentation

- Add provider to `docs/providers.md`
- Update `README.md` if needed
- Add any provider-specific notes

### 5. Requirements Checklist

- [ ] Uses official provider SDK (when available)
- [ ] Implements `Provider` interface
- [ ] Implements `CapabilityProvider` interface
- [ ] Normalizes responses to OpenAI format
- [ ] Implements `ErrorConverter` interface with `ConvertError()` method
- [ ] Has unit tests with >80% coverage
- [ ] Has integration tests (skipped when no API key)
- [ ] Passes `golangci-lint`
- [ ] Documentation updated

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
feat(providers): add Mistral provider support

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
