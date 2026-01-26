# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

any-llm-go is the official Go port of [any-llm](https://github.com/mozilla-ai/any-llm). It provides a unified interface for multiple LLM providers, normalizing responses to OpenAI's format.

## Go Guidelines

Follow [Go Proverbs](https://go-proverbs.github.io/) and [Effective Go](https://go.dev/doc/effective_go).

Style preferences:
- Flat control flow: early returns, avoid deep nesting
- Small, focused functions with single responsibility
- Prefer functional/declarative over imperative/mutating
- Never mutate receiver state or parameters

## Commands

```bash
make lint       # Run linter with auto-fix
make test       # Lint + run all tests
make test-only  # Run tests without linting
make test-unit  # Run unit tests only (skip integration)
make build      # Verify compilation
```

## Project Structure

```
any-llm-go/
├── anyllm.go           # Root package - re-exports types for simple imports
├── config/config.go    # Functional options pattern for configuration
├── errors/errors.go    # Normalized error types with sentinel errors
├── providers/
│   ├── types.go        # Core interfaces and shared types
│   ├── anthropic/      # Anthropic Claude provider (reference implementation)
│   ├── openai/         # OpenAI provider
│   └── ollama/         # Ollama local provider
├── internal/testutil/  # Test utilities and fixtures
└── docs/               # Documentation
```

## Architecture

### Import Pattern

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/openai"
)
```

### Core Interfaces (providers/types.go)

- `Provider` - Required: `Name()`, `Completion()`, `CompletionStream()`
- `CapabilityProvider` - Optional: `Capabilities()`
- `EmbeddingProvider` - Optional: `Embedding()`
- `ModelLister` - Optional: `ListModels()`
- `ErrorConverter` - Optional: `ConvertError()`

### Error Handling

Normalized errors in `errors/errors.go`: `ErrRateLimit`, `ErrAuthentication`, `ErrContextLength`, `ErrContentFilter`, `ErrModelNotFound`, `ErrInvalidRequest`, `ErrMissingAPIKey`.

Providers implement `ErrorConverter` using `errors.As` with SDK typed errors (not string matching).

## Provider Implementation Guidelines

### File Organization

1. Package declaration & imports
2. Constants (grouped by purpose, unexported)
3. Interface assertions (`var _ Interface = (*Type)(nil)`)
4. Types (exported first, then unexported helpers)
5. Constructor (`New()`)
6. Exported methods (alphabetically)
7. Unexported methods (alphabetically)
8. Package-level functions (alphabetically)

### Key Patterns

- **Configuration**: Functional options with validation
- **Constants**: Extract ALL magic strings to named constants
- **Streaming**: Break monolithic handlers into focused methods (see `anthropic/anthropic.go`)
- **ID Generation**: Use `crypto/rand`, not package-level mutable state
- **Error Conversion**: Use `errors.As` with SDK typed errors

### Testing

- Use `t.Parallel()` except when using `t.Setenv()`
- Use `t.Helper()` in test helpers
- Use `require` from testify, not `assert`
- Name test case variable `tc`, not `tt`
- Name helpers/mocks with `test`, `mock`, `fake` to distinguish from production code
- Skip integration tests gracefully when provider unavailable

## Adding a New Provider

1. Create `providers/{name}/{name}.go`
2. Implement `Provider` interface (required)
3. Implement optional interfaces as needed
4. Implement `ErrorConverter` using SDK typed errors
5. Add tests with `t.Parallel()`
6. Document in `docs/providers.md`

Reference `providers/anthropic/` as the canonical example.
