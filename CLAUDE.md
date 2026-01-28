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
- **Constants**: Extract ALL magic strings to named constants (including response format types like `json_object`)
- **Streaming**: Break monolithic handlers into focused methods (see `anthropic/anthropic.go`)
- **Streaming Safety**: Always use `select` with `ctx.Done()` when sending to channels in goroutines to prevent blocking forever if consumer abandons
- **ID Generation**: Use `crypto/rand`, not package-level mutable state
- **Error Conversion**: Use `errors.As` with SDK typed errors; avoid string matching when possible
- **Input Validation**: Validate required fields (Model non-empty, Messages has entries) before API calls
- **Unknown Values**: Never silently convert unknown enum values (e.g., unknown role → user); error or log warning instead
- **Struct Field Order**: Order struct fields A-Z (don't optimize for padding)

### OpenAI-Compatible Providers

For providers that expose OpenAI-compatible APIs but don't have their own Go SDK (Llamafile, vLLM, LM Studio, etc.):
- Use the base provider in `providers/openai/base.go`
- Import: `"github.com/mozilla-ai/any-llm-go/providers/openai"`
- Create thin wrapper that calls `openai.NewBase()` with provider-specific config
- Add interface assertions in the wrapper package

### Testing

- Use `t.Parallel()` except when using `t.Setenv()`
- Use `t.Helper()` in test helpers
- Use `require` from testify, not `assert`
- Name test case variable `tc`, not `tt`
- Name helpers/mocks with `test`, `mock`, `fake` to distinguish from production code
- Skip integration tests gracefully when provider unavailable
- Use constants (e.g., `objectChatCompletion`) instead of string literals in test assertions
- Base packages need their own test suites, not just wrapper tests

## Adding a New Provider

1. Create `providers/{name}/{name}.go`
2. Implement `Provider` interface (required)
3. Implement optional interfaces as needed
4. Implement `ErrorConverter` using SDK typed errors
5. Add tests with `t.Parallel()`
6. Document in `docs/providers.md`

Reference `providers/anthropic/` as the canonical example.
