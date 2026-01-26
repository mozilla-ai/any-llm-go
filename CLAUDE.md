# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

any-llm-go is the official Go port of [any-llm](https://github.com/mozilla-ai/any-llm). It provides a unified interface for communicating with multiple LLM providers, normalizing all responses to OpenAI's format.

## Build and Test Commands

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run tests for a specific package
go test ./providers/openai/...
go test ./providers/anthropic/...

# Run linter
golangci-lint run

# Build
go build ./...
```

## Project Structure

```
any-llm-go/
├── anyllm.go           # Root package - re-exports types for simple imports
├── config/
│   └── config.go       # Functional options pattern for configuration
├── errors/
│   └── errors.go       # Normalized error types with sentinel errors
├── providers/
│   ├── types.go        # Core interfaces and shared types
│   ├── anthropic/      # Anthropic Claude provider
│   ├── openai/         # OpenAI provider
│   ├── ollama/         # Ollama local provider
│   └── platform/       # Internal platform proxy provider
├── internal/
│   └── testutil/       # Test utilities and mocks
├── examples/           # Usage examples
└── docs/               # Documentation
```

## Architecture

### Import Pattern

The root `anyllm` package re-exports all types, enabling two-import usage:

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/openai"
)
```

### Core Interfaces (providers/types.go)

**Provider** - Required interface for all providers:
```go
type Provider interface {
    Name() string
    Completion(ctx context.Context, params CompletionParams) (*ChatCompletion, error)
    CompletionStream(ctx context.Context, params CompletionParams) (<-chan ChatCompletionChunk, <-chan error)
}
```

**Optional interfaces:**
- `CapabilityProvider` - Reports provider capabilities via `Capabilities()`
- `EmbeddingProvider` - Adds `Embedding()` for text embeddings
- `ModelLister` - Adds `ListModels()` to enumerate available models
- `ErrorConverter` - Converts SDK errors to normalized error types

### Configuration (config/config.go)

Uses functional options pattern with validation:

```go
provider, err := openai.New(
    anyllm.WithAPIKey("sk-..."),
    anyllm.WithBaseURL("https://custom.endpoint"),
    anyllm.WithTimeout(60 * time.Second),
)
```

Options: `WithAPIKey`, `WithBaseURL`, `WithTimeout`, `WithHTTPClient`, `WithExtra`

### Error Handling (errors/errors.go)

Normalized error types with sentinel errors for `errors.Is()` checks:

- `ErrRateLimit` / `RateLimitError`
- `ErrAuthentication` / `AuthenticationError`
- `ErrContextLength` / `ContextLengthError`
- `ErrContentFilter` / `ContentFilterError`
- `ErrModelNotFound` / `ModelNotFoundError`
- `ErrInvalidRequest` / `InvalidRequestError`
- `ErrMissingAPIKey` / `MissingAPIKeyError`

Providers implement `ErrorConverter` to translate SDK-specific errors.

## Code Style Guidelines

1. **Functional Options** - Use for all configuration with validation in each option
2. **SDK Error Types** - Use `errors.As` with provider SDK typed errors (e.g., `api.StatusError`, `api.AuthorizationError`); string matching only acceptable for network-level errors
3. **Flat Control Flow** - Early returns, extract helper functions, avoid deep nesting
4. **Constants** - Extract ALL magic strings to named constants (see Provider File Organization below)
5. **Immutable Parameters** - Never mutate params passed to methods; clone if needed
6. **Parallel Tests** - Use `t.Parallel()` in all tests (except when using `t.Setenv`)
7. **Lazy Initialization** - Use `sync.Once` for lazy HTTP client creation
8. **Interface Assertions** - Use compile-time interface checks: `var _ Interface = (*Type)(nil)`
9. **No Package-Level Mutable State** - Avoid global variables; use crypto/rand for ID generation
10. **Single Responsibility** - Functions should do one thing; split mixed-concern functions

## Provider File Organization Convention

### Order of declarations:

1. Package declaration & doc comment
2. Imports (stdlib → external → internal, alphabetically within groups)
3. Constants (grouped by purpose, unexported, alphabetically within groups)
4. Interface assertions (`var _ Interface = (*Type)(nil)`)
5. Type definitions (exported types first, then unexported helper types)
6. Constructor (`New()`)
7. Exported methods on primary type (alphabetically: `Capabilities`, `Completion`, `CompletionStream`, `ConvertError`, `Embedding`, `ListModels`, `Name`)
8. Unexported methods on primary type (alphabetically, or near their call site)
9. Helper type constructors and methods (grouped by type)
10. Package-level unexported functions (alphabetically)

### Constants organization:

Group constants by purpose with comments:

```go
// Provider configuration constants.
const (
    defaultBaseURL = "..."
    envAPIKey      = "..."
    providerName   = "..."
)

// Done/stop reasons.
const (
    doneReasonLength = "length"
    doneReasonStop   = "stop"
)

// Option keys (for API options maps).
const (
    optionNumCtx      = "num_ctx"
    optionTemperature = "temperature"
    // ...
)

// JSON schema keys and types.
const (
    schemaKeyDescription = "description"
    schemaKeyProperties  = "properties"
    schemaKeyRequired    = "required"
    schemaKeyType        = "type"
    schemaTypeObject     = "object"
)

// Tool and response format constants.
const (
    emptyJSONObject      = "{}"
    responseFormatJSON   = "json_object"
    responseFormatSchema = "json_schema"
    toolCallIDFormat     = "call_%d"
    toolTypeFunction     = "function"
)

// Object type constants.
const (
    objectChatCompletion      = "chat.completion"
    objectChatCompletionChunk = "chat.completion.chunk"
    objectEmbedding           = "embedding"
    objectList                = "list"
    objectModel               = "model"
)
```

### Streaming state pattern:

Break monolithic `handleChunk` into focused methods:

```go
// chunk creates a ChatCompletionChunk with common fields populated.
func (s *streamState) chunk() ChatCompletionChunk { ... }

// handleChunk processes a streaming response.
func (s *streamState) handleChunk(resp *Response) ChatCompletionChunk {
    s.updateMetadata(resp)
    chunk := s.chunk()
    chunk.Choices[0].Delta = s.buildDelta(resp)
    if resp.Done {
        s.handleDone(resp, &chunk)
    }
    return chunk
}
```

### Message conversion pattern:

Use switch-on-role with dedicated functions:

```go
func convertMessage(msg Message) *ProviderMessage {
    switch msg.Role {
    case RoleTool:
        return convertToolMessage(msg)
    case RoleAssistant:
        return convertAssistantMessage(msg)
    case RoleUser:
        return convertUserMessage(msg)
    default:
        return &ProviderMessage{Role: msg.Role, Content: msg.ContentString()}
    }
}
```

### Test helpers:

Test helpers should use `t.Helper()` and descriptive names:

```go
const testTimeout = 5 * time.Second

func skipIfProviderUnavailable(t *testing.T) {
    t.Helper()
    ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
    defer cancel()
    // ... check availability
}
```

### Test assertions:

- Use constants from the package (e.g., `providers.RoleUser`, `objectChatCompletion`)
- Remove redundant assertion messages unless they add clarity
- Use `t.Parallel()` except when `t.Setenv()` is needed

## Adding a New Provider

1. Create `providers/{name}/{name}.go`
2. Implement `Provider` interface
3. Implement `CapabilityProvider` to report capabilities
4. Implement `ErrorConverter` for error normalization
5. Use official SDK when available (prefer over raw HTTP)
6. Add comprehensive tests with `t.Parallel()`
7. Document in `docs/providers.md`

Reference existing providers (e.g., `providers/anthropic/`) for patterns.
