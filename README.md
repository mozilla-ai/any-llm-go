<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/mozilla-ai/any-llm/refs/heads/main/docs/public/images/any-llm-logo-mark.png" width="20%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

# any-llm (Go)

[![Go Reference](https://pkg.go.dev/badge/github.com/mozilla-ai/any-llm-go.svg)](https://pkg.go.dev/github.com/mozilla-ai/any-llm-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/mozilla-ai/any-llm-go)](https://goreportcard.com/report/github.com/mozilla-ai/any-llm-go)
![Go 1.26+](https://img.shields.io/badge/go-1.26%2B-blue.svg)
<a href="https://discord.gg/4gf3zXrQUc">
    <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
</a>

**Communicate with any LLM provider using a single, unified interface.**
Switch between OpenAI, Anthropic, Mistral, Ollama, and more without changing your code.

[Python SDK](https://github.com/mozilla-ai/any-llm) | [Documentation](https://mozilla-ai.github.io/any-llm/) | [Platform (Beta)](https://any-llm.ai/)

</div>

## Quickstart

```bash
go get github.com/mozilla-ai/any-llm-go
export OPENAI_API_KEY="YOUR_KEY_HERE"  # or ANTHROPIC_API_KEY, etc
```

```go
package main

import (
    "context"
    "fmt"
    "log"

    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/openai"
)

func main() {
    ctx := context.Background()

    provider, err := openai.New()
    if err != nil {
        log.Fatal(err)
    }

    response, err := provider.Completion(ctx, anyllm.CompletionParams{
        Model: "gpt-4o-mini",
        Messages: []anyllm.Message{
            {Role: anyllm.RoleUser, Content: "Hello!"},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Choices[0].Message.Content)
}
```

**That's it!** To switch providers, change the import and constructor (e.g., `anthropic.New()` instead of `openai.New()`).

## Installation

### Requirements

- Go 1.26 or newer
- API keys for whichever LLM providers you want to use

Import the main package and the providers you need:

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/openai"    // OpenAI
    "github.com/mozilla-ai/any-llm-go/providers/anthropic" // Anthropic
)
```

See our [list of supported providers](docs/providers.md) to choose which ones you need.

### Setting Up API Keys

Set environment variables for your chosen providers:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export MISTRAL_API_KEY="your-key-here"
# ... etc
```

Alternatively, pass API keys directly in your code:

```go
provider, err := openai.New(anyllm.WithAPIKey("your-key-here"))
```

## any-llm-gateway

any-llm-gateway is an **optional** FastAPI-based proxy server that adds enterprise-grade features on top of the core library:

- **Budget Management** - Enforce spending limits with automatic daily, weekly, or monthly resets
- **API Key Management** - Issue, revoke, and monitor virtual API keys without exposing provider credentials
- **Usage Analytics** - Track every request with full token counts, costs, and metadata
- **Multi-tenant Support** - Manage access and budgets across users and teams

The gateway sits between your applications and LLM providers, exposing an OpenAI-compatible API that works with any supported provider.

### Quick Start

```bash
docker run \
  -e GATEWAY_MASTER_KEY="your-secure-master-key" \
  -e OPENAI_API_KEY="your-api-key" \
  -p 8000:8000 \
  ghcr.io/mozilla-ai/any-llm/gateway:latest
```

> **Note:** You can use a specific release version instead of `latest` (e.g., `1.2.0`). See [available versions](https://github.com/orgs/mozilla-ai/packages/container/package/any-llm%2Fgateway).

### Managed Platform (Beta)

Prefer a hosted experience? The [any-llm platform](https://any-llm.ai/) provides a managed control plane for keys, usage tracking, and cost visibility across providers, while still building on the same `any-llm` interfaces.

## Usage

Create a provider instance and use it for requests:

```go
import (
    "context"
    "fmt"
    "log"

    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/openai"
)

provider, err := openai.New(anyllm.WithAPIKey("your-api-key"))
if err != nil {
    log.Fatal(err)
}

ctx := context.Background()

response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "Hello!"},
    },
})
if err != nil {
    log.Fatal(err)
}

fmt.Println(response.Choices[0].Message.Content)
```

Provider instances are reusable and recommended for production applications.

### Streaming

```go
chunks, errs := provider.CompletionStream(ctx, anyllm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "Write a short poem about Go."},
    },
})

for chunk := range chunks {
    if len(chunk.Choices) > 0 {
        fmt.Print(chunk.Choices[0].Delta.Content)
    }
}

if err := <-errs; err != nil {
    log.Fatal(err)
}
```

### Tools / Function Calling

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "What's the weather in Paris?"},
    },
    Tools: []anyllm.Tool{
        {
            Type: "function",
            Function: anyllm.Function{
                Name:        "get_weather",
                Description: "Get the current weather for a location",
                Parameters: map[string]any{
                    "type": "object",
                    "properties": map[string]any{
                        "location": map[string]any{
                            "type":        "string",
                            "description": "The city name",
                        },
                    },
                    "required": []string{"location"},
                },
            },
        },
    },
    ToolChoice: "auto",
})

// Check for tool calls.
if len(response.Choices[0].Message.ToolCalls) > 0 {
    tc := response.Choices[0].Message.ToolCalls[0]
    fmt.Printf("Function: %s, Args: %s\n", tc.Function.Name, tc.Function.Arguments)
}
```

### Extended Thinking (Reasoning)

For models that support extended thinking (like Claude):

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "claude-sonnet-4-20250514",
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "Solve this step by step: What is 15% of 80?"},
    },
    ReasoningEffort: anyllm.ReasoningEffortMedium,
})

if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
fmt.Println("Answer:", response.Choices[0].Message.Content)
```

### Embeddings

```go
provider, _ := openai.New()
result, err := provider.Embedding(ctx, anyllm.EmbeddingParams{
    Model: "text-embedding-3-small",
    Input: "Hello world",
})
```

### Listing Models

```go
provider, _ := openai.New()
models, err := provider.ListModels(ctx)
for _, model := range models.Data {
    fmt.Println(model.ID)
}
```

### Moderation

The gateway provider supports OpenAI-compatible content moderation. Use
`errors.As` with `*anyllm.UnsupportedOperationError` (or `errors.Is` with
`anyllm.ErrUnsupported`) to detect providers that do not support moderation.

```go
import (
    stderrors "errors"

    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/config"
    "github.com/mozilla-ai/any-llm-go/providers/gateway"
)

provider, err := gateway.New(config.WithBaseURL("https://gw.example.com"))
if err != nil {
    log.Fatal(err)
}

resp, err := provider.Moderation(ctx, anyllm.ModerationParams{
    Model: "openai:omni-moderation-latest",
    Input: "I want to hurt someone",
})
if err != nil {
    var unsup *anyllm.UnsupportedOperationError
    if stderrors.As(err, &unsup) {
        // Provider does not support moderation; pick another model.
        log.Printf("%s cannot do %s", unsup.Provider, unsup.Operation)
        return
    }
    log.Fatal(err)
}
if resp.Results[0].Flagged {
    // Handle flagged content.
}
```

### Error Handling

All provider errors are normalized to common error types:

```go
response, err := provider.Completion(ctx, params)
if err != nil {
    switch {
    case errors.Is(err, anyllm.ErrRateLimit):
        // Handle rate limiting - maybe retry with backoff.
    case errors.Is(err, anyllm.ErrAuthentication):
        // Handle auth errors - check API key.
    case errors.Is(err, anyllm.ErrContextLength):
        // Handle context too long - reduce input.
    default:
        // Handle other errors.
    }
}
```

You can also use type assertions for more details:

```go
var rateLimitErr *anyllm.RateLimitError
if errors.As(err, &rateLimitErr) {
    fmt.Printf("Rate limited by %s: %s\n", rateLimitErr.Provider, rateLimitErr.Message)
}
```

## Supported Providers

|  Provider  | Completion  |  Streaming  |  Tools |  Reasoning  |  Embeddings  |
|:----------:|:-----------:|:-----------:|:------:|:-----------:|:------------:|
| Anthropic  |      ✅      |      ✅      |      ✅ |      ✅      |      ❌       |
| Azure OpenAI |    ✅      |      ✅      |      ✅ |      ✅      |      ✅       |
|  DeepSeek  |      ✅      |      ✅      |      ✅ |      ✅      |      ❌       |
|   Gemini   |      ✅      |      ✅      |      ✅ |      ✅      |      ✅       |
|    Groq    |      ✅      |      ✅      |      ✅ |      ❌      |      ❌       |
|  llama.cpp |      ✅      |      ✅      |      ✅ |      ❌      |      ✅       |
| Llamafile  |      ✅      |      ✅      |      ✅ |      ❌      |      ✅       |
|  Mistral   |      ✅      |      ✅      |      ✅ |      ✅      |      ✅       |
|   Ollama   |      ✅      |      ✅      |      ✅ |      ✅      |      ✅       |
|   OpenAI   |      ✅      |      ✅      |      ✅ |      ✅      |      ✅       |
|    z.ai    |      ✅      |      ✅      |      ✅ |      ✅      |      ❌       |

## Why choose `any-llm-go`?

- **Simple, unified interface** - Same types and patterns across all providers, switch models with just a string change
- **Developer friendly** - Full type definitions for better IDE support and clear, actionable error messages
- **Leverages official provider SDKs** - Uses `github.com/openai/openai-go` and `github.com/anthropics/anthropic-sdk-go` for maximum compatibility
- **Stays framework-agnostic** so it can be used across different projects and use cases
- **Idiomatic Go** - Follows Go conventions with proper error handling and context support
- **Streaming support** - Channel-based streaming that's natural in Go
- **Battle-tested** - Based on the proven [any-llm](https://github.com/mozilla-ai/any-llm) Python library

## Development

```bash
make lint       # Run linter with auto-fix
make test       # Lint + run all tests
make test-only  # Run tests without linting
make test-unit  # Run unit tests only (skip integration)
make build      # Verify compilation
```

## Documentation

- **[Full Documentation](https://mozilla-ai.github.io/any-llm/)** - Complete guides and API reference
- **[Supported Providers](https://mozilla-ai.github.io/any-llm/providers/)** - List of all supported LLM providers
- **[Gateway Documentation](https://mozilla-ai.github.io/any-llm/gateway/overview/)** - Gateway setup and deployment
- **[Python SDK](https://github.com/mozilla-ai/any-llm)** - The full Python SDK with direct provider access
- **[any-llm Platform (Beta)](https://any-llm.ai/)** - Hosted control plane for key management, usage tracking, and cost visibility

## Contributing

We welcome contributions from developers of all skill levels! Please see our [Contributing Guide](CONTRIBUTING.md) or open an issue to discuss changes.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
