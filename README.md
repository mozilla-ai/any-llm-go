<p align="center">
  <picture>
    <img src="https://raw.githubusercontent.com/mozilla-ai/any-llm/refs/heads/main/docs/images/any-llm-logo-mark.png" width="20%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

# any-llm-go

[![Go Reference](https://pkg.go.dev/badge/github.com/mozilla-ai/any-llm-go.svg)](https://pkg.go.dev/github.com/mozilla-ai/any-llm-go)
[![Go Report Card](https://goreportcard.com/badge/github.com/mozilla-ai/any-llm-go)](https://goreportcard.com/report/github.com/mozilla-ai/any-llm-go)

![Go 1.23+](https://img.shields.io/badge/go-1.23%2B-blue.svg)

**Communicate with any LLM provider using a single, unified interface.**
Switch between OpenAI, Anthropic, Mistral, Ollama, and more without changing your code.

[Documentation](docs/) | [Examples](examples/) | [Contributing](CONTRIBUTING.md)

</div>

## Quickstart

```go
go get github.com/mozilla-ai/any-llm-go

export OPENAI_API_KEY="YOUR_KEY_HERE"  // or ANTHROPIC_API_KEY, etc
```

```go
package main

import (
    "context"
    "fmt"
    "log"

    github.com/mozilla-ai/any-llm-go"
    _ "github.com/mozilla-ai/any-llm-go/providers/openai" // Register provider
)

func main() {
    ctx := context.Background()

    response, err := llm.Completion(ctx, "openai:gpt-4o-mini", []llm.Message{
        {Role: "user", Content: "Hello!"},
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Choices[0].Message.Content)
}
```

**That's it!** Change the provider name and model to switch between LLM providers.

## Installation

### Requirements

- Go 1.23 or newer
- API keys for whichever LLM providers you want to use

### Basic Installation

```bash
go get github.com/mozilla-ai/any-llm-go
```

Import the providers you need:

```go
import (
    github.com/mozilla-ai/any-llm-go"
    _ "github.com/mozilla-ai/any-llm-go/providers/openai"    // OpenAI
    _ "github.com/mozilla-ai/any-llm-go/providers/anthropic" // Anthropic
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

Alternatively, pass API keys directly in your code using options:

```go
provider, err := openai.New(llm.WithAPIKey("your-key-here"))
```

## Why choose `any-llm-go`?

- **Simple, unified interface** - Single function for all providers, switch models with just a string change
- **Idiomatic Go** - Follows Go conventions with proper error handling and context support
- **Leverages official provider SDKs** - Uses `github.com/openai/openai-go` and `github.com/anthropics/anthropic-sdk-go`
- **Type-safe** - Full type definitions for all request and response types
- **Streaming support** - Channel-based streaming that's natural in Go
- **Battle-tested patterns** - Based on the proven [any-llm](https://github.com/mozilla-ai/any-llm) Python library

## Usage

`any-llm-go` offers two main approaches for interacting with LLM providers:

### Option 1: Direct API Functions (Recommended for Quick Usage)

Use the convenience functions with `provider:model` format:

```go
import (
    "context"

    github.com/mozilla-ai/any-llm-go"
    _ "github.com/mozilla-ai/any-llm-go/providers/openai"
)

ctx := context.Background()

response, err := llm.Completion(ctx, "openai:gpt-4o-mini", []llm.Message{
    {Role: "user", Content: "Hello!"},
})
if err != nil {
    log.Fatal(err)
}

fmt.Println(response.Choices[0].Message.Content)
```

### Option 2: Provider Instance (Recommended for Production)

For applications that need to reuse providers or require more control:

```go
import (
    "context"

    github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Create provider once, reuse for multiple requests
provider, err := openai.New(llm.WithAPIKey("your-api-key"))
if err != nil {
    log.Fatal(err)
}

ctx := context.Background()

response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []llm.Message{
        {Role: "user", Content: "Hello!"},
    },
})
```

### When to Use Which Approach

| Approach | Best For | Connection Handling |
|----------|----------|---------------------|
| **Direct API Functions** (`llm.Completion`) | Scripts, quick prototypes, single requests | New client per call |
| **Provider Instance** (`provider.Completion`) | Production apps, multiple requests | Reuses client |

Both approaches support identical features: streaming, tools, reasoning, etc.

### Streaming

Use channels for streaming responses:

```go
chunks, errs := provider.CompletionStream(ctx, llm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []llm.Message{
        {Role: "user", Content: "Write a short poem about Go."},
    },
    Stream: true,
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
response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []llm.Message{
        {Role: "user", Content: "What's the weather in Paris?"},
    },
    Tools: []llm.Tool{
        {
            Type: "function",
            Function: llm.Function{
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

// Check for tool calls
if len(response.Choices[0].Message.ToolCalls) > 0 {
    tc := response.Choices[0].Message.ToolCalls[0]
    fmt.Printf("Function: %s, Args: %s\n", tc.Function.Name, tc.Function.Arguments)
}
```

### Extended Thinking (Reasoning)

For models that support extended thinking (like Claude):

```go
response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "claude-sonnet-4-20250514",
    Messages: []llm.Message{
        {Role: "user", Content: "Solve this step by step: What is 15% of 80?"},
    },
    ReasoningEffort: llm.ReasoningEffortMedium,
})

if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
fmt.Println("Answer:", response.Choices[0].Message.Content)
```

### Error Handling

All provider errors are normalized to common error types:

```go
response, err := provider.Completion(ctx, params)
if err != nil {
    switch {
    case errors.Is(err, llm.ErrRateLimit):
        // Handle rate limiting - maybe retry with backoff
    case errors.Is(err, llm.ErrAuthentication):
        // Handle auth errors - check API key
    case errors.Is(err, llm.ErrContextLength):
        // Handle context too long - reduce input
    default:
        // Handle other errors
    }
}
```

You can also use type assertions for more details:

```go
var rateLimitErr *llm.RateLimitError
if errors.As(err, &rateLimitErr) {
    fmt.Printf("Rate limited by %s: %s\n", rateLimitErr.Provider, rateLimitErr.Message)
}
```

### Finding the Right Model

The model format is `provider:model_id` where:
- `provider` matches our [supported provider names](docs/providers.md)
- `model_id` is passed directly to the provider

To find available models:
- Check the provider's documentation
- Use the `ListModels` API (if the provider supports it):

```go
models, err := llm.ListModels(ctx, "openai")
for _, model := range models.Data {
    fmt.Println(model.ID)
}
```

## Supported Providers

| Provider | Completion | Streaming | Tools | Reasoning | Embeddings |
|----------|:----------:|:---------:|:-----:|:---------:|:----------:|
| OpenAI | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ | ✅ | ❌ |

More providers coming soon! See [docs/providers.md](docs/providers.md) for the full list.

## Documentation

- **[Quickstart Guide](docs/quickstart.md)** - Get up and running quickly
- **[Supported Providers](docs/providers.md)** - List of all supported LLM providers
- **[API Reference](docs/api/)** - Complete API documentation
- **[Examples](examples/)** - Code examples for common use cases

## Comparison with Python any-llm

This is the official Go port of [any-llm](https://github.com/mozilla-ai/any-llm). Key differences:

| Feature | Python any-llm | Go any-llm |
|---------|----------------|------------|
| Async support | `async`/`await` | Goroutines + channels |
| Streaming | Iterators | Channels |
| Error handling | Exceptions | `error` return values |
| Type hints | Type annotations | Static types |
| Provider registration | Automatic | Import for side effects |

## Contributing

We welcome contributions from developers of all skill levels! Please see our [Contributing Guide](CONTRIBUTING.md) or open an issue to discuss changes.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
