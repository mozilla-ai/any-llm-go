# Quickstart Guide

Get up and running with any-llm-go in minutes.

## Prerequisites

- **Go 1.23+** - [Download Go](https://go.dev/dl/)
- **API Keys** - Get API keys from your chosen providers:
  - [OpenAI](https://platform.openai.com/api-keys)
  - [Anthropic](https://console.anthropic.com/)

## Installation

Add any-llm-go to your project:

```bash
go get github.com/mozilla-ai/any-llm-go
```

## Setting Up API Keys

Set environment variables for your providers:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Your First Request

### Using the Convenience Function

The simplest way to make a request:

```go
package main

import (
    "context"
    "fmt"
    "log"

    github.com/mozilla-ai/any-llm-go"
    _ "github.com/mozilla-ai/any-llm-go/providers/openai" // Register the provider
)

func main() {
    ctx := context.Background()

    response, err := llm.Completion(ctx, "openai:gpt-4o-mini", []llm.Message{
        {Role: llm.RoleUser, Content: "Say hello in three languages!"},
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Choices[0].Message.Content)
}
```

The model string format is `provider:model_id`. This tells any-llm-go which provider to use and which model to request.

### Using a Provider Instance

For production code where you'll make multiple requests:

```go
package main

import (
    "context"
    "fmt"
    "log"

    github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/anthropic"
)

func main() {
    // Create provider once
    provider, err := anthropic.New()
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    // Make requests
    response, err := provider.Completion(ctx, llm.CompletionParams{
        Model: "claude-3-5-haiku-latest",
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: "What's the capital of France?"},
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Choices[0].Message.Content)
}
```

## Passing API Keys Directly

If you prefer not to use environment variables:

```go
provider, err := openai.New(llm.WithAPIKey("sk-your-api-key"))
```

Or with the convenience function:

```go
response, err := llm.Completion(ctx, "openai:gpt-4o-mini", messages,
    llm.WithAPIKey("sk-your-api-key"),
)
```

## Streaming Responses

For real-time output, use streaming:

```go
package main

import (
    "context"
    "fmt"
    "log"

    github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/openai"
)

func main() {
    provider, err := openai.New()
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    chunks, errs := provider.CompletionStream(ctx, llm.CompletionParams{
        Model: "gpt-4o-mini",
        Messages: []llm.Message{
            {Role: llm.RoleUser, Content: "Write a haiku about programming."},
        },
        Stream: true,
    })

    // Print chunks as they arrive
    for chunk := range chunks {
        if len(chunk.Choices) > 0 {
            fmt.Print(chunk.Choices[0].Delta.Content)
        }
    }
    fmt.Println() // Final newline

    // Check for errors
    if err := <-errs; err != nil {
        log.Fatal(err)
    }
}
```

## Using System Messages

Guide the model's behavior with system messages:

```go
response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []llm.Message{
        {Role: llm.RoleSystem, Content: "You are a helpful assistant that speaks like a pirate."},
        {Role: llm.RoleUser, Content: "How do I make coffee?"},
    },
})
```

## Adjusting Parameters

Control the model's output:

```go
temp := 0.7
maxTokens := 500

response, err := provider.Completion(ctx, llm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: "Write a creative story."},
    },
    Temperature: &temp,
    MaxTokens:   &maxTokens,
    Stop:        []string{"\n\n", "THE END"},
})
```

## Error Handling

Handle errors appropriately:

```go
import "errors"

response, err := provider.Completion(ctx, params)
if err != nil {
    // Check for specific error types
    if errors.Is(err, llm.ErrRateLimit) {
        fmt.Println("Rate limited - please retry later")
        return
    }
    if errors.Is(err, llm.ErrAuthentication) {
        fmt.Println("Invalid API key")
        return
    }
    if errors.Is(err, llm.ErrContextLength) {
        fmt.Println("Input too long - please reduce message size")
        return
    }

    // Generic error
    log.Fatal(err)
}
```

## Switching Providers

One of the main benefits of any-llm-go is easy provider switching:

```go
import (
    _ "github.com/mozilla-ai/any-llm-go/providers/openai"
    _ "github.com/mozilla-ai/any-llm-go/providers/anthropic"
)

// Same code works with different providers
models := []string{
    "openai:gpt-4o-mini",
    "anthropic:claude-3-5-haiku-latest",
}

for _, model := range models {
    response, err := llm.Completion(ctx, model, messages)
    if err != nil {
        log.Printf("Error with %s: %v", model, err)
        continue
    }
    fmt.Printf("%s: %s\n", model, response.Choices[0].Message.Content)
}
```

## Next Steps

- [Supported Providers](providers.md) - See all available providers
- [API Reference](api/) - Detailed API documentation
- [Examples](../examples/) - More code examples
