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
    // Create provider instance.
    provider, err := openai.New()
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    // Make a completion request.
    response, err := provider.Completion(ctx, anyllm.CompletionParams{
        Model: "gpt-4o-mini",
        Messages: []anyllm.Message{
            {Role: anyllm.RoleUser, Content: "Say hello in three languages!"},
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
provider, err := openai.New(anyllm.WithAPIKey("sk-your-api-key"))
```

## Streaming Responses

For real-time output, use streaming:

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
    provider, err := openai.New()
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()

    chunks, errs := provider.CompletionStream(ctx, anyllm.CompletionParams{
        Model: "gpt-4o-mini",
        Messages: []anyllm.Message{
            {Role: anyllm.RoleUser, Content: "Write a haiku about programming."},
        },
        Stream: true,
    })

    // Print chunks as they arrive.
    for chunk := range chunks {
        if len(chunk.Choices) > 0 {
            fmt.Print(chunk.Choices[0].Delta.Content)
        }
    }
    fmt.Println()

    // Check for errors.
    if err := <-errs; err != nil {
        log.Fatal(err)
    }
}
```

## Using System Messages

Guide the model's behavior with system messages:

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []anyllm.Message{
        {Role: anyllm.RoleSystem, Content: "You are a helpful assistant that speaks like a pirate."},
        {Role: anyllm.RoleUser, Content: "How do I make coffee?"},
    },
})
```

## Adjusting Parameters

Control the model's output:

```go
temp := 0.7
maxTokens := 500

response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "Write a creative story."},
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
    // Check for specific error types.
    if errors.Is(err, anyllm.ErrRateLimit) {
        fmt.Println("Rate limited - please retry later")
        return
    }
    if errors.Is(err, anyllm.ErrAuthentication) {
        fmt.Println("Invalid API key")
        return
    }
    if errors.Is(err, anyllm.ErrContextLength) {
        fmt.Println("Input too long - please reduce message size")
        return
    }

    // Generic error.
    log.Fatal(err)
}
```

## Switching Providers

One of the main benefits of any-llm-go is easy provider switching:

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/anthropic"
    "github.com/mozilla-ai/any-llm-go/providers/openai"
)

func tryProvider(providerName string, model string, messages []anyllm.Message) error {
    var provider anyllm.Provider
    var err error

    switch providerName {
    case "openai":
        provider, err = openai.New()
    case "anthropic":
        provider, err = anthropic.New()
    }
    if err != nil {
        return err
    }

    response, err := provider.Completion(ctx, anyllm.CompletionParams{
        Model:    model,
        Messages: messages,
    })
    if err != nil {
        return err
    }

    fmt.Printf("%s: %s\n", providerName, response.Choices[0].Message.Content)
    return nil
}
```

## Next Steps

- [Supported Providers](providers.md) - See all available providers
- [API Reference](api/) - Detailed API documentation
- [Examples](../examples/) - More code examples
