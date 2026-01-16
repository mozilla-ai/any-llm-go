# Completion API

The completion API is the primary way to interact with LLM providers.

## Quick Start

```go
import (
    "context"

    anyllm "github.com/mozilla-ai/any-llm-go"
    _ "github.com/mozilla-ai/any-llm-go/providers/openai"
)

ctx := context.Background()

response, err := anyllm.Completion(ctx, "openai:gpt-4o-mini", []anyllm.Message{
    {Role: anyllm.RoleUser, Content: "Hello!"},
})
```

## Functions

### `Completion`

```go
func Completion(
    ctx context.Context,
    model string,
    messages []Message,
    opts ...Option,
) (*ChatCompletion, error)
```

Performs a chat completion request using the specified model.

**Parameters:**
- `ctx` - Context for cancellation and timeouts
- `model` - Model string in format `provider:model_id` (e.g., `"openai:gpt-4o-mini"`)
- `messages` - Slice of messages comprising the conversation
- `opts` - Optional configuration options

**Returns:**
- `*ChatCompletion` - The completion response
- `error` - Any error that occurred

**Example:**

```go
response, err := anyllm.Completion(ctx, "anthropic:claude-3-5-haiku-latest", []anyllm.Message{
    {Role: anyllm.RoleSystem, Content: "You are a helpful assistant."},
    {Role: anyllm.RoleUser, Content: "What is Go?"},
})
if err != nil {
    log.Fatal(err)
}

fmt.Println(response.Choices[0].Message.Content)
```

### `CompletionWithParams`

```go
func CompletionWithParams(
    ctx context.Context,
    model string,
    params CompletionParams,
    opts ...Option,
) (*ChatCompletion, error)
```

Performs a chat completion with full parameter control.

**Parameters:**
- `ctx` - Context for cancellation and timeouts
- `model` - Model string in format `provider:model_id`
- `params` - Full completion parameters
- `opts` - Optional configuration options

**Example:**

```go
temp := 0.7
maxTokens := 1000

response, err := anyllm.CompletionWithParams(ctx, "openai:gpt-4o", anyllm.CompletionParams{
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "Write a poem about coding."},
    },
    Temperature: &temp,
    MaxTokens:   &maxTokens,
})
```

## CompletionParams

Full parameters for completion requests:

```go
type CompletionParams struct {
    // Model is the model ID to use (required if not using convenience function)
    Model string `json:"model"`

    // Messages is the conversation history (required)
    Messages []Message `json:"messages"`

    // Temperature controls randomness (0.0-2.0, default varies by provider)
    Temperature *float64 `json:"temperature,omitempty"`

    // TopP controls nucleus sampling (0.0-1.0)
    TopP *float64 `json:"top_p,omitempty"`

    // MaxTokens limits the response length
    MaxTokens *int `json:"max_tokens,omitempty"`

    // Stop sequences that will halt generation
    Stop []string `json:"stop,omitempty"`

    // Stream enables streaming responses
    Stream bool `json:"stream,omitempty"`

    // Tools available for the model to call
    Tools []Tool `json:"tools,omitempty"`

    // ToolChoice controls tool selection behavior
    // Can be "auto", "none", "required", or a ToolChoice struct
    ToolChoice any `json:"tool_choice,omitempty"`

    // ParallelToolCalls allows multiple tool calls in one response
    ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

    // ResponseFormat specifies the output format
    ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

    // ReasoningEffort controls extended thinking (for supported models)
    ReasoningEffort ReasoningEffort `json:"reasoning_effort,omitempty"`

    // Seed for deterministic outputs (if supported)
    Seed *int `json:"seed,omitempty"`

    // User identifier for tracking
    User string `json:"user,omitempty"`
}
```

## Message Types

### Basic Message

```go
type Message struct {
    Role       string      `json:"role"`
    Content    any         `json:"content"` // string or []ContentPart
    Name       string      `json:"name,omitempty"`
    ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
    ToolCallID string      `json:"tool_call_id,omitempty"`
    Reasoning  *Reasoning  `json:"reasoning,omitempty"`
}
```

### Role Constants

```go
const (
    RoleSystem    = "system"
    RoleUser      = "user"
    RoleAssistant = "assistant"
    RoleTool      = "tool"
)
```

### Multimodal Content

For messages with images or other content types:

```go
message := anyllm.Message{
    Role: anyllm.RoleUser,
    Content: []anyllm.ContentPart{
        {Type: "text", Text: "What's in this image?"},
        {Type: "image_url", ImageURL: &anyllm.ImageURL{
            URL: "https://example.com/image.jpg",
        }},
    },
}
```

## Response Types

### ChatCompletion

```go
type ChatCompletion struct {
    ID                string   `json:"id"`
    Object            string   `json:"object"`
    Created           int64    `json:"created"`
    Model             string   `json:"model"`
    Choices           []Choice `json:"choices"`
    Usage             *Usage   `json:"usage,omitempty"`
    SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}
```

### Choice

```go
type Choice struct {
    Index        int     `json:"index"`
    Message      Message `json:"message"`
    FinishReason string  `json:"finish_reason,omitempty"`
    Logprobs     any     `json:"logprobs,omitempty"`
}
```

### Finish Reasons

```go
const (
    FinishReasonStop       = "stop"
    FinishReasonLength     = "length"
    FinishReasonToolCalls  = "tool_calls"
    FinishReasonContentFilter = "content_filter"
)
```

## Tool Calling

### Defining Tools

```go
tools := []anyllm.Tool{
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
                        "description": "City name",
                    },
                },
                "required": []string{"location"},
            },
        },
    },
}
```

### Processing Tool Calls

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model:    "gpt-4o-mini",
    Messages: messages,
    Tools:    tools,
})

if response.Choices[0].FinishReason == anyllm.FinishReasonToolCalls {
    for _, tc := range response.Choices[0].Message.ToolCalls {
        // Process tool call
        result := executeFunction(tc.Function.Name, tc.Function.Arguments)

        // Add tool result to messages
        messages = append(messages, response.Choices[0].Message)
        messages = append(messages, anyllm.Message{
            Role:       anyllm.RoleTool,
            Content:    result,
            ToolCallID: tc.ID,
        })
    }

    // Continue conversation with tool results
    response, err = provider.Completion(ctx, anyllm.CompletionParams{
        Model:    "gpt-4o-mini",
        Messages: messages,
        Tools:    tools,
    })
}
```

## Provider Instance

For better performance with multiple requests:

```go
import "github.com/mozilla-ai/any-llm-go/providers/openai"

provider, err := openai.New()
if err != nil {
    log.Fatal(err)
}

// Use provider.Completion() directly
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model:    "gpt-4o-mini",
    Messages: messages,
})
```

## See Also

- [Streaming](streaming.md) - Streaming responses
- [Errors](errors.md) - Error handling
- [Types](types.md) - All type definitions
