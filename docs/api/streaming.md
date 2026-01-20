# Streaming API

Streaming allows you to receive partial responses as they're generated, enabling real-time output and better user experience.

## Quick Start

```go
chunks, errs := provider.CompletionStream(ctx, anyllm.CompletionParams{
    Model: "gpt-4o-mini",
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "Write a short story."},
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

## Provider Interface

### `CompletionStream`

```go
func (p *Provider) CompletionStream(
    ctx context.Context,
    params CompletionParams,
) (<-chan ChatCompletionChunk, <-chan error)
```

Performs a streaming chat completion request.

**Parameters:**
- `ctx` - Context for cancellation and timeouts
- `params` - Completion parameters including model and messages

**Returns:**
- `<-chan ChatCompletionChunk` - Channel that receives chunks as they arrive
- `<-chan error` - Channel that receives any error (at most one, then closed)

Both channels are closed when the stream ends.

## Chunk Types

### ChatCompletionChunk

```go
type ChatCompletionChunk struct {
    ID                string        `json:"id"`
    Object            string        `json:"object"` // "chat.completion.chunk"
    Created           int64         `json:"created"`
    Model             string        `json:"model"`
    Choices           []ChunkChoice `json:"choices"`
    Usage             *Usage        `json:"usage,omitempty"`
    SystemFingerprint string        `json:"system_fingerprint,omitempty"`
}
```

### ChunkChoice

```go
type ChunkChoice struct {
    Index        int        `json:"index"`
    Delta        ChunkDelta `json:"delta"`
    FinishReason string     `json:"finish_reason,omitempty"`
}
```

### ChunkDelta

```go
type ChunkDelta struct {
    Role      string      `json:"role,omitempty"`
    Content   string      `json:"content,omitempty"`
    ToolCalls []ToolCall  `json:"tool_calls,omitempty"`
    Reasoning *Reasoning  `json:"reasoning,omitempty"`
}
```

## Usage Patterns

### Basic Streaming

```go
chunks, errs := provider.CompletionStream(ctx, params)

// Process chunks.
for chunk := range chunks {
    if len(chunk.Choices) > 0 {
        delta := chunk.Choices[0].Delta
        if delta.Content != "" {
            fmt.Print(delta.Content)
        }
    }
}

// Always check for errors.
if err := <-errs; err != nil {
    log.Printf("Stream error: %v", err)
}
```

### Collecting Full Response

```go
chunks, errs := provider.CompletionStream(ctx, params)

var fullContent strings.Builder
var finishReason string

for chunk := range chunks {
    if len(chunk.Choices) > 0 {
        choice := chunk.Choices[0]
        fullContent.WriteString(choice.Delta.Content)
        if choice.FinishReason != "" {
            finishReason = choice.FinishReason
        }
    }
}

if err := <-errs; err != nil {
    log.Fatal(err)
}

fmt.Printf("Complete response: %s\n", fullContent.String())
fmt.Printf("Finish reason: %s\n", finishReason)
```

### Streaming with Tool Calls

```go
chunks, errs := provider.CompletionStream(ctx, anyllm.CompletionParams{
    Model:    "gpt-4o-mini",
    Messages: messages,
    Tools:    tools,
    Stream:   true,
})

var toolCalls []anyllm.ToolCall
toolCallArgs := make(map[int]strings.Builder)

for chunk := range chunks {
    if len(chunk.Choices) > 0 {
        delta := chunk.Choices[0].Delta

        // Handle content.
        if delta.Content != "" {
            fmt.Print(delta.Content)
        }

        // Handle tool calls.
        for _, tc := range delta.ToolCalls {
            if tc.Function.Name != "" {
                // New tool call.
                toolCalls = append(toolCalls, tc)
            }
            if tc.Function.Arguments != "" {
                // Accumulate arguments.
                idx := len(toolCalls) - 1
                toolCallArgs[idx].WriteString(tc.Function.Arguments)
            }
        }
    }
}

if err := <-errs; err != nil {
    log.Fatal(err)
}

// Process completed tool calls.
for i, tc := range toolCalls {
    tc.Function.Arguments = toolCallArgs[i].String()
    fmt.Printf("Tool: %s(%s)\n", tc.Function.Name, tc.Function.Arguments)
}
```

### Streaming with Reasoning (Claude)

```go
chunks, errs := provider.CompletionStream(ctx, anyllm.CompletionParams{
    Model:           "claude-sonnet-4-20250514",
    Messages:        messages,
    ReasoningEffort: anyllm.ReasoningEffortMedium,
    Stream:          true,
})

fmt.Println("Thinking...")
for chunk := range chunks {
    if len(chunk.Choices) > 0 {
        delta := chunk.Choices[0].Delta

        // Print thinking content.
        if delta.Reasoning != nil && delta.Reasoning.Content != "" {
            fmt.Print(delta.Reasoning.Content)
        }

        // Print response content.
        if delta.Content != "" {
            fmt.Print(delta.Content)
        }
    }
}
```

### Cancellation

Use context cancellation to stop a stream:

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

chunks, errs := provider.CompletionStream(ctx, params)

for chunk := range chunks {
    // Process chunk...

    // Cancel if needed.
    if someCondition {
        cancel()
        break
    }
}

// Check for cancellation error.
if err := <-errs; err != nil {
    if errors.Is(err, context.Canceled) {
        fmt.Println("Stream cancelled")
    } else if errors.Is(err, context.DeadlineExceeded) {
        fmt.Println("Stream timed out")
    } else {
        log.Fatal(err)
    }
}
```

## Provider-Specific Notes

### OpenAI

- Supports streaming for all chat completion models.
- `usage` field included in final chunk (optional, depends on request).

### Anthropic

- Full streaming support including thinking content.
- Events include: `message_start`, `content_block_start`, `content_block_delta`, `message_delta`.
- All events normalized to OpenAI chunk format.

## Best Practices

1. **Always drain the channels** - Read from both channels until they're closed.
2. **Check errors** - Always check the error channel after processing chunks.
3. **Use context** - Pass a context with timeout/cancellation for production code.
4. **Handle partial data** - Be prepared for chunks with empty content.
5. **Buffer tool call arguments** - Tool call arguments come in pieces during streaming.

## See Also

- [Completion](completion.md) - Non-streaming completions
- [Errors](errors.md) - Error handling
