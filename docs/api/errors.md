# Error Handling

any-llm-go normalizes provider-specific errors into common error types, making it easy to handle errors consistently across different providers.

## Error Types

### Sentinel Errors

Use `errors.Is()` to check for specific error categories:

```go
import "errors"

response, err := provider.Completion(ctx, params)
if err != nil {
    switch {
    case errors.Is(err, anyllm.ErrRateLimit):
        // Rate limit exceeded - retry with backoff.
    case errors.Is(err, anyllm.ErrAuthentication):
        // Invalid API key.
    case errors.Is(err, anyllm.ErrInvalidRequest):
        // Malformed request.
    case errors.Is(err, anyllm.ErrContextLength):
        // Input too long.
    case errors.Is(err, anyllm.ErrContentFilter):
        // Content blocked by safety filters.
    case errors.Is(err, anyllm.ErrModelNotFound):
        // Model doesn't exist.
    case errors.Is(err, anyllm.ErrProvider):
        // General provider error.
    case errors.Is(err, anyllm.ErrMissingAPIKey):
        // No API key provided.
    default:
        // Unknown error.
    }
}
```

### Error Type Definitions

| Sentinel Error | Description |
|----------------|-------------|
| `ErrRateLimit` | Rate limit exceeded |
| `ErrAuthentication` | Authentication failed (invalid API key) |
| `ErrInvalidRequest` | Request is malformed |
| `ErrContextLength` | Context exceeds model's limit |
| `ErrContentFilter` | Content blocked by safety filters |
| `ErrModelNotFound` | Requested model doesn't exist |
| `ErrProvider` | General provider-side error |
| `ErrMissingAPIKey` | No API key provided |
| `ErrUnsupportedParam` | Parameter not supported by provider |

## Structured Error Types

For more details, use `errors.As()` to access structured error types:

### RateLimitError

```go
var rateLimitErr *anyllm.RateLimitError
if errors.As(err, &rateLimitErr) {
    fmt.Printf("Provider: %s\n", rateLimitErr.Provider)
    fmt.Printf("Message: %s\n", rateLimitErr.Message)
    fmt.Printf("Retry after: %d seconds\n", rateLimitErr.RetryAfter)
}
```

### AuthenticationError

```go
var authErr *anyllm.AuthenticationError
if errors.As(err, &authErr) {
    fmt.Printf("Provider: %s\n", authErr.Provider)
    fmt.Printf("Message: %s\n", authErr.Message)
}
```

### ContextLengthError

```go
var ctxErr *anyllm.ContextLengthError
if errors.As(err, &ctxErr) {
    fmt.Printf("Provider: %s\n", ctxErr.Provider)
    fmt.Printf("Message: %s\n", ctxErr.Message)
}
```

### ProviderError

```go
var providerErr *anyllm.ProviderError
if errors.As(err, &providerErr) {
    fmt.Printf("Provider: %s\n", providerErr.Provider)
    fmt.Printf("Status code: %d\n", providerErr.StatusCode)
    fmt.Printf("Message: %s\n", providerErr.Message)
}
```

### MissingAPIKeyError

```go
var keyErr *anyllm.MissingAPIKeyError
if errors.As(err, &keyErr) {
    fmt.Printf("Provider: %s\n", keyErr.Provider)
    fmt.Printf("Expected env var: %s\n", keyErr.EnvVar)
}
```

## BaseError Structure

All error types embed `BaseError`:

```go
type BaseError struct {
    Code     string // Short error code (e.g., "rate_limit").
    Message  string // Human-readable message.
    Provider string // Provider name (e.g., "openai").
    Err      error  // Underlying error.
}
```

## Accessing the Original Error

All any-llm errors wrap the original provider error:

```go
var baseErr *anyllm.BaseError
if errors.As(err, &baseErr) {
    // Access the original provider error.
    originalErr := baseErr.Err
    fmt.Printf("Original error: %v\n", originalErr)
}
```

## Error Handling Patterns

### Retry with Backoff

```go
func completionWithRetry(ctx context.Context, provider anyllm.Provider, params anyllm.CompletionParams) (*anyllm.ChatCompletion, error) {
    maxRetries := 3
    backoff := time.Second

    for i := 0; i < maxRetries; i++ {
        response, err := provider.Completion(ctx, params)
        if err == nil {
            return response, nil
        }

        if errors.Is(err, anyllm.ErrRateLimit) {
            // Check for retry-after hint.
            var rateLimitErr *anyllm.RateLimitError
            if errors.As(err, &rateLimitErr) && rateLimitErr.RetryAfter > 0 {
                backoff = time.Duration(rateLimitErr.RetryAfter) * time.Second
            }

            select {
            case <-ctx.Done():
                return nil, ctx.Err()
            case <-time.After(backoff):
                backoff *= 2 // Exponential backoff.
                continue
            }
        }

        // Don't retry other errors.
        return nil, err
    }

    return nil, fmt.Errorf("max retries exceeded")
}
```

### User-Friendly Error Messages

```go
func userFriendlyError(err error) string {
    switch {
    case errors.Is(err, anyllm.ErrRateLimit):
        return "Too many requests. Please try again in a moment."
    case errors.Is(err, anyllm.ErrAuthentication):
        return "Authentication failed. Please check your API configuration."
    case errors.Is(err, anyllm.ErrContextLength):
        return "Your message is too long. Please shorten it and try again."
    case errors.Is(err, anyllm.ErrContentFilter):
        return "Your request was blocked by content filters."
    case errors.Is(err, anyllm.ErrModelNotFound):
        return "The requested model is not available."
    default:
        return "An error occurred. Please try again later."
    }
}
```

## Provider-Specific Error Mapping

### OpenAI Errors

| OpenAI Error | any-llm Error |
|--------------|---------------|
| 401 Unauthorized | `ErrAuthentication` |
| 429 Rate Limit | `ErrRateLimit` |
| 400 Invalid Request | `ErrInvalidRequest` |
| 404 Model Not Found | `ErrModelNotFound` |
| Context Length Error | `ErrContextLength` |

### Anthropic Errors

| Anthropic Error | any-llm Error |
|-----------------|---------------|
| Authentication Error | `ErrAuthentication` |
| Rate Limit Error | `ErrRateLimit` |
| Invalid Request | `ErrInvalidRequest` |
| Context Too Long | `ErrContextLength` |

## See Also

- [Completion](completion.md) - Completion API
- [Streaming](streaming.md) - Streaming API
