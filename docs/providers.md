# Supported Providers

any-llm-go supports multiple LLM providers through a unified interface. Each provider is implemented as a separate package.

## Provider Status

| Provider                | ID          | Completion | Streaming | Tools | Reasoning | Embeddings | List Models |
|-------------------------|:------------|:----------:|:---------:|:-----:|:---------:|:----------:|:-----------:|
| [Anthropic](#anthropic) | `anthropic` |     ✅      |     ✅     |   ✅   |     ✅     |     ❌      |      ❌      |
| [DeepSeek](#deepseek)   | `deepseek`  |     ✅      |     ✅     |   ✅   |     ✅     |     ❌      |      ✅      |
| [Gemini](#gemini)       | `gemini`    |     ✅      |     ✅     |   ✅   |     ✅     |     ✅      |      ✅      |
| [Llamafile](#llamafile) | `llamafile` |     ✅      |     ✅     |   ✅   |     ❌     |     ✅      |      ✅      |
| [Ollama](#ollama)       | `ollama`    |     ✅      |     ✅     |   ✅   |     ✅     |     ✅      |      ✅      |
| [OpenAI](#openai)       | `openai`    |     ✅      |     ✅     |   ✅   |     ✅     |     ✅      |      ✅      |

### Legend

- **Completion** - Basic chat completion support
- **Streaming** - Real-time streaming responses
- **Tools** - Function calling / tool use
- **Reasoning** - Extended thinking (e.g., Claude's thinking, OpenAI o1 reasoning)
- **Embeddings** - Text embedding generation
- **List Models** - API to list available models

## Provider Details

### Anthropic

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/anthropic"
)

// Using environment variable (ANTHROPIC_API_KEY).
provider, err := anthropic.New()

// Or with explicit API key.
provider, err := anthropic.New(anyllm.WithAPIKey("sk-ant-..."))
```

**Environment Variable:** `ANTHROPIC_API_KEY`

**Popular Models:**
- `claude-sonnet-4-20250514` - Latest Sonnet model
- `claude-3-5-sonnet-latest` - Previous Sonnet
- `claude-3-5-haiku-latest` - Fast and cost-effective
- `claude-3-opus-latest` - Most capable (legacy)

**Extended Thinking:**

Anthropic's Claude models support extended thinking for complex reasoning tasks:

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "claude-sonnet-4-20250514",
    Messages: messages,
    ReasoningEffort: anyllm.ReasoningEffortMedium, // low, medium, or high
})

// Access the thinking content.
if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

### DeepSeek

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/deepseek"
)

// Using environment variable (DEEPSEEK_API_KEY).
provider, err := deepseek.New()

// Or with explicit API key.
provider, err := deepseek.New(anyllm.WithAPIKey("sk-..."))
```

**Environment Variable:** `DEEPSEEK_API_KEY`

**Popular Models:**
- `deepseek-chat` - General-purpose chat model
- `deepseek-reasoner` - Reasoning model (DeepSeek R1)

**Reasoning/Thinking:**

DeepSeek R1 supports extended thinking for complex reasoning tasks:

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "deepseek-reasoner",
    Messages: messages,
    ReasoningEffort: anyllm.ReasoningEffortMedium,
})

if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

**JSON Schema:**

DeepSeek doesn't support `json_schema` response format directly. The provider automatically handles this by injecting the schema into the user message and using `json_object` mode instead.

### Gemini

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/gemini"
)

// Using environment variable (GEMINI_API_KEY or GOOGLE_API_KEY).
provider, err := gemini.New()

// Or with explicit API key.
provider, err := gemini.New(anyllm.WithAPIKey("your-key"))
```

**Environment Variables:** `GEMINI_API_KEY` or `GOOGLE_API_KEY`

**Popular Models:**
- `gemini-2.5-flash` - Fast and cost-effective
- `gemini-2.5-pro` - Most capable model
- `gemini-3-flash-preview` - Reasoning-capable model

**Embedding Models:**
- `gemini-embedding-001` - Text embeddings

**Reasoning/Thinking:**

Gemini models support extended thinking for complex reasoning tasks:

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "gemini-3-flash-preview",
    Messages: messages,
    ReasoningEffort: anyllm.ReasoningEffortMedium, // low, medium, or high
})

// Access the thinking content.
if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

### Llamafile

Llamafile is a single-file executable that bundles a model with llama.cpp for easy local deployment. It exposes an OpenAI-compatible API. No API key is required.

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/llamafile"
)

// Using default settings (localhost:8080/v1).
provider, err := llamafile.New()

// Or with custom base URL.
provider, err := llamafile.New(anyllm.WithBaseURL("http://localhost:8081/v1"))
```

**Environment Variable:** `LLAMAFILE_BASE_URL` (optional, defaults to `http://localhost:8080/v1`)

**Running Llamafile:**

Download a llamafile from [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) and run it:

```bash
# Download a llamafile (example: LLaVA)
curl -LO https://huggingface.co/Mozilla/llava-v1.5-7b-llamafile/resolve/main/llava-v1.5-7b-q4.llamafile
chmod +x llava-v1.5-7b-q4.llamafile
./llava-v1.5-7b-q4.llamafile --server
```

**Completion:**

```go
provider, _ := llamafile.New()
resp, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "LLaMA_CPP", // Llamafile uses "LLaMA_CPP" as the model name.
    Messages: []anyllm.Message{
        {Role: anyllm.RoleUser, Content: "Hello!"},
    },
})
```

**Streaming:**

```go
provider, _ := llamafile.New()
chunks, errs := provider.CompletionStream(ctx, anyllm.CompletionParams{
    Model: "LLaMA_CPP",
    Messages: messages,
})

for chunk := range chunks {
    fmt.Print(chunk.Choices[0].Delta.Content)
}
if err := <-errs; err != nil {
    log.Fatal(err)
}
```

**List Models:**

```go
provider, _ := llamafile.New()
models, err := provider.ListModels(ctx)
for _, model := range models.Data {
    fmt.Println(model.ID) // Typically "LLaMA_CPP"
}
```

### Ollama

Ollama is a local LLM server that allows you to run models on your own hardware. No API key is required.

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/ollama"
)

// Using default settings (localhost:11434).
provider, err := ollama.New()

// Or with custom base URL.
provider, err := ollama.New(anyllm.WithBaseURL("http://localhost:11435"))
```

**Environment Variable:** `OLLAMA_HOST` (optional, defaults to `http://localhost:11434`)

**Popular Models:**
- `llama3.2` - Meta's Llama 3.2
- `mistral` - Mistral 7B
- `codellama` - Code-focused Llama
- `deepseek-r1` - DeepSeek reasoning model

**Reasoning/Thinking:**

Ollama supports extended thinking for models that support it:

```go
response, err := provider.Completion(ctx, anyllm.CompletionParams{
    Model: "deepseek-r1",
    Messages: messages,
    ReasoningEffort: anyllm.ReasoningEffortMedium,
})

if response.Choices[0].Message.Reasoning != nil {
    fmt.Println("Thinking:", response.Choices[0].Message.Reasoning.Content)
}
```

**Embeddings:**

```go
provider, _ := ollama.New()
resp, err := provider.Embedding(ctx, anyllm.EmbeddingParams{
    Model: "nomic-embed-text",
    Input: "Hello, world!",
})
```

**List Models:**

```go
provider, _ := ollama.New()
models, err := provider.ListModels(ctx)
for _, model := range models.Data {
    fmt.Println(model.ID)
}
```

### OpenAI

```go
import (
    anyllm "github.com/mozilla-ai/any-llm-go"
    "github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Using environment variable (OPENAI_API_KEY).
provider, err := openai.New()

// Or with explicit API key.
provider, err := openai.New(anyllm.WithAPIKey("sk-..."))

// Or with custom base URL (for Azure, proxies, etc.).
provider, err := openai.New(
    anyllm.WithAPIKey("your-key"),
    anyllm.WithBaseURL("https://your-endpoint.openai.azure.com"),
)
```

**Environment Variable:** `OPENAI_API_KEY`

**Popular Models:**
- `gpt-4o` - Most capable model
- `gpt-4o-mini` - Fast and cost-effective
- `gpt-4-turbo` - Previous generation flagship
- `o1-preview` - Reasoning model
- `o1-mini` - Smaller reasoning model

**Embedding Models:**
- `text-embedding-3-small` - Cost-effective embeddings
- `text-embedding-3-large` - Higher quality embeddings

## Coming Soon

The following providers are planned for future releases:

| Provider     | Status                                            |
|--------------|---------------------------------------------------|
| Mistral      | Planned                                           |
| Groq         | Planned                                           |
| Cohere       | Planned                                           |
| Together AI  | Planned                                           |
| AWS Bedrock  | Planned                                           |
| Azure OpenAI | Planned (use OpenAI with custom base URL for now) |

## Adding a New Provider

Want to add support for a new provider? See our [Contributing Guide](../CONTRIBUTING.md) for instructions on implementing a new provider.

The basic requirements are:

1. Implement the `Provider` interface
2. Use the official provider SDK when available
3. Normalize responses to OpenAI format
4. Add comprehensive tests
5. Document the provider in this file

## Provider-Specific Notes

### Response Format

All providers normalize their responses to OpenAI's format:

```go
type ChatCompletion struct {
    ID      string   `json:"id"`
    Object  string   `json:"object"`
    Created int64    `json:"created"`
    Model   string   `json:"model"`
    Choices []Choice `json:"choices"`
    Usage   *Usage   `json:"usage,omitempty"`
}
```

This means you can write provider-agnostic code that works with any supported provider.

### Error Handling

Provider-specific errors are normalized to common error types:

| Error Type | Description |
|------------|-------------|
| `ErrRateLimit` | Rate limit exceeded |
| `ErrAuthentication` | Invalid or missing API key |
| `ErrInvalidRequest` | Malformed request |
| `ErrContextLength` | Input exceeds model's context window |
| `ErrContentFilter` | Content blocked by safety filters |
| `ErrModelNotFound` | Requested model doesn't exist |

See [Error Handling](api/errors.md) for more details.
