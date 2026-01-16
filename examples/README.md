# Examples

This directory contains example code demonstrating various features of any-llm-go.

## Prerequisites

Set up your API keys:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Examples

### Basic Completion

The simplest way to make a completion request.

```bash
cd basic
go run main.go
```

### Streaming

Receive responses in real-time as they're generated.

```bash
cd streaming
go run main.go
```

### Tool Calling

Use function calling to give the model access to external tools.

```bash
cd tools
go run main.go
```

### Multi-Provider

Use multiple providers with the same code.

```bash
cd multi-provider
go run main.go
```

## Running All Examples

From the examples directory:

```bash
for dir in */; do
    echo "=== Running $dir ==="
    (cd "$dir" && go run main.go)
    echo
done
```
