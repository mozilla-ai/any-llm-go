// Example: Multi-provider usage
//
// This example demonstrates how to use multiple providers with the same code.
//
// Run with:
//
//	export OPENAI_API_KEY="sk-..."
//	export ANTHROPIC_API_KEY="sk-ant-..."
//	go run main.go
package main

import (
	"context"
	"errors"
	"fmt"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/providers/anthropic"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

func main() {
	ctx := context.Background()

	prompt := "What is 2 + 2? Reply with just the number."
	fmt.Printf("Prompt: %s\n\n", prompt)

	// Try OpenAI.
	fmt.Println("OpenAI:")
	if err := tryProvider(ctx, "openai", "gpt-4o-mini", prompt); err != nil {
		fmt.Printf("  Error: %v\n\n", err)
	}

	// Try Anthropic.
	fmt.Println("Anthropic:")
	if err := tryProvider(ctx, "anthropic", "claude-3-5-haiku-latest", prompt); err != nil {
		fmt.Printf("  Error: %v\n\n", err)
	}
}

func tryProvider(ctx context.Context, providerName, model, prompt string) error {
	var provider anyllm.Provider
	var err error

	switch providerName {
	case "openai":
		provider, err = openai.New()
	case "anthropic":
		provider, err = anthropic.New()
	default:
		return fmt.Errorf("unknown provider: %s", providerName)
	}

	if err != nil {
		if errors.Is(err, anyllm.ErrMissingAPIKey) {
			fmt.Printf("  Skipped: API key not configured\n\n")
			return nil
		}
		return err
	}

	response, err := provider.Completion(ctx, anyllm.CompletionParams{
		Model: model,
		Messages: []anyllm.Message{
			{Role: anyllm.RoleUser, Content: prompt},
		},
	})
	if err != nil {
		return err
	}

	fmt.Printf("  Response: %s\n", response.Choices[0].Message.Content)
	fmt.Printf("  Tokens: %d\n\n", response.Usage.TotalTokens)
	return nil
}
