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
	"log"

	llm "github.com/mozilla-ai/any-llm-go"
	_ "github.com/mozilla-ai/any-llm-go/providers/anthropic" // Register Anthropic
	_ "github.com/mozilla-ai/any-llm-go/providers/openai"    // Register OpenAI
)

func main() {
	ctx := context.Background()

	// Define models from different providers
	models := []string{
		"openai:gpt-4o-mini",
		"anthropic:claude-3-5-haiku-latest",
	}

	prompt := "What is 2 + 2? Reply with just the number."

	fmt.Printf("Prompt: %s\n\n", prompt)

	// Try each provider
	for _, model := range models {
		fmt.Printf("Model: %s\n", model)

		response, err := llm.Completion(ctx, model, []llm.Message{
			{Role: llm.RoleUser, Content: prompt},
		})
		if err != nil {
			// Handle provider-specific errors gracefully
			if errors.Is(err, llm.ErrMissingAPIKey) {
				fmt.Printf("  Skipped: API key not configured\n\n")
				continue
			}
			log.Printf("  Error: %v\n\n", err)
			continue
		}

		fmt.Printf("  Response: %s\n", response.Choices[0].Message.Content)
		fmt.Printf("  Tokens: %d\n\n", response.Usage.TotalTokens)
	}
}
