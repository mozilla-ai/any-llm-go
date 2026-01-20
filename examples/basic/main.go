// Example: Basic completion request
//
// This example demonstrates the simplest way to use any-llm-go.
//
// Run with:
//
//	export OPENAI_API_KEY="sk-..."
//	go run main.go
package main

import (
	"context"
	"fmt"
	"log"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

func main() {
	ctx := context.Background()

	// Create a provider directly.
	provider, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}

	// Make a completion request.
	response, err := provider.Completion(ctx, anyllm.CompletionParams{
		Model: "gpt-4o-mini",
		Messages: []anyllm.Message{
			{Role: anyllm.RoleUser, Content: "What is the capital of France? Reply in one word."},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Response: %s\n", response.Choices[0].Message.Content)
	fmt.Printf("Model: %s\n", response.Model)
	fmt.Printf("Tokens used: %d\n", response.Usage.TotalTokens)
}
