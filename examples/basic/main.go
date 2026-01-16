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
	_ "github.com/mozilla-ai/any-llm-go/providers/openai" // Register provider
)

func main() {
	ctx := context.Background()

	// Simple completion using the convenience function
	response, err := anyllm.Completion(ctx, "openai:gpt-4o-mini", []anyllm.Message{
		{Role: anyllm.RoleUser, Content: "What is the capital of France? Reply in one word."},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Response: %s\n", response.Choices[0].Message.Content)
	fmt.Printf("Model: %s\n", response.Model)
	fmt.Printf("Tokens used: %d\n", response.Usage.TotalTokens)
}
