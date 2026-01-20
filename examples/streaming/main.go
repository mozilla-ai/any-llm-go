// Example: Streaming responses
//
// This example demonstrates how to receive streaming responses.
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
	// Create a provider instance for better performance with multiple requests.
	provider, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Request a streaming completion.
	chunks, errs := provider.CompletionStream(ctx, anyllm.CompletionParams{
		Model: "gpt-4o-mini",
		Messages: []anyllm.Message{
			{Role: anyllm.RoleUser, Content: "Write a short poem about programming in Go."},
		},
		Stream: true,
	})

	fmt.Println("Streaming response:")
	fmt.Println("---")

	// Process chunks as they arrive.
	for chunk := range chunks {
		if len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			if content != "" {
				fmt.Print(content)
			}
		}
	}

	fmt.Println("\n---")

	// Always check for errors after the stream completes.
	if err := <-errs; err != nil {
		log.Fatal(err)
	}

	fmt.Println("Stream completed successfully!")
}
