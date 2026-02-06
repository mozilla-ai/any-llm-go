// Example: Tool/Function calling with Llamafile
//
// This example demonstrates how to use real tools (function calling) with Llamafile,
// a local LLM server that requires no API key.
//
// The tools in this example are real implementations (not mocks):
//   - get_current_datetime: Returns the actual current date and time
//   - calculate: Performs real mathematical calculations
//
// Prerequisites:
//  1. Download a llamafile from https://github.com/Mozilla-Ocho/llamafile
//  2. Run it: ./your-model.llamafile --server --port 8080
//
// Run with:
//
//	go run main.go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	anyllm "github.com/mozilla-ai/any-llm-go"
	"github.com/mozilla-ai/any-llm-go/providers/llamafile"
)

// Define real tools that the model can call.
var tools = []anyllm.Tool{
	{
		Type: "function",
		Function: anyllm.Function{
			Name:        "get_current_datetime",
			Description: "Get the current date and time. Use this when the user asks about today's date, current time, or day of the week.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"timezone": map[string]any{
						"type":        "string",
						"description": "Optional timezone (e.g., 'America/New_York', 'Europe/London'). Defaults to local timezone.",
					},
				},
			},
		},
	},
	{
		Type: "function",
		Function: anyllm.Function{
			Name:        "calculate",
			Description: "Perform mathematical calculations. Use this for any math operations like addition, subtraction, multiplication, division, or more complex expressions.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"operation": map[string]any{
						"type":        "string",
						"enum":        []string{"add", "subtract", "multiply", "divide"},
						"description": "The mathematical operation to perform",
					},
					"a": map[string]any{
						"type":        "number",
						"description": "The first number",
					},
					"b": map[string]any{
						"type":        "number",
						"description": "The second number",
					},
				},
				"required": []string{"operation", "a", "b"},
			},
		},
	},
}

// getCurrentDatetime returns the actual current date and time.
func getCurrentDatetime(timezone string) string {
	loc := time.Local
	if timezone != "" {
		if parsedLoc, err := time.LoadLocation(timezone); err == nil {
			loc = parsedLoc
		}
	}

	now := time.Now().In(loc)
	return fmt.Sprintf(`{"date": "%s", "time": "%s", "day_of_week": "%s", "timezone": "%s", "unix_timestamp": %d}`,
		now.Format("2006-01-02"),
		now.Format("15:04:05"),
		now.Weekday().String(),
		loc.String(),
		now.Unix(),
	)
}

// calculate performs real mathematical operations.
func calculate(operation string, a, b float64) string {
	var result float64

	switch operation {
	case "add":
		result = a + b
	case "subtract":
		result = a - b
	case "multiply":
		result = a * b
	case "divide":
		if b == 0 {
			return `{"error": "division by zero"}`
		}
		result = a / b
	default:
		return fmt.Sprintf(`{"error": "unknown operation: %s"}`, operation)
	}

	return fmt.Sprintf(`{"operation": "%s", "a": %g, "b": %g, "result": %g}`, operation, a, b, result)
}

// executeTool runs the appropriate tool based on the function name.
func executeTool(name, arguments string) (string, error) {
	switch name {
	case "get_current_datetime":
		var args struct {
			Timezone string `json:"timezone"`
		}
		if err := json.Unmarshal([]byte(arguments), &args); err != nil {
			return "", err
		}
		return getCurrentDatetime(args.Timezone), nil

	case "calculate":
		var args struct {
			Operation string  `json:"operation"`
			A         float64 `json:"a"`
			B         float64 `json:"b"`
		}
		if err := json.Unmarshal([]byte(arguments), &args); err != nil {
			return "", err
		}
		return calculate(args.Operation, args.A, args.B), nil

	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}

func main() {
	// Create a Llamafile provider. By default, it connects to http://localhost:8080/v1
	// You can customize with: llamafile.New(config.WithBaseURL("http://localhost:9000/v1"))
	provider, err := llamafile.New()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// List available models to verify connection.
	models, err := provider.ListModels(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to Llamafile server: %v\nMake sure Llamafile is running on port 8080", err)
	}
	fmt.Printf("Connected to Llamafile. Available models: %d\n", len(models.Data))
	fmt.Println()

	// Use the model name from the server, or fallback to a default.
	modelName := "LLaMA_CPP"
	if len(models.Data) > 0 {
		modelName = models.Data[0].ID
	}

	// Try different prompts that use real tools.
	prompts := []string{
		"What is today's date and what day of the week is it?",
		"What is 1547 multiplied by 382?",
	}

	for _, prompt := range prompts {
		fmt.Printf("User: %s\n\n", prompt)

		messages := []anyllm.Message{
			{Role: anyllm.RoleUser, Content: prompt},
		}

		// First request - model may call a tool.
		response, err := provider.Completion(ctx, anyllm.CompletionParams{
			Model:      modelName,
			Messages:   messages,
			Tools:      tools,
			ToolChoice: "auto",
		})
		if err != nil {
			log.Fatal(err)
		}

		// Check if the model wants to call a tool.
		if response.Choices[0].FinishReason == anyllm.FinishReasonToolCalls {
			fmt.Println("Model is calling tools...")

			// Add the assistant's message (with tool calls) to the conversation.
			messages = append(messages, response.Choices[0].Message)

			// Process each tool call.
			for _, tc := range response.Choices[0].Message.ToolCalls {
				fmt.Printf("  Tool: %s\n", tc.Function.Name)
				fmt.Printf("  Arguments: %s\n", tc.Function.Arguments)

				// Execute the real tool.
				result, execErr := executeTool(tc.Function.Name, tc.Function.Arguments)
				if execErr != nil {
					log.Fatal(execErr)
				}
				fmt.Printf("  Result: %s\n\n", result)

				// Add the tool result to the conversation.
				messages = append(messages, anyllm.Message{
					Role:       anyllm.RoleTool,
					Content:    result,
					ToolCallID: tc.ID,
				})
			}

			// Continue the conversation with the tool results.
			response, err = provider.Completion(ctx, anyllm.CompletionParams{
				Model:    modelName,
				Messages: messages,
				Tools:    tools,
			})
			if err != nil {
				log.Fatal(err)
			}
		}

		// Print the final response.
		fmt.Printf("Assistant: %s\n", response.Choices[0].Message.Content)
		fmt.Println()
		fmt.Println("---")
		fmt.Println()
	}
}
