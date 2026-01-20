// Package testutil provides testing utilities and fixtures for any-llm.
package testutil

import (
	"os"
	"time"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/providers"
)

// ProviderModelMap maps providers to small, cheap test models.
var ProviderModelMap = map[string]string{
	"openai":     "gpt-4o-mini",
	"anthropic":  "claude-3-5-haiku-latest",
	"mistral":    "mistral-small-latest",
	"gemini":     "gemini-1.5-flash",
	"cohere":     "command-r",
	"groq":       "llama-3.1-8b-instant",
	"ollama":     "llama3.2",
	"together":   "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
	"perplexity": "llama-3.1-sonar-small-128k-online",
	"deepseek":   "deepseek-chat",
	"fireworks":  "accounts/fireworks/models/llama-v3p1-8b-instruct",
	"xai":        "grok-beta",
	"cerebras":   "llama3.1-8b",
	"openrouter": "meta-llama/llama-3.1-8b-instruct",
}

// ProviderReasoningModelMap maps providers to reasoning-capable models.
var ProviderReasoningModelMap = map[string]string{
	"openai":    "o1-mini",
	"anthropic": "claude-sonnet-4-20250514",
	"mistral":   "magistral-small-latest",
	"deepseek":  "deepseek-reasoner",
}

// ProviderImageModelMap maps providers to vision-capable models.
var ProviderImageModelMap = map[string]string{
	"openai":    "gpt-4o-mini",
	"anthropic": "claude-3-5-haiku-latest",
	"gemini":    "gemini-1.5-flash",
}

// EmbeddingProviderModelMap maps providers to embedding models.
var EmbeddingProviderModelMap = map[string]string{
	"openai":   "text-embedding-3-small",
	"cohere":   "embed-english-v3.0",
	"mistral":  "mistral-embed",
	"together": "togethercomputer/m2-bert-80M-8k-retrieval",
}

// ProviderClientConfig holds provider-specific configuration for tests.
var ProviderClientConfig = map[string][]config.Option{
	"anthropic": {config.WithTimeout(60 * time.Second)},
}

// LocalProviders are providers that run locally and don't need API keys.
var LocalProviders = map[string]bool{
	"ollama":    true,
	"lmstudio":  true,
	"llamacpp":  true,
	"llamafile": true,
	"vllm":      true,
}

// providerEnvKeys maps provider names to their API key environment variable names.
var providerEnvKeys = map[string]string{
	"anthropic":  "ANTHROPIC_API_KEY",
	"cerebras":   "CEREBRAS_API_KEY",
	"cohere":     "COHERE_API_KEY",
	"deepseek":   "DEEPSEEK_API_KEY",
	"fireworks":  "FIREWORKS_API_KEY",
	"gemini":     "GEMINI_API_KEY",
	"groq":       "GROQ_API_KEY",
	"mistral":    "MISTRAL_API_KEY",
	"openai":     "OPENAI_API_KEY",
	"openrouter": "OPENROUTER_API_KEY",
	"perplexity": "PERPLEXITY_API_KEY",
	"together":   "TOGETHER_API_KEY",
	"xai":        "XAI_API_KEY",
}

// SimpleMessages returns a simple test message.
func SimpleMessages() []providers.Message {
	return []providers.Message{
		{Role: providers.RoleUser, Content: "Say 'Hello World' exactly, nothing else."},
	}
}

// MessagesWithSystem returns messages with a system prompt.
func MessagesWithSystem() []providers.Message {
	return []providers.Message{
		{Role: providers.RoleSystem, Content: "You are a helpful assistant that follows instructions exactly."},
		{Role: providers.RoleUser, Content: "Say 'Hello World' exactly, nothing else."},
	}
}

// ConversationMessages returns a multi-turn conversation.
func ConversationMessages() []providers.Message {
	return []providers.Message{
		{Role: providers.RoleUser, Content: "My name is Alice."},
		{Role: providers.RoleAssistant, Content: "Hello Alice! Nice to meet you."},
		{Role: providers.RoleUser, Content: "What is my name?"},
	}
}

// ToolCallMessages returns messages for testing tool calls.
func ToolCallMessages() []providers.Message {
	return []providers.Message{
		{Role: providers.RoleUser, Content: "What is the weather in Paris?"},
	}
}

// AgentLoopMessages returns messages for testing agent loops.
func AgentLoopMessages() []providers.Message {
	return []providers.Message{
		{Role: providers.RoleUser, Content: "What is the weather like in Salvaterra?"},
		{
			Role:    providers.RoleAssistant,
			Content: "",
			ToolCalls: []providers.ToolCall{
				{
					ID:   "call_123",
					Type: "function",
					Function: providers.FunctionCall{
						Name:      "get_weather",
						Arguments: `{"location": "Salvaterra"}`,
					},
				},
			},
		},
		{
			Role:       providers.RoleTool,
			Content:    "sunny, 22°C",
			ToolCallID: "call_123",
		},
	}
}

// WeatherTool returns a weather tool definition for testing.
func WeatherTool() providers.Tool {
	return providers.Tool{
		Type: "function",
		Function: providers.Function{
			Name:        "get_weather",
			Description: "Get the current weather for a location.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The city name, e.g. 'Paris, France'",
					},
				},
				"required": []string{"location"},
			},
		},
	}
}

// DateTool returns a date tool definition for testing.
func DateTool() providers.Tool {
	return providers.Tool{
		Type: "function",
		Function: providers.Function{
			Name:        "get_current_date",
			Description: "Get the current date and time.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}
}

// HasAPIKey checks if the API key environment variable is set for a provider.
func HasAPIKey(provider string) bool {
	if LocalProviders[provider] {
		return true
	}

	envKey, ok := providerEnvKeys[provider]
	if !ok {
		return false
	}
	return os.Getenv(envKey) != ""
}

// SkipIfNoAPIKey skips the test if the API key is not set.
// Returns true if the test should be skipped.
func SkipIfNoAPIKey(provider string) bool {
	return !HasAPIKey(provider)
}

// GetTestModel returns the test model for a provider.
func GetTestModel(provider string) string {
	if model, ok := ProviderModelMap[provider]; ok {
		return model
	}
	return ""
}

// GetReasoningModel returns the reasoning model for a provider.
func GetReasoningModel(provider string) string {
	if model, ok := ProviderReasoningModelMap[provider]; ok {
		return model
	}
	return ""
}

// GetEmbeddingModel returns the embedding model for a provider.
func GetEmbeddingModel(provider string) string {
	if model, ok := EmbeddingProviderModelMap[provider]; ok {
		return model
	}
	return ""
}

// GetClientOptions returns the client options for a provider.
func GetClientOptions(provider string) []config.Option {
	if opts, ok := ProviderClientConfig[provider]; ok {
		return opts
	}
	return nil
}
