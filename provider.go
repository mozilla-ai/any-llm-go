package anyllm

import (
	"context"
)

// Provider is the core interface that all LLM providers must implement.
type Provider interface {
	// Name returns the provider's identifier (e.g., "openai", "anthropic").
	Name() string

	// Completion performs a chat completion request.
	Completion(ctx context.Context, params CompletionParams) (*ChatCompletion, error)

	// CompletionStream performs a streaming chat completion request.
	// Returns a channel of chunks and a channel for any error that occurs.
	// The error channel will receive at most one error and then be closed.
	// Both channels are closed when the stream ends.
	CompletionStream(ctx context.Context, params CompletionParams) (<-chan ChatCompletionChunk, <-chan error)
}

// EmbeddingProvider is an optional interface for providers that support embeddings.
type EmbeddingProvider interface {
	Provider
	Embedding(ctx context.Context, params EmbeddingParams) (*EmbeddingResponse, error)
}

// ModelLister is an optional interface for providers that support listing models.
type ModelLister interface {
	Provider
	ListModels(ctx context.Context) (*ModelsResponse, error)
}

// ProviderCapabilities describes what features a provider supports.
type ProviderCapabilities struct {
	Completion          bool
	CompletionStreaming bool
	CompletionReasoning bool
	CompletionImage     bool
	CompletionPDF       bool
	Embedding           bool
	ListModels          bool
}

// CapabilityProvider is an optional interface for providers to report capabilities.
type CapabilityProvider interface {
	Provider
	Capabilities() ProviderCapabilities
}

// SupportsStreaming returns true if the provider supports streaming.
func SupportsStreaming(p Provider) bool {
	if cp, ok := p.(CapabilityProvider); ok {
		return cp.Capabilities().CompletionStreaming
	}
	// Default to true if not explicitly specified
	return true
}

// SupportsEmbedding returns true if the provider supports embeddings.
func SupportsEmbedding(p Provider) bool {
	_, ok := p.(EmbeddingProvider)
	return ok
}

// SupportsListModels returns true if the provider supports listing models.
func SupportsListModels(p Provider) bool {
	_, ok := p.(ModelLister)
	return ok
}

// SupportsReasoning returns true if the provider supports extended thinking.
func SupportsReasoning(p Provider) bool {
	if cp, ok := p.(CapabilityProvider); ok {
		return cp.Capabilities().CompletionReasoning
	}
	return false
}

// SupportsImages returns true if the provider supports image inputs.
func SupportsImages(p Provider) bool {
	if cp, ok := p.(CapabilityProvider); ok {
		return cp.Capabilities().CompletionImage
	}
	return false
}

// LLMProvider represents the available LLM providers.
type LLMProvider string

const (
	ProviderOpenAI      LLMProvider = "openai"
	ProviderAnthropic   LLMProvider = "anthropic"
	ProviderMistral     LLMProvider = "mistral"
	ProviderGemini      LLMProvider = "gemini"
	ProviderCohere      LLMProvider = "cohere"
	ProviderGroq        LLMProvider = "groq"
	ProviderOllama      LLMProvider = "ollama"
	ProviderTogether    LLMProvider = "together"
	ProviderPerplexity  LLMProvider = "perplexity"
	ProviderDeepseek    LLMProvider = "deepseek"
	ProviderFireworks   LLMProvider = "fireworks"
	ProviderAzureOpenAI LLMProvider = "azureopenai"
	ProviderBedrock     LLMProvider = "bedrock"
	ProviderVertexAI    LLMProvider = "vertexai"
	ProviderXAI         LLMProvider = "xai"
	ProviderCerebras    LLMProvider = "cerebras"
	ProviderSambanova   LLMProvider = "sambanova"
	ProviderOpenRouter  LLMProvider = "openrouter"
	ProviderLMStudio    LLMProvider = "lmstudio"
	ProviderLlamaCPP    LLMProvider = "llamacpp"
	ProviderVLLM        LLMProvider = "vllm"
)

// String returns the string representation of the provider.
func (p LLMProvider) String() string {
	return string(p)
}

// ProviderEnvKeyName returns the environment variable name for the provider's API key.
func ProviderEnvKeyName(p LLMProvider) string {
	envKeys := map[LLMProvider]string{
		ProviderOpenAI:      "OPENAI_API_KEY",
		ProviderAnthropic:   "ANTHROPIC_API_KEY",
		ProviderMistral:     "MISTRAL_API_KEY",
		ProviderGemini:      "GEMINI_API_KEY",
		ProviderCohere:      "COHERE_API_KEY",
		ProviderGroq:        "GROQ_API_KEY",
		ProviderOllama:      "", // No API key needed
		ProviderTogether:    "TOGETHER_API_KEY",
		ProviderPerplexity:  "PERPLEXITY_API_KEY",
		ProviderDeepseek:    "DEEPSEEK_API_KEY",
		ProviderFireworks:   "FIREWORKS_API_KEY",
		ProviderAzureOpenAI: "AZURE_OPENAI_API_KEY",
		ProviderBedrock:     "", // Uses AWS credentials
		ProviderVertexAI:    "", // Uses Google credentials
		ProviderXAI:         "XAI_API_KEY",
		ProviderCerebras:    "CEREBRAS_API_KEY",
		ProviderSambanova:   "SAMBANOVA_API_KEY",
		ProviderOpenRouter:  "OPENROUTER_API_KEY",
		ProviderLMStudio:    "", // Local
		ProviderLlamaCPP:    "", // Local
		ProviderVLLM:        "", // Local
	}
	return envKeys[p]
}
