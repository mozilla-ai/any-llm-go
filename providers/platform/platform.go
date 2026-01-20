// Package platform provides a platform provider implementation for any-llm.
// It acts as a proxy that authenticates with the ANY LLM platform to get
// provider API keys, then delegates calls to the underlying provider.
package platform

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	anyllmplatform "github.com/mozilla-ai/any-llm-platform-client-go"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/anthropic"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
)

const (
	providerName       = "platform"
	envAPIKey          = "ANY_LLM_KEY"
	envPlatformURL     = "ANY_LLM_PLATFORM_URL"
	defaultPlatformURL = "https://platform-api.any-llm.ai/api/v1"
)

// newProviderFunc creates a provider with the given options.
type newProviderFunc func(opts ...config.Option) (providers.Provider, error)

// supportedProviders maps provider names to their constructors.
var supportedProviders = map[string]newProviderFunc{
	"openai": func(opts ...config.Option) (providers.Provider, error) {
		return openai.New(opts...)
	},
	"anthropic": func(opts ...config.Option) (providers.Provider, error) {
		return anthropic.New(opts...)
	},
}

// Provider implements the providers.Provider interface for the ANY LLM platform.
// It proxies requests to underlying providers (OpenAI, Anthropic, etc.) after
// authenticating with the platform to get decrypted API keys.
type Provider struct {
	config         *config.Config
	platformClient *anyllmplatform.Client
	httpClient     *http.Client
	anyLLMKey      string
	clientName     string

	// Cached provider information.
	providerKeyID string
	projectID     string

	// The underlying provider that handles actual LLM calls.
	underlyingProvider providers.Provider
	underlyingName     string
}

// Ensure Provider implements the required interfaces.
var (
	_ providers.Provider           = (*Provider)(nil)
	_ providers.CapabilityProvider = (*Provider)(nil)
)

// New creates a new platform provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg := config.New()
	cfg.ApplyOptions(opts...)

	anyLLMKey := cfg.GetAPIKeyFromEnv(envAPIKey)
	if anyLLMKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	platformURL := os.Getenv(envPlatformURL)
	if platformURL == "" {
		platformURL = defaultPlatformURL
	}

	platformClient := anyllmplatform.NewClient(
		anyllmplatform.WithPlatformURL(platformURL),
	)

	return &Provider{
		config:         cfg,
		platformClient: platformClient,
		httpClient:     &http.Client{Timeout: 30 * time.Second},
		anyLLMKey:      anyLLMKey,
	}, nil
}

// WithClientName sets a client name for per-client usage tracking.
func WithClientName(name string) config.Option {
	return func(cfg *config.Config) {
		cfg.Extra["client_name"] = name
	}
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// Capabilities returns the provider's capabilities.
// Since this is a proxy, capabilities depend on the underlying provider.
func (p *Provider) Capabilities() providers.Capabilities {
	// Return full capabilities since we can proxy to any provider.
	return providers.Capabilities{
		Completion:          true,
		CompletionStreaming: true,
		CompletionReasoning: true,
		CompletionImage:     true,
		CompletionPDF:       true,
		Embedding:           true,
		ListModels:          true,
	}
}

// initializeProvider initializes the underlying provider for the given provider name.
func (p *Provider) initializeProvider(ctx context.Context, providerName string) error {
	if p.underlyingProvider != nil && p.underlyingName == providerName {
		return nil // Already initialized for this provider
	}

	// Get decrypted provider key from the platform
	result, err := p.platformClient.GetDecryptedProviderKey(ctx, p.anyLLMKey, providerName)
	if err != nil {
		return fmt.Errorf("failed to get provider key: %w", err)
	}

	p.providerKeyID = result.ProviderKeyID.String()
	p.projectID = result.ProjectID.String()

	// Create the underlying provider using the decrypted API key.
	constructor, ok := supportedProviders[strings.ToLower(providerName)]
	if !ok {
		return fmt.Errorf("unsupported provider: %s", providerName)
	}
	provider, err := constructor(config.WithAPIKey(result.APIKey))
	if err != nil {
		return fmt.Errorf("failed to create provider %q: %w", providerName, err)
	}

	p.underlyingProvider = provider
	p.underlyingName = providerName
	return nil
}

// parseModelString parses a model string in the format "provider:model" or just "model".
func parseModelString(model string) (providerName, modelID string) {
	parts := strings.SplitN(model, ":", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return "", model
}

// Completion performs a chat completion request.
func (p *Provider) Completion(ctx context.Context, params providers.CompletionParams) (*providers.ChatCompletion, error) {
	startTime := time.Now()

	// Parse the model to get the provider name
	providerName, modelID := parseModelString(params.Model)
	if providerName == "" {
		return nil, fmt.Errorf("model must be in format 'provider:model', got %q", params.Model)
	}

	// Initialize the underlying provider
	if err := p.initializeProvider(ctx, providerName); err != nil {
		return nil, err
	}

	// Update params with just the model ID (without provider prefix)
	params.Model = modelID

	// Delegate to the underlying provider
	completion, err := p.underlyingProvider.Completion(ctx, params)
	if err != nil {
		return nil, err
	}

	// Post usage event
	totalDurationMs := float64(time.Since(startTime).Milliseconds())
	go p.postUsageEvent(context.Background(), completion, nil, totalDurationMs)

	return completion, nil
}

// CompletionStream performs a streaming chat completion request.
func (p *Provider) CompletionStream(ctx context.Context, params providers.CompletionParams) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		startTime := time.Now()

		// Parse the model to get the provider name
		providerName, modelID := parseModelString(params.Model)
		if providerName == "" {
			errs <- fmt.Errorf("model must be in format 'provider:model', got %q", params.Model)
			return
		}

		// Initialize the underlying provider
		if err := p.initializeProvider(ctx, providerName); err != nil {
			errs <- err
			return
		}

		// Update params with just the model ID
		params.Model = modelID

		// Ensure we get usage data in the streaming response for tracking.
		if params.StreamOptions == nil {
			params.StreamOptions = &providers.StreamOptions{IncludeUsage: true}
		} else if !params.StreamOptions.IncludeUsage {
			params.StreamOptions.IncludeUsage = true
		}

		// Get the stream from the underlying provider
		upstreamChunks, upstreamErrs := p.underlyingProvider.CompletionStream(ctx, params)

		// Track streaming metrics.
		var (
			collectedChunks     []providers.ChatCompletionChunk
			timeToFirstTokenMs  *float64
			timeToLastContentMs *float64
			previousChunkTime   *time.Time
			chunkLatencies      []float64
		)

		// Forward chunks and collect for usage tracking
		for chunk := range upstreamChunks {
			currentTime := time.Now()

			// Track time to first token
			if timeToFirstTokenMs == nil && len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				ms := float64(currentTime.Sub(startTime).Milliseconds())
				timeToFirstTokenMs = &ms
			}

			// Track time to last content token
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				ms := float64(currentTime.Sub(startTime).Milliseconds())
				timeToLastContentMs = &ms
			}

			// Track inter-chunk latency
			if previousChunkTime != nil {
				latencyMs := float64(currentTime.Sub(*previousChunkTime).Milliseconds())
				chunkLatencies = append(chunkLatencies, latencyMs)
			}
			previousChunkTime = &currentTime

			collectedChunks = append(collectedChunks, chunk)
			chunks <- chunk
		}

		// Check for upstream errors
		if err := <-upstreamErrs; err != nil {
			errs <- err
			return
		}

		// Post usage event with streaming metrics
		if len(collectedChunks) > 0 {
			totalDurationMs := float64(time.Since(startTime).Milliseconds())
			completion := combineChunks(collectedChunks)

			metrics := &streamingMetrics{
				TimeToFirstTokenMs: timeToFirstTokenMs,
				TimeToLastTokenMs:  timeToLastContentMs,
				TotalDurationMs:    totalDurationMs,
				ChunksReceived:     len(collectedChunks),
			}

			// Calculate tokens per second if we have usage data
			if completion.Usage != nil && completion.Usage.CompletionTokens > 0 && timeToLastContentMs != nil && *timeToLastContentMs > 0 {
				tps := float64(completion.Usage.CompletionTokens*1000) / *timeToLastContentMs
				metrics.TokensPerSecond = &tps

				avgChunkSize := float64(completion.Usage.CompletionTokens) / float64(len(collectedChunks))
				metrics.AvgChunkSize = &avgChunkSize
			}

			// Calculate inter-chunk latency variance if we have enough data points
			if len(chunkLatencies) > 1 {
				variance := calculateVariance(chunkLatencies)
				metrics.InterChunkLatencyVarianceMs = &variance
			}

			go p.postUsageEvent(context.Background(), completion, metrics, totalDurationMs)
		}
	}()

	return chunks, errs
}

// streamingMetrics holds performance metrics for streaming requests.
type streamingMetrics struct {
	TimeToFirstTokenMs          *float64
	TimeToLastTokenMs           *float64
	TotalDurationMs             float64
	TokensPerSecond             *float64
	ChunksReceived              int
	AvgChunkSize                *float64
	InterChunkLatencyVarianceMs *float64
}

// calculateVariance calculates the sample variance of a slice of float64 values.
func calculateVariance(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}

	// Calculate mean
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	// Calculate sum of squared differences
	var sumSquaredDiff float64
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}

	// Return sample variance (n-1)
	return sumSquaredDiff / float64(len(values)-1)
}

// combineChunks combines streaming chunks into a ChatCompletion for usage tracking.
func combineChunks(chunks []providers.ChatCompletionChunk) *providers.ChatCompletion {
	if len(chunks) == 0 {
		return nil
	}

	lastChunk := chunks[len(chunks)-1]

	return &providers.ChatCompletion{
		ID:      lastChunk.ID,
		Object:  "chat.completion",
		Created: lastChunk.Created,
		Model:   lastChunk.Model,
		Choices: []providers.Choice{},
		Usage:   lastChunk.Usage,
	}
}

// usageEventPayload represents the payload for usage events.
type usageEventPayload struct {
	ProviderKeyID string                 `json:"provider_key_id"`
	Provider      string                 `json:"provider"`
	Model         string                 `json:"model"`
	Data          map[string]interface{} `json:"data"`
	ID            string                 `json:"id"`
	ClientName    string                 `json:"client_name,omitempty"`
}

// postUsageEvent posts a usage event to the platform.
func (p *Provider) postUsageEvent(ctx context.Context, completion *providers.ChatCompletion, metrics *streamingMetrics, totalDurationMs float64) {
	if completion == nil || completion.Usage == nil {
		return
	}

	// Get access token for Bearer authentication
	accessToken, err := p.platformClient.GetAccessToken(ctx, p.anyLLMKey)
	if err != nil {
		return
	}

	// Build data payload
	data := map[string]interface{}{
		"input_tokens":  fmt.Sprintf("%d", completion.Usage.PromptTokens),
		"output_tokens": fmt.Sprintf("%d", completion.Usage.CompletionTokens),
	}

	// Add performance metrics
	performance := map[string]interface{}{}
	if totalDurationMs > 0 {
		performance["total_duration_ms"] = totalDurationMs
	}
	if metrics != nil {
		if metrics.TimeToFirstTokenMs != nil {
			performance["time_to_first_token_ms"] = *metrics.TimeToFirstTokenMs
		}
		if metrics.TimeToLastTokenMs != nil {
			performance["time_to_last_token_ms"] = *metrics.TimeToLastTokenMs
		}
		if metrics.TokensPerSecond != nil {
			performance["tokens_per_second"] = *metrics.TokensPerSecond
		}
		if metrics.ChunksReceived > 0 {
			performance["chunks_received"] = metrics.ChunksReceived
		}
		if metrics.AvgChunkSize != nil {
			performance["avg_chunk_size"] = *metrics.AvgChunkSize
		}
		if metrics.InterChunkLatencyVarianceMs != nil {
			performance["inter_chunk_latency_variance_ms"] = *metrics.InterChunkLatencyVarianceMs
		}
	}
	if len(performance) > 0 {
		data["performance"] = performance
	}

	payload := usageEventPayload{
		ProviderKeyID: p.providerKeyID,
		Provider:      p.underlyingName,
		Model:         completion.Model,
		Data:          data,
		ID:            uuid.New().String(),
		ClientName:    p.clientName,
	}

	jsonPayload, err := json.Marshal(payload)
	if err != nil {
		return
	}

	platformURL := os.Getenv(envPlatformURL)
	if platformURL == "" {
		platformURL = defaultPlatformURL
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, platformURL+"/usage-events/", strings.NewReader(string(jsonPayload)))
	if err != nil {
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+accessToken)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return
	}

	// Drain and close the response body to allow connection reuse
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
}
