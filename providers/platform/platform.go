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
	"runtime"
	"strings"
	"time"

	"github.com/google/uuid"
	anyllmplatform "github.com/mozilla-ai/any-llm-platform-client-go"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
	"github.com/mozilla-ai/any-llm-go/providers/anthropic"
	"github.com/mozilla-ai/any-llm-go/providers/openai"
	"github.com/mozilla-ai/any-llm-go/sdk"
)

// Provider configuration constants.
const (
	defaultPlatformURL = "https://platform-api.any-llm.ai/api/v1"
	envAPIKey          = "ANY_LLM_KEY"
	envPlatformURL     = "ANY_LLM_PLATFORM_URL"
	providerName       = "platform"
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
)

// supportedProviders maps provider names to their constructors.
var supportedProviders = map[string]newProviderFunc{
	"anthropic": func(opts ...config.Option) (providers.Provider, error) {
		return anthropic.New(opts...)
	},
	"openai": func(opts ...config.Option) (providers.Provider, error) {
		return openai.New(opts...)
	},
}

// Provider implements the providers.Provider interface for the ANY LLM platform.
// It proxies requests to underlying providers (OpenAI, Anthropic, etc.) after
// authenticating with the platform to get decrypted API keys.
type Provider struct {
	anyLLMKey          string
	clientName         string
	config             *config.Config
	httpClient         *http.Client
	platformClient     *anyllmplatform.Client
	platformURL        string
	projectID          string
	providerKeyID      string
	underlyingName     string
	underlyingProvider providers.Provider
}

// newProviderFunc creates a provider with the given options.
type newProviderFunc func(opts ...config.Option) (providers.Provider, error)

// streamingMetrics holds performance metrics for streaming requests.
type streamingMetrics struct {
	avgChunkSize                *float64
	chunksReceived              int
	interChunkLatencyVarianceMs *float64
	timeToFirstTokenMs          *float64
	timeToLastTokenMs           *float64
	tokensPerSecond             *float64
	totalDurationMs             float64
}

// usageEventPayload represents the payload for usage events.
type usageEventPayload struct {
	ClientName    string         `json:"client_name,omitempty"`
	Data          map[string]any `json:"data"`
	ID            string         `json:"id"`
	Model         string         `json:"model"`
	Provider      string         `json:"provider"`
	ProviderKeyID string         `json:"provider_key_id"`
}

// New creates a new platform provider.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	anyLLMKey := cfg.ResolveAPIKey(envAPIKey)
	if anyLLMKey == "" {
		return nil, errors.NewMissingAPIKeyError(providerName, envAPIKey)
	}

	platformURL := cfg.ResolveEnv(envPlatformURL)
	if platformURL == "" {
		platformURL = defaultPlatformURL
	}

	platformClient := anyllmplatform.NewClient(
		anyllmplatform.WithPlatformURL(platformURL),
	)

	// Read client name from Extra if set.
	var clientName string
	if v, ok := cfg.ExtraValue("client_name"); ok {
		clientName, _ = v.(string)
	}

	return &Provider{
		anyLLMKey:      anyLLMKey,
		clientName:     clientName,
		config:         cfg,
		httpClient:     &http.Client{Timeout: 30 * time.Second},
		platformClient: platformClient,
		platformURL:    platformURL,
	}, nil
}

// WithClientName sets a client name for per-client usage tracking.
func WithClientName(name string) config.Option {
	return config.WithExtra("client_name", strings.TrimSpace(name))
}

// Capabilities returns the provider's capabilities.
// Since this is a proxy, capabilities depend on the underlying provider.
func (p *Provider) Capabilities() providers.Capabilities {
	// Return full capabilities since we can proxy to any provider.
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       true,
		CompletionReasoning: true,
		CompletionStreaming: true,
		CompletionTools:     true,
		Embedding:           true,
		ListModels:          true,
	}
}

// Completion performs a chat completion request.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	startTime := time.Now()

	// Parse the model to get the provider name.
	providerName, modelID := parseModelString(params.Model)
	if providerName == "" {
		return nil, fmt.Errorf("model must be in format 'provider:model', got %q", params.Model)
	}

	// Initialize the underlying provider.
	if err := p.initializeProvider(ctx, providerName); err != nil {
		return nil, err
	}

	// Create a copy with the provider-specific model ID.
	completionParams := params
	completionParams.Model = modelID

	// Delegate to the underlying provider.
	completion, err := p.underlyingProvider.Completion(ctx, completionParams)
	if err != nil {
		return nil, err
	}

	// Post usage event.
	totalDurationMs := float64(time.Since(startTime).Milliseconds())
	go p.postUsageEvent(context.Background(), completion, nil, totalDurationMs)

	return completion, nil
}

// CompletionStream performs a streaming chat completion request.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		startTime := time.Now()

		// Parse the model to get the provider name.
		providerName, modelID := parseModelString(params.Model)
		if providerName == "" {
			errs <- fmt.Errorf("model must be in format 'provider:model', got %q", params.Model)
			return
		}

		// Initialize the underlying provider.
		if err := p.initializeProvider(ctx, providerName); err != nil {
			errs <- err
			return
		}

		// Create a copy with updated fields.
		streamParams := params
		streamParams.Model = modelID
		streamParams.Stream = true

		// Ensure we get usage data in the streaming response for tracking.
		if streamParams.StreamOptions == nil {
			streamParams.StreamOptions = &providers.StreamOptions{IncludeUsage: true}
		} else if !streamParams.StreamOptions.IncludeUsage {
			// Create a copy of StreamOptions to avoid mutating the original.
			opts := *streamParams.StreamOptions
			opts.IncludeUsage = true
			streamParams.StreamOptions = &opts
		}

		// Get the stream from the underlying provider.
		upstreamChunks, upstreamErrs := p.underlyingProvider.CompletionStream(ctx, streamParams)

		// Track streaming metrics.
		var (
			collectedChunks     []providers.ChatCompletionChunk
			timeToFirstTokenMs  *float64
			timeToLastContentMs *float64
			previousChunkTime   *time.Time
			chunkLatencies      []float64
		)

		// Forward chunks and collect for usage tracking.
		for chunk := range upstreamChunks {
			currentTime := time.Now()

			// Track time to first token.
			if timeToFirstTokenMs == nil && len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				ms := float64(currentTime.Sub(startTime).Milliseconds())
				timeToFirstTokenMs = &ms
			}

			// Track time to last content token.
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				ms := float64(currentTime.Sub(startTime).Milliseconds())
				timeToLastContentMs = &ms
			}

			// Track inter-chunk latency.
			if previousChunkTime != nil {
				latencyMs := float64(currentTime.Sub(*previousChunkTime).Milliseconds())
				chunkLatencies = append(chunkLatencies, latencyMs)
			}
			previousChunkTime = &currentTime

			collectedChunks = append(collectedChunks, chunk)
			chunks <- chunk
		}

		// Check for upstream errors.
		if err := <-upstreamErrs; err != nil {
			errs <- err
			return
		}

		// Post usage event with streaming metrics.
		if len(collectedChunks) > 0 {
			totalDurationMs := float64(time.Since(startTime).Milliseconds())
			completion := combineChunks(collectedChunks)

			metrics := &streamingMetrics{
				timeToFirstTokenMs: timeToFirstTokenMs,
				timeToLastTokenMs:  timeToLastContentMs,
				totalDurationMs:    totalDurationMs,
				chunksReceived:     len(collectedChunks),
			}

			// Calculate tokens per second if we have usage data.
			if completion.Usage != nil && completion.Usage.CompletionTokens > 0 && timeToLastContentMs != nil &&
				*timeToLastContentMs > 0 {
				tps := float64(completion.Usage.CompletionTokens*1000) / *timeToLastContentMs
				metrics.tokensPerSecond = &tps

				avgChunkSize := float64(completion.Usage.CompletionTokens) / float64(len(collectedChunks))
				metrics.avgChunkSize = &avgChunkSize
			}

			// Calculate inter-chunk latency variance if we have enough data points.
			if len(chunkLatencies) > 1 {
				variance := calculateVariance(chunkLatencies)
				metrics.interChunkLatencyVarianceMs = &variance
			}

			go p.postUsageEvent(context.Background(), completion, metrics, totalDurationMs)
		}
	}()

	return chunks, errs
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return providerName
}

// initializeProvider initializes the underlying provider for the given provider name.
func (p *Provider) initializeProvider(ctx context.Context, providerName string) error {
	if p.underlyingProvider != nil && p.underlyingName == providerName {
		return nil // Already initialized for this provider.
	}

	// Get decrypted provider key from the platform.
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

// postUsageEvent posts a usage event to the platform.
func (p *Provider) postUsageEvent(
	ctx context.Context,
	completion *providers.ChatCompletion,
	metrics *streamingMetrics,
	totalDurationMs float64,
) {
	if completion == nil || completion.Usage == nil {
		return
	}

	// Get access token for Bearer authentication.
	accessToken, err := p.platformClient.GetAccessToken(ctx, p.anyLLMKey)
	if err != nil {
		return
	}

	// Build data payload.
	data := map[string]any{
		"input_tokens":  fmt.Sprintf("%d", completion.Usage.PromptTokens),
		"output_tokens": fmt.Sprintf("%d", completion.Usage.CompletionTokens),
	}

	// Add performance metrics.
	performance := map[string]any{}
	if totalDurationMs > 0 {
		performance["total_duration_ms"] = totalDurationMs
	}
	if metrics != nil {
		if metrics.timeToFirstTokenMs != nil {
			performance["time_to_first_token_ms"] = *metrics.timeToFirstTokenMs
		}
		if metrics.timeToLastTokenMs != nil {
			performance["time_to_last_token_ms"] = *metrics.timeToLastTokenMs
		}
		if metrics.tokensPerSecond != nil {
			performance["tokens_per_second"] = *metrics.tokensPerSecond
		}
		if metrics.chunksReceived > 0 {
			performance["chunks_received"] = metrics.chunksReceived
		}
		if metrics.avgChunkSize != nil {
			performance["avg_chunk_size"] = *metrics.avgChunkSize
		}
		if metrics.interChunkLatencyVarianceMs != nil {
			performance["inter_chunk_latency_variance_ms"] = *metrics.interChunkLatencyVarianceMs
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

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		p.platformURL+"/usage-events/",
		strings.NewReader(string(jsonPayload)),
	)
	if err != nil {
		return
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("User-Agent", userAgent())

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return
	}

	// Drain and close the response body to allow connection reuse.
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
}

// calculateVariance calculates the sample variance of a slice of float64 values.
func calculateVariance(values []float64) float64 {
	if len(values) < 2 {
		return 0
	}

	// Calculate mean.
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	// Calculate sum of squared differences.
	var sumSquaredDiff float64
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}

	// Return sample variance (n-1).
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

// parseModelString parses a model string in the format "provider:model" or just "model".
func parseModelString(model string) (providerName string, modelID string) {
	parts := strings.SplitN(model, ":", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return "", model
}

// userAgent returns the formatted User-Agent header value per RFC 9110 section 10.1.5.
func userAgent() string {
	goVersion := strings.TrimPrefix(runtime.Version(), "go")
	return fmt.Sprintf("%s/%s go/%s", sdk.Name, strings.TrimPrefix(sdk.Version, "v"), goVersion)
}
