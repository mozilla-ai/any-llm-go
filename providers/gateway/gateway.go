// Package gateway provides a gateway provider implementation for any-llm.
// It connects to an any-llm gateway server, which proxies requests to
// underlying LLM providers.
//
// This package supersedes providers/platform. Where the platform provider
// decrypts provider keys client-side and fans out to individual provider
// SDKs, the gateway delegates everything to a single server-side endpoint.
//
// The gateway supports two authentication modes:
//   - Platform mode: uses a platform token as standard Bearer auth
//   - Non-platform mode: sends a gateway API key via a custom authentication
//     header (AnyLLM-Key)
package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"

	"github.com/openai/openai-go"

	"github.com/mozilla-ai/any-llm-go/config"
	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
	openaiProvider "github.com/mozilla-ai/any-llm-go/providers/openai"
)

// Provider configuration constants.
const (
	// apiKeyHeaderName is the HTTP header that carries the gateway API key in
	// non-platform mode.
	apiKeyHeaderName = "AnyLLM-Key"

	// authorizationHeader is the standard HTTP Authorization header.
	authorizationHeader = "Authorization"

	// bearerPrefix is the Authorization scheme prefix applied to the gateway
	// key before it is placed in the gateway header (e.g. "Bearer <key>").
	bearerPrefix = "Bearer "

	// contentTypeHeader is the standard HTTP Content-Type header.
	contentTypeHeader = "Content-Type"

	// contentTypeJSON is the Content-Type value for JSON request bodies.
	contentTypeJSON = "application/json"

	// placeholderAPIKey satisfies the OpenAI SDK's requirement that an API key
	// be set. The SDK still sends it in the Authorization header, but in
	// non-platform mode real auth is carried by the gateway header, so this
	// is a non-secret placeholder that the gateway ignores.
	placeholderAPIKey = "gateway-no-key"

	// envAPIBase is the environment variable read for the gateway base URL
	// when WithBaseURL is not passed to New.
	envAPIBase = "GATEWAY_API_BASE"

	// envAPIKey is the environment variable read for the gateway API key used
	// in non-platform mode when WithGatewayKey is not passed to New.
	envAPIKey = "GATEWAY_API_KEY"

	// envPlatformToken is the environment variable read for the platform
	// token used as Bearer auth in platform mode.
	envPlatformToken = "GATEWAY_PLATFORM_TOKEN"

	// extraKeyGatewayKey is the config.Extra key used to coordinate
	// WithGatewayKey (writer) with the resolver logic in New (reader).
	extraKeyGatewayKey = "gateway_key"

	// extraKeyPlatformMode is the config.Extra key used to coordinate
	// WithPlatformMode (writer) with the resolver logic in New (reader).
	extraKeyPlatformMode = "platform_mode"

	// providerName is the value returned by Provider.Name and embedded in
	// errors produced by this package.
	providerName = "gateway"
)

// Gateway-specific error codes.
const (
	// errCodeBatchNotComplete is the BaseError.Code set when a batch result
	// is requested before the batch has finished processing (HTTP 409).
	errCodeBatchNotComplete = "batch_not_complete"

	// errCodeTimeout is the BaseError.Code set on gateway timeout errors
	// (HTTP 504).
	errCodeTimeout = "gateway_timeout"

	// errCodeUpstreamProvider is the BaseError.Code set on upstream provider
	// errors (HTTP 502).
	errCodeUpstreamProvider = "upstream_provider"
)

// Batch API path constants.
const (
	// batchesPath is the base path for batch operations on the gateway.
	batchesPath = "/v1/batches"

	// providerQueryParam is the query-string key identifying the upstream
	// provider for batch operations.
	providerQueryParam = "provider"
)

// Gateway-specific sentinel errors for type checking with errors.Is().
var (
	// ErrBatchNotComplete is matched by errors.Is when retrieving batch
	// results on a batch that has not yet finished processing (HTTP 409).
	ErrBatchNotComplete = stderrors.New("batch not yet complete")

	// ErrTimeout is matched by errors.Is on gateway timeout errors (HTTP 504).
	ErrTimeout = stderrors.New("gateway timeout")

	// ErrUpstreamProvider is matched by errors.Is on upstream provider errors
	// (HTTP 502).
	ErrUpstreamProvider = stderrors.New("upstream provider error")
)

// Ensure Provider implements the required interfaces.
var (
	_ providers.BatchProvider      = (*Provider)(nil)
	_ providers.CapabilityProvider = (*Provider)(nil)
	_ providers.EmbeddingProvider  = (*Provider)(nil)
	_ providers.ErrorConverter     = (*Provider)(nil)
	_ providers.ModelLister        = (*Provider)(nil)
	_ providers.Provider           = (*Provider)(nil)
	_ providers.RerankProvider     = (*Provider)(nil)
)

// BatchNotCompleteError is returned when RetrieveBatchResults is called on
// a batch that has not finished processing yet (HTTP 409).
type BatchNotCompleteError struct {
	errors.BaseError
	BatchID string
	Status  string
}

// TimeoutError is returned when the gateway times out (HTTP 504).
type TimeoutError struct {
	errors.BaseError
}

// UpstreamProviderError is returned when the upstream provider is
// unreachable (HTTP 502).
type UpstreamProviderError struct {
	errors.BaseError
}

// newBatchNotCompleteError constructs a BatchNotCompleteError for the given
// batch and upstream-reported status. status may be empty when the gateway
// did not include a structured status field in the response.
func newBatchNotCompleteError(batchID, status string, cause error) *BatchNotCompleteError {
	if cause == nil {
		switch {
		case batchID != "" && status != "":
			cause = fmt.Errorf("batch %q is not yet complete (status: %s)", batchID, status)
		case batchID != "":
			cause = fmt.Errorf("batch %q is not yet complete", batchID)
		default:
			cause = stderrors.New("batch is not yet complete")
		}
	}
	return &BatchNotCompleteError{
		BaseError: errors.New(errCodeBatchNotComplete, providerName, cause, ErrBatchNotComplete),
		BatchID:   batchID,
		Status:    status,
	}
}

// Provider implements the providers.Provider interface for the any-llm gateway.
// It embeds openai.CompatibleProvider since the gateway exposes an
// OpenAI-compatible API.
type Provider struct {
	*openaiProvider.CompatibleProvider

	// apiBase is the gateway base URL, used for batch API calls that bypass
	// the OpenAI SDK.
	apiBase string

	// apiKey is the gateway API key for non-platform mode, or the platform
	// token in platform mode, used for batch API calls that bypass the
	// OpenAI SDK.
	apiKey string

	// baseURL is the resolved gateway base URL without a trailing /v1 suffix.
	// Used by Rerank to construct the /v1/rerank endpoint URL.
	baseURL string

	// httpClient is the HTTP client used for raw HTTP calls (e.g. rerank,
	// batch API) that bypass the OpenAI SDK. In non-platform mode this is
	// the header-injecting client so gateway auth is preserved.
	httpClient *http.Client

	// platformMode indicates whether the provider is operating in platform
	// mode (using platform token for Bearer auth) or non-platform mode
	// (using gateway key in custom header). This affects how authentication
	// is handled and how errors are converted.
	platformMode bool
}

// headerTransport wraps an http.RoundTripper to inject custom headers into
// every request. Used in non-platform mode to add the gateway authentication
// header.
type headerTransport struct {
	base   http.RoundTripper
	header string
	value  string
}

// New creates a new gateway provider.
//
// The gateway base URL is required and can be set via config.WithBaseURL() or
// the GATEWAY_API_BASE environment variable.
//
// Authentication mode is determined as follows:
//   - If WithPlatformMode() is passed, platform mode is used. The token is
//     resolved from config.WithAPIKey() or GATEWAY_PLATFORM_TOKEN.
//   - If GATEWAY_PLATFORM_TOKEN is set and no explicit API key or gateway key
//     is provided, platform mode is auto-detected.
//   - Otherwise, non-platform mode is used with the key from WithGatewayKey()
//     or GATEWAY_API_KEY, sent via the gateway authentication header.
func New(opts ...config.Option) (*Provider, error) {
	cfg, err := config.New(opts...)
	if err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}

	// Resolve gateway key from extras or GATEWAY_API_KEY env var.
	var gatewayKey string
	if v, ok := cfg.ExtraValue(extraKeyGatewayKey); ok {
		if key, ok := v.(string); ok && key != "" {
			gatewayKey = key
		}
	}
	if gatewayKey == "" {
		gatewayKey = cfg.ResolveEnv(envAPIKey)
	}

	// Determine authentication mode.
	platformMode := false
	var platformToken string

	// Explicit opt-in via WithPlatformMode().
	if v, ok := cfg.ExtraValue(extraKeyPlatformMode); ok {
		if enabled, ok := v.(bool); ok && enabled {
			platformMode = true
			platformToken = cfg.APIKey
			if platformToken == "" {
				platformToken = cfg.ResolveEnv(envPlatformToken)
			}
		}
	}

	// Auto-detect: GATEWAY_PLATFORM_TOKEN set and no explicit API key or
	// gateway key configured. A gateway key signals non-platform intent.
	if !platformMode {
		envToken := cfg.ResolveEnv(envPlatformToken)
		if envToken != "" && cfg.APIKey == "" && gatewayKey == "" {
			platformMode = true
			platformToken = envToken
		}
	}

	if platformMode && platformToken == "" {
		return nil, fmt.Errorf(
			"platform mode requires a token (pass WithAPIKey option or set %s env var)",
			envPlatformToken,
		)
	}

	// Resolve apiBase once here and pass the resolved value through to
	// NewCompatible via WithBaseURL. Resolving in a single place avoids
	// the possibility of the batch HTTP client and the OpenAI SDK client
	// disagreeing (e.g. if config state changed between resolutions) and
	// lets us surface resolution errors directly instead of relying on
	// NewCompatible's RequireBaseURL check.
	apiBase, err := cfg.ResolveBaseURL(envAPIBase, "")
	if err != nil {
		return nil, err
	}
	if apiBase == "" {
		return nil, fmt.Errorf(
			"gateway base URL is required (set via WithBaseURL option or %s env var)",
			envAPIBase,
		)
	}
	apiBase = strings.TrimRight(apiBase, "/")

	// Pass the user's opts straight through and layer auth-specific opts on
	// top (order matters: later options win). WithBaseURL is appended last
	// so NewCompatible sees the already-resolved, trimmed URL and does not
	// re-run ResolveBaseURL.
	compatOpts := slices.Clone(opts)
	httpClient := cfg.HTTPClient()
	if platformMode {
		compatOpts = append(compatOpts, config.WithAPIKey(platformToken))
		// Wrap the HTTP client so raw HTTP calls (e.g. Rerank) that bypass
		// the OpenAI SDK also carry the platform Bearer token.
		httpClient = newBearerClient(httpClient, platformToken)
	} else {
		// Non-platform mode: override any user-supplied API key with the
		// placeholder so secrets can't leak via the Authorization header.
		// Real auth, if any, is carried by the gateway header below.
		compatOpts = append(compatOpts, config.WithAPIKey(placeholderAPIKey))
		if gatewayKey != "" {
			httpClient = newHeaderClient(cfg.HTTPClient(), bearerPrefix+gatewayKey)
			compatOpts = append(compatOpts, config.WithHTTPClient(httpClient))
		}
	}
	compatOpts = append(compatOpts, config.WithBaseURL(apiBase))

	base, err := openaiProvider.NewCompatible(openaiProvider.CompatibleConfig{
		APIKeyEnvVar:   "", // Gateway uses its own key resolution.
		BaseURLEnvVar:  "", // Base URL is already resolved above.
		Capabilities:   capabilities(),
		DefaultAPIKey:  placeholderAPIKey, // Placeholder; non-platform doesn't need real auth.
		DefaultBaseURL: "",                // No default; base URL is required.
		Name:           providerName,
		RequireAPIKey:  false, // Gateway handles auth separately.
		RequireBaseURL: true,  // Guarded above; defensive double-check.
	}, compatOpts...)
	if err != nil {
		return nil, err
	}

	// Determine the key to use for batch API calls.
	var batchAPIKey string
	if platformMode {
		batchAPIKey = platformToken
	} else {
		batchAPIKey = gatewayKey
	}

	// rawBaseURL strips any trailing /v1 suffix so that raw HTTP endpoints
	// (e.g. /v1/rerank) can prepend the version prefix themselves.
	rawBaseURL := strings.TrimSuffix(apiBase, "/v1")
	rawBaseURL = strings.TrimSuffix(rawBaseURL, "/v1/")

	return &Provider{
		CompatibleProvider: base,
		apiBase:            apiBase,
		apiKey:             batchAPIKey,
		baseURL:            rawBaseURL,
		httpClient:         httpClient,
		platformMode:       platformMode,
	}, nil
}

// capabilities returns the full set of capabilities for the gateway provider.
// Since the gateway proxies to any backend provider, all features are
// optimistically marked as supported. Actual support depends on the
// underlying provider behind the gateway; consumers should handle
// unsupported-operation errors at call time.
func capabilities() providers.Capabilities {
	return providers.Capabilities{
		Completion:          true,
		CompletionImage:     true,
		CompletionPDF:       true,
		CompletionReasoning: true,
		CompletionStreaming:  true,
		CompletionTools:     true,
		Embedding:           true,
		ListModels:          true,
		Rerank:              true,
	}
}

// WithGatewayKey sets the gateway API key for non-platform mode authentication.
// The key is sent as a Bearer-formatted value in the gateway authentication
// header (AnyLLM-Key).
func WithGatewayKey(key string) config.Option {
	return config.WithExtra(extraKeyGatewayKey, key)
}

// WithPlatformMode explicitly enables platform mode authentication.
// In platform mode, the token (from config.WithAPIKey or GATEWAY_PLATFORM_TOKEN)
// is used as standard Bearer authentication.
func WithPlatformMode() config.Option {
	return config.WithExtra(extraKeyPlatformMode, true)
}

// Completion performs a chat completion request.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) Completion(
	ctx context.Context,
	params providers.CompletionParams,
) (*providers.ChatCompletion, error) {
	resp, err := p.CompatibleProvider.Completion(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return resp, nil
}

// CompletionStream performs a streaming chat completion request.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) CompletionStream(
	ctx context.Context,
	params providers.CompletionParams,
) (<-chan providers.ChatCompletionChunk, <-chan error) {
	upstreamChunks, upstreamErrs := p.CompatibleProvider.CompletionStream(ctx, params)

	chunks := make(chan providers.ChatCompletionChunk)
	errs := make(chan error, 1)

	go func() {
		defer close(chunks)
		defer close(errs)

		for chunk := range upstreamChunks {
			select {
			case chunks <- chunk:
			case <-ctx.Done():
				errs <- ctx.Err()
				return
			}
		}

		if err := <-upstreamErrs; err != nil {
			errs <- p.ConvertError(err)
		}
	}()

	return chunks, errs
}

// ConvertError converts errors to gateway-specific typed errors where
// applicable, falling back to the base OpenAI error conversion for raw
// errors. Already-converted errors (containing a BaseError) are not
// double-wrapped.
func (p *Provider) ConvertError(err error) error {
	if err == nil {
		return nil
	}

	// Check for gateway-specific HTTP status codes.
	var apiErr *openai.Error
	if stderrors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case http.StatusPaymentRequired:
			return errors.NewInsufficientFundsError(providerName, apiErr)
		case http.StatusBadGateway:
			return &UpstreamProviderError{
				BaseError: errors.New(errCodeUpstreamProvider, providerName, apiErr, ErrUpstreamProvider),
			}
		case http.StatusGatewayTimeout:
			return &TimeoutError{
				BaseError: errors.New(errCodeTimeout, providerName, apiErr, ErrTimeout),
			}
		}
	}

	// If the error was already converted (has a BaseError), return as-is
	// to avoid double-wrapping.
	var baseErr *errors.BaseError
	if stderrors.As(err, &baseErr) {
		return err
	}

	// Raw error: delegate to base OpenAI error conversion.
	return p.CompatibleProvider.ConvertError(err)
}

// Embedding generates embeddings for the given input.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) Embedding(
	ctx context.Context,
	params providers.EmbeddingParams,
) (*providers.EmbeddingResponse, error) {
	resp, err := p.CompatibleProvider.Embedding(ctx, params)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return resp, nil
}

// ListModels returns a list of available models from the gateway.
// Gateway-specific errors (402, 502, 504) are converted to typed errors.
func (p *Provider) ListModels(ctx context.Context) (*providers.ModelsResponse, error) {
	resp, err := p.CompatibleProvider.ListModels(ctx)
	if err != nil {
		return nil, p.ConvertError(err)
	}

	return resp, nil
}

// Rerank reranks documents by relevance to a query via the gateway's /v1/rerank endpoint.
// The response contains results sorted by relevance_score in descending order.
func (p *Provider) Rerank(ctx context.Context, params providers.RerankParams) (*providers.RerankResponse, error) {
	if params.Model == "" {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("model is required"))
	}
	if params.Query == "" {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("query is required"))
	}
	if len(params.Documents) == 0 {
		return nil, errors.NewInvalidRequestError(providerName, fmt.Errorf("at least one document is required"))
	}

	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshaling rerank request: %w", err)
	}

	reqURL := p.baseURL + "/v1/rerank"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, reqURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating rerank request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, p.ConvertError(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, p.handleRerankErrorResponse(resp)
	}

	var result providers.RerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decoding rerank response: %w", err)
	}
	return &result, nil
}

// handleRerankErrorResponse parses an HTTP error response from the /v1/rerank
// endpoint and returns a typed error.
func (p *Provider) handleRerankErrorResponse(resp *http.Response) error {
	// ReadAll error is ignored: if reading fails, we fall back to an empty
	// body and still produce a typed error based on the status code.
	body, _ := io.ReadAll(resp.Body)

	parsed := parseRerankError(body)
	msg := parsed.message()

	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return errors.NewAuthenticationError(providerName, fmt.Errorf("%s", msg))
	case http.StatusNotFound:
		return errors.NewModelNotFoundError(providerName, fmt.Errorf("%s", msg))
	case http.StatusPaymentRequired:
		return errors.NewInsufficientFundsError(providerName, fmt.Errorf("%s", msg))
	case http.StatusTooManyRequests:
		return errors.NewRateLimitError(providerName, fmt.Errorf("%s", msg))
	case http.StatusBadGateway:
		return &UpstreamProviderError{BaseError: errors.New(errCodeUpstreamProvider, providerName, fmt.Errorf("%s", msg), ErrUpstreamProvider)}
	case http.StatusGatewayTimeout:
		return &TimeoutError{BaseError: errors.New(errCodeTimeout, providerName, fmt.Errorf("%s", msg), ErrTimeout)}
	default:
		return errors.NewProviderError(providerName, fmt.Errorf("%s", msg))
	}
}

// rerankError holds the parsed fields from a gateway rerank error response.
// parseErr is non-nil when the body was not valid JSON or did not contain
// the expected "detail" field, so callers can detect drift in the gateway
// error shape.
type rerankError struct {
	Detail   string
	raw      string
	parseErr error
}

// message returns the best human-readable description of the error body.
func (e rerankError) message() string {
	if e.Detail != "" {
		return e.Detail
	}
	return e.raw
}

// parseRerankError parses a gateway rerank error response. Non-JSON bodies
// are preserved via the raw field so callers can still surface the server's
// text in the typed error. parseErr is set when the body cannot be decoded
// as JSON or when the "detail" field is empty, making any future callers
// that depend on structured fields aware of format drift.
func parseRerankError(body []byte) rerankError {
	raw := string(body)
	var wrapper struct {
		Detail string `json:"detail"`
	}
	if err := json.Unmarshal(body, &wrapper); err != nil {
		return rerankError{raw: raw, parseErr: fmt.Errorf("unmarshal error body: %w", err)}
	}
	if wrapper.Detail == "" {
		return rerankError{raw: raw, parseErr: stderrors.New("error body missing \"detail\" field")}
	}
	return rerankError{Detail: wrapper.Detail, raw: raw}
}

// RoundTrip implements http.RoundTripper by cloning the request and injecting
// the configured header before delegating to the base transport.
func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.Header.Set(t.header, t.value)
	return t.base.RoundTrip(clone)
}

// newBearerClient wraps the given HTTP client's transport to inject a standard
// Authorization: Bearer <token> header into every request. Used in platform
// mode for raw HTTP calls (e.g. Rerank) that bypass the OpenAI SDK.
func newBearerClient(base *http.Client, token string) *http.Client {
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	return &http.Client{
		Timeout: base.Timeout,
		Transport: &headerTransport{
			base:   transport,
			header: "Authorization",
			value:  bearerPrefix + token,
		},
	}
}

// newHeaderClient wraps the given HTTP client's transport to inject the
// gateway authentication header into every request, preserving the base
// client's timeout and transport settings.
func newHeaderClient(base *http.Client, headerValue string) *http.Client {
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	return &http.Client{
		Timeout: base.Timeout,
		Transport: &headerTransport{
			base:   transport,
			header: apiKeyHeaderName,
			value:  headerValue,
		},
	}
}

// --- Batch API methods ---
//
// These methods use raw HTTP (via doRequest) rather than the embedded OpenAI
// SDK because the SDK does not expose the gateway batch endpoints.

// CreateBatch creates a new batch job.
func (p *Provider) CreateBatch(
	ctx context.Context,
	params providers.CreateBatchParams,
) (*providers.Batch, error) {
	body, err := json.Marshal(params)
	if err != nil {
		return nil, errors.NewInvalidRequestError(providerName, err)
	}

	var batch providers.Batch
	if err := p.callBatchJSON(ctx, http.MethodPost, batchesPath, "", body, &batch); err != nil {
		return nil, err
	}

	return &batch, nil
}

// RetrieveBatch retrieves a batch job by ID.
func (p *Provider) RetrieveBatch(
	ctx context.Context,
	batchID string,
	provider string,
) (*providers.Batch, error) {
	path := batchItemPath(batchID, "", provider)

	var batch providers.Batch
	if err := p.callBatchJSON(ctx, http.MethodGet, path, batchID, nil, &batch); err != nil {
		return nil, err
	}

	return &batch, nil
}

// CancelBatch cancels a batch job.
func (p *Provider) CancelBatch(
	ctx context.Context,
	batchID string,
	provider string,
) (*providers.Batch, error) {
	path := batchItemPath(batchID, "cancel", provider)

	var batch providers.Batch
	if err := p.callBatchJSON(ctx, http.MethodPost, path, batchID, nil, &batch); err != nil {
		return nil, err
	}

	return &batch, nil
}

// ListBatches lists batch jobs for a provider.
func (p *Provider) ListBatches(
	ctx context.Context,
	provider string,
	opts providers.ListBatchesOptions,
) ([]providers.Batch, error) {
	params := url.Values{providerQueryParam: {provider}}
	if opts.After != "" {
		params.Set("after", opts.After)
	}
	if opts.Limit != nil {
		params.Set("limit", strconv.Itoa(*opts.Limit))
	}

	u := url.URL{Path: batchesPath, RawQuery: params.Encode()}
	path := u.RequestURI()

	var listResp struct {
		Data []providers.Batch `json:"data"`
	}
	if err := p.callBatchJSON(ctx, http.MethodGet, path, "", nil, &listResp); err != nil {
		return nil, err
	}

	return listResp.Data, nil
}

// RetrieveBatchResults retrieves the results of a completed batch job.
func (p *Provider) RetrieveBatchResults(
	ctx context.Context,
	batchID string,
	provider string,
) (*providers.BatchResult, error) {
	path := batchItemPath(batchID, "results", provider)

	var result providers.BatchResult
	if err := p.callBatchJSON(ctx, http.MethodGet, path, batchID, nil, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

// batchItemPath builds a path for /v1/batches/{id}[/action]?provider=X.
// It uses url.URL.JoinPath (Go 1.19+) which escapes each path segment
// individually, so a batchID containing "/" or ".." is encoded as a single
// segment rather than traversing into a sibling route.
func batchItemPath(batchID, action, provider string) string {
	u := (&url.URL{Path: batchesPath}).JoinPath(batchID)
	if action != "" {
		u = u.JoinPath(action)
	}
	u.RawQuery = url.Values{providerQueryParam: {provider}}.Encode()
	return u.RequestURI()
}

// callBatchJSON performs a batch HTTP request and decodes a JSON success
// response into out. Non-2xx responses are mapped to typed errors via
// handleBatchError. batchID is used to enrich 409 errors with the specific
// batch identifier the caller requested; pass "" when not applicable
// (create, list).
func (p *Provider) callBatchJSON(
	ctx context.Context,
	method, path, batchID string,
	body []byte,
	out any,
) error {
	resp, err := p.doRequest(ctx, method, path, body)
	if err != nil {
		return err
	}
	defer closeBody(resp)

	if resp.StatusCode != http.StatusOK {
		return p.handleBatchError(resp, batchID)
	}

	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return errors.NewProviderError(providerName, fmt.Errorf("failed to decode batch response: %w", err))
	}

	return nil
}

// closeBody closes an HTTP response body; intended for use in defer
// statements. The close error is deliberately discarded because this SDK
// does not expose a logger through config, and callers have already
// consumed the body by the time this runs.
func closeBody(resp *http.Response) {
	_ = resp.Body.Close()
}

// doRequest sends an HTTP request to the gateway API for batch operations.
func (p *Provider) doRequest(
	ctx context.Context,
	method, path string,
	body []byte,
) (*http.Response, error) {
	fullURL := p.apiBase + path

	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}

	req, err := http.NewRequestWithContext(ctx, method, fullURL, bodyReader)
	if err != nil {
		return nil, errors.NewProviderError(providerName, err)
	}

	req.Header.Set(contentTypeHeader, contentTypeJSON)
	if p.apiKey != "" {
		if p.platformMode {
			req.Header.Set(authorizationHeader, bearerPrefix+p.apiKey)
		} else {
			req.Header.Set(apiKeyHeaderName, bearerPrefix+p.apiKey)
		}
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, errors.NewProviderError(providerName, err)
	}

	return resp, nil
}

// handleBatchError maps a non-2xx batch HTTP response to a typed error.
// batchID is the batch identifier the caller operated on (empty for
// create/list); it is used to enrich 409 errors without parsing
// free-text error messages.
//
// All callers of this function operate on /v1/batches, so a 404 is
// interpreted as "gateway does not expose the batch API" and is not
// mapped to ModelNotFoundError.
func (p *Provider) handleBatchError(resp *http.Response, batchID string) error {
	// ReadAll error is ignored: if reading fails, we fall back to an empty
	// body and still produce a typed error based on status code.
	bodyBytes, _ := io.ReadAll(resp.Body)

	detail, parseErr := parseBatchError(bodyBytes)

	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden:
		return errors.NewAuthenticationError(providerName,
			fmt.Errorf("unauthorized: %s", detail.message()))
	case http.StatusNotFound:
		return errors.NewProviderError(providerName,
			stderrors.New("this gateway does not support batch operations; upgrade your gateway"))
	case http.StatusConflict:
		// For 409 the Status field matters (it tells the caller why the
		// batch isn't ready). If the body wasn't valid JSON, wrap the
		// parse error so drift in the gateway's response format is
		// visible instead of silently returning an empty Status.
		if parseErr != nil {
			return newBatchNotCompleteError(batchID, "",
				fmt.Errorf("failed to parse batch error response: %w", parseErr))
		}
		return newBatchNotCompleteError(batchID, detail.Status, detail.toError())
	case http.StatusUnprocessableEntity:
		return errors.NewProviderError(providerName,
			fmt.Errorf("unprocessable request: %s", detail.message()))
	case http.StatusTooManyRequests:
		return errors.NewRateLimitError(providerName,
			fmt.Errorf("rate limit: %s", detail.message()))
	case http.StatusBadGateway:
		return errors.NewProviderError(providerName,
			fmt.Errorf("upstream provider error: %s", detail.message()))
	default:
		return errors.NewProviderError(providerName,
			fmt.Errorf("HTTP %d: %s", resp.StatusCode, detail.message()))
	}
}

// batchError is the structured JSON shape gateway error responses may use.
// All fields are optional; callers must cope with arbitrary bodies.
type batchError struct {
	// Detail is the human-readable error message (FastAPI-style payload).
	Detail string `json:"detail"`

	// Status is the batch status when the gateway returns a 409 for
	// RetrieveBatchResults; empty when not applicable or not provided.
	Status string `json:"status"`

	// raw is the untouched response body, used as a last-resort message
	// when the body is not JSON or does not include Detail.
	raw string
}

// message returns the best human-readable description of the error body.
func (b batchError) message() string {
	if b.Detail != "" {
		return b.Detail
	}
	return b.raw
}

// toError wraps message() in an error value suitable for embedding in a
// typed error's cause chain, or nil when no detail is available.
func (b batchError) toError() error {
	msg := b.message()
	if msg == "" {
		return nil
	}
	return stderrors.New(msg)
}

// parseBatchError parses a gateway error response body. The parse error is
// returned so callers can decide per-status whether a JSON parse failure is
// tolerable (e.g. log and fall through for generic errors) or should be
// treated as a hard failure (e.g. for 409 where Status matters). Non-JSON
// bodies are preserved via the raw field so callers can still surface the
// server's text in the typed error.
func parseBatchError(body []byte) (batchError, error) {
	out := batchError{raw: string(body)}
	err := json.Unmarshal(body, &out)
	return out, err
}
