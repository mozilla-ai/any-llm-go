package llm

import (
	"strings"
	"sync"
)

// ProviderFactory is a function that creates a new Provider instance.
type ProviderFactory func(opts ...Option) (Provider, error)

// registry holds the registered provider factories.
var (
	registry   = make(map[string]ProviderFactory)
	registryMu sync.RWMutex
)

// Register registers a provider factory with the given name.
// This is typically called in init() functions of provider packages.
func Register(name string, factory ProviderFactory) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registry[strings.ToLower(name)] = factory
}

// NewProvider creates a new Provider instance for the given provider name.
// It returns an error if the provider is not registered or if creation fails.
func NewProvider(name string, opts ...Option) (Provider, error) {
	registryMu.RLock()
	factory, ok := registry[strings.ToLower(name)]
	registryMu.RUnlock()

	if !ok {
		return nil, NewUnsupportedProviderError(name)
	}

	return factory(opts...)
}

// RegisteredProviders returns a list of all registered provider names.
func RegisteredProviders() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()

	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	return names
}

// IsRegistered returns true if a provider with the given name is registered.
func IsRegistered(name string) bool {
	registryMu.RLock()
	defer registryMu.RUnlock()
	_, ok := registry[strings.ToLower(name)]
	return ok
}

// MustNewProvider creates a new Provider instance and panics if creation fails.
// This is useful for initialization code where failure should be fatal.
func MustNewProvider(name string, opts ...Option) Provider {
	p, err := NewProvider(name, opts...)
	if err != nil {
		panic(err)
	}
	return p
}

// ParseModelString parses a model string in the format "provider:model" or just "model".
// Returns the provider name and model name.
// If no provider is specified, returns empty string for provider.
func ParseModelString(model string) (provider, modelName string) {
	parts := strings.SplitN(model, ":", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return "", model
}
