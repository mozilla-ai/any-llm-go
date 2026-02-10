// Package version provides version information for the any-llm-go library.
package version

// Version is the library version. Defaults to "dev" for development builds.
// Can be set at build time via:
//
//	go build -ldflags="-X github.com/mozilla-ai/any-llm-go/version.Version=x.y.z"
var Version = "dev"
