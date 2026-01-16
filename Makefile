.PHONY: lint test build clean fmt

# Run linting with auto-fix
lint:
	golangci-lint run --fix ./...

# Run all tests
test: lint
	go test -v -race ./...

# Run tests without linting (faster)
test-only:
	go test -v -race ./...

# Run unit tests only (skip integration tests)
test-unit:
	go test -v -race -short ./...

# Build and verify compilation
build:
	go build ./...

# Format code
fmt:
	gofmt -s -w .
	goimports -w .

# Clean test cache
clean:
	go clean -testcache

# Tidy dependencies
tidy:
	go mod tidy

# Run all checks (lint + test + build)
all: lint test build
