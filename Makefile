# Show available targets
.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        Run lint, test, and build"
	@echo "  build      Verify compilation"
	@echo "  clean      Clean test cache"
	@echo "  help       Show this help"
	@echo "  lint       Run linting with auto-fix"
	@echo "  test       Run linting then all tests"
	@echo "  test-only  Run all tests without linting"
	@echo "  test-unit  Run unit tests only (skip integration)"
	@echo "  tidy       Tidy go.mod dependencies"

# Run all checks (lint + test + build)
.PHONY: all
all: test build

# Build and verify compilation
.PHONY: build
build:
	go build ./...

# Clean test cache
.PHONY: clean
clean:
	go clean -testcache

# Run linting with auto-fix
.PHONY: lint
lint:
	golangci-lint run --fix ./...

# Run all tests
.PHONY: test
test: lint
	go test -race ./...

# Run tests without linting (faster)
.PHONY: test-only
test-only:
	go test -v -race ./...

# Run unit tests only (skip integration tests)
.PHONY: test-unit
test-unit:
	go test -v -race -short ./...

# Tidy dependencies
.PHONY: tidy
tidy:
	go mod tidy
