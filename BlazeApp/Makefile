SHELL := /bin/bash
BINARY_NAME := bin/api
MAIN_FILE := cmd/main.go

# Config for Wrapper
W_GO  := wrapper/wrapper.go
W_DEV := wrapper/wrapper.go.dev
W_PRD := wrapper/wrapper.go.prod

# ANSI color codes
GREEN  := \033[1;32m
RED    := \033[1;31m
YELLOW := \033[1;33m
BLUE   := \033[1;34m
RESET  := \033[0m

.PHONY: dev prod status build run help

## dev: Switch to Development mode 🛠️
dev:
	@echo -e "$(YELLOW)🛠️  Switching to [DEV] mode...$(RESET)"
	@if [ -f $(W_GO) ]; then mv $(W_GO) $(W_PRD) 2>/dev/null; fi
	@if [ -f $(W_DEV) ]; then \
		mv $(W_DEV) $(W_GO); \
		echo -e "$(GREEN)✅ wrapper.dev is now ACTIVE as wrapper.go$(RESET)"; \
	else \
		echo -e "$(RED)⚠️  Already in DEV mode or wrapper.dev is missing.$(RESET)"; \
	fi

## prod: Switch to Production mode 🚀
prod:
	@echo -e "$(YELLOW)🚀 Switching to [PROD] mode...$(RESET)"
	@if [ -f $(W_GO) ]; then mv $(W_GO) $(W_DEV) 2>/dev/null; fi
	@if [ -f $(W_PRD) ]; then \
		mv $(W_PRD) $(W_GO); \
		echo -e "$(GREEN)✅ wrapper.prod is now ACTIVE as wrapper.go$(RESET)"; \
	else \
		echo -e "$(RED)⚠️  Already in PROD mode or wrapper.prod is missing.$(RESET)"; \
	fi

## status: Check current active mode 🔍
status:
	@echo -e "$(BLUE)🔎 Checking environment status...$(RESET)"
	@if [ -f $(W_PRD) ]; then \
		echo -e "Status: $(GREEN)🛠️  [ DEVELOPMENT MODE ]$(RESET)"; \
	elif [ -f $(W_DEV) ]; then \
		echo -e "Status: $(GREEN)🚀 [ PRODUCTION MODE ]$(RESET)"; \
	else \
		echo -e "Status: $(RED)❓ [ UNKNOWN ]$(RESET)"; \
	fi

## run: Execute the application (go run) 🏃
run:
	@echo -e "$(BLUE)🔍 Checking environment...$(RESET)"
	@if [ -f $(W_PRD) ]; then \
		echo -e "Starting in: $(GREEN)🛠️  DEV MODE$(RESET)"; \
	elif [ -f $(W_DEV) ]; then \
		echo -e "Starting in: $(GREEN)🚀 PROD MODE$(RESET)"; \
	fi
	@echo -e "$(YELLOW)⚡ Running $(MAIN_FILE)...$(RESET)"
	@echo -e "$(BLUE)------------------------------------------$(RESET)"
	@go run $(MAIN_FILE)

## build: Compile the Go binary 📦
build:
	@echo -e "$(BLUE)🔍 Checking if 'go' command is available...$(RESET)"
	@command -v go >/dev/null 2>&1 || { echo -e "$(RED)❌ Go is not installed.$(RESET)"; exit 1; }
	@echo -e "$(YELLOW)🚀 Starting Go build process...$(RESET)"
	@START=$$(date +%s); \
	GO111MODULE=on CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -trimpath -o $(BINARY_NAME) $(MAIN_FILE); \
	END=$$(date +%s); \
	DURATION=$$((END - START)); \
	if [ -f "$(BINARY_NAME)" ]; then \
		SIZE_MB=$$(awk "BEGIN {printf \"%.2f\", $$(stat -c%s $(BINARY_NAME)) / 1024 / 1024}"); \
		echo -e "$(GREEN)✅ Build completed successfully!$(RESET)"; \
		echo -e "$(BLUE)📦 Output:$(RESET) $(BINARY_NAME)"; \
		echo -e "$(BLUE)📏 Size:  $(RESET) $${SIZE_MB} MB"; \
		echo -e "$(BLUE)⏱️  Time:  $(RESET) $${DURATION} seconds"; \
	else \
		echo -e "$(RED)❌ Build failed. Output file not found.$(RESET)"; \
		exit 1; \
	fi

## help: Show available commands 📖
help:
	@echo -e "$(BLUE)Usage: make [target]$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
