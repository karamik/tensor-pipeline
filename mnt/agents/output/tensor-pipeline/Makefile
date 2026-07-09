# =============================================================================
# Tensor Pipeline — Makefile
# =============================================================================

.PHONY: help train optimize serve stop test benchmark clean lint

help: ## Show this help message
	@echo "Tensor Pipeline — Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------
train: ## Run training service
	docker compose run --rm training

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------
optimize: ## Run optimization service (converts to ONNX/TensorRT)
	docker compose run --rm optimization

# ------------------------------------------------------------------------------
# Serving
# ------------------------------------------------------------------------------
serve: ## Start Triton + Gateway (inference only)
	docker compose up --build triton gateway

serve-all: ## Start full stack with monitoring (Triton + Gateway + Prometheus + Grafana)
	docker compose --profile monitoring up --build

# ------------------------------------------------------------------------------
# Full Pipeline
# ------------------------------------------------------------------------------
pipeline: ## Run full pipeline: train → optimize → serve
	@echo "🚀 Step 1/3: Training..."
	$(MAKE) train
	@echo "🔧 Step 2/3: Optimization..."
	$(MAKE) optimize
	@echo "🌐 Step 3/3: Starting inference services..."
	$(MAKE) serve

# ------------------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------------------
stop: ## Stop all services
	docker compose down

clean: ## Remove all containers, volumes, and artifacts
	docker compose down -v --remove-orphans
	docker system prune -f

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
test: ## Run integration tests against running Gateway
	pytest tests/test_gateway.py -v

benchmark: ## Run load benchmark (requires locust)
	locust -f tests/benchmark.py --host=http://localhost:8080

# ------------------------------------------------------------------------------
# Development
# ------------------------------------------------------------------------------
lint: ## Run code linters
	black src/ tests/
	isort src/ tests/
	flake8 src/ tests/
