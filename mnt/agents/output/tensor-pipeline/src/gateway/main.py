"""
================================================================================
Tensor Pipeline — FastAPI Inference Gateway
================================================================================
Lightweight REST/gRPC proxy between clients and NVIDIA Triton Inference Server.

Features:
  • Input validation via Pydantic
  • Prometheus metrics (latency, throughput, errors)
  • Health checks & readiness probes
  • Dynamic batching support
  • Graceful error handling with structured responses
  • Configurable via environment variables

Endpoints:
  POST /predict          → Single prediction
  POST /predict/batch    → Batch prediction
  GET  /health           → Liveness probe
  GET  /ready            → Readiness probe (checks Triton connection)
  GET  /metrics          → Prometheus metrics
  GET  /model/info       → Model metadata from Triton
================================================================================
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import numpy as np
import httpx
import tritonclient.http as triton_http
import tritonclient.grpc as triton_grpc
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# =============================================================================
# CONFIGURATION
# =============================================================================
TRITON_HTTP_URL = os.getenv("TRITON_HTTP_URL", "http://localhost:8000")
TRITON_GRPC_URL = os.getenv("TRITON_GRPC_URL", "localhost:8001")
MODEL_NAME = os.getenv("MODEL_NAME", "resnet50")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("gateway")

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================
PREDICTION_COUNTER = Counter(
    "gateway_predictions_total",
    "Total predictions served",
    ["model_name", "status"]
)
PREDICTION_LATENCY = Histogram(
    "gateway_prediction_duration_seconds",
    "Prediction latency in seconds",
    ["model_name", "batch_size"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
ACTIVE_REQUESTS = Gauge(
    "gateway_active_requests",
    "Number of active prediction requests",
    ["model_name"]
)
TRITON_HEALTH = Gauge(
    "gateway_triton_health",
    "Triton server health status (1=healthy, 0=unhealthy)"
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class PredictRequest(BaseModel):
    """Single prediction request."""
    inputs: List[List[float]] = Field(
        ...,
        description="Input tensor(s). For image: [[R,G,B,...], ...]",
        min_items=1,
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Override default model name"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Override default model version"
    )

    @validator("inputs")
    def validate_inputs(cls, v):
        if not v or not all(v):
            raise ValueError("Inputs cannot be empty")
        return v


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    inputs: List[List[List[float]]] = Field(
        ...,
        description="Batch of input tensors",
        min_items=1,
        max_items=MAX_BATCH_SIZE,
    )
    model_name: Optional[str] = None
    model_version: Optional[str] = None

    @validator("inputs")
    def validate_batch_size(cls, v):
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")
        return v


class PredictResponse(BaseModel):
    """Standardized prediction response."""
    success: bool
    model_name: str
    model_version: str
    predictions: List[Any]
    inference_time_ms: float
    batch_size: int
    request_id: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    triton_connected: bool
    model_loaded: bool
    timestamp: float


class ErrorResponse(BaseModel):
    """Standardized error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


# =============================================================================
# TRITON CLIENT MANAGER
# =============================================================================
class TritonClient:
    """Manages connections to Triton Inference Server."""
    
    def __init__(self):
        self.http_client: Optional[triton_http.InferenceServerClient] = None
        self.grpc_client: Optional[triton_grpc.InferenceServerClient] = None
        self.model_metadata: Dict[str, Any] = {}
        
    def connect(self) -> bool:
        """Initialize connections to Triton."""
        try:
            # HTTP client (for metadata, health checks)
            self.http_client = triton_http.InferenceServerClient(
                url=TRITON_HTTP_URL.replace("http://", ""),
                verbose=False,
                concurrency=10,
            )
            # gRPC client (for inference — faster)
            self.grpc_client = triton_grpc.InferenceServerClient(
                url=TRITON_GRPC_URL,
                verbose=False,
            )
            logger.info(f"✅ Connected to Triton — HTTP: {TRITON_HTTP_URL}, gRPC: {TRITON_GRPC_URL}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Triton: {e}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if Triton server is ready."""
        if not self.http_client:
            return False
        try:
            return self.http_client.is_server_ready() and self.http_client.is_model_ready(MODEL_NAME)
        except Exception as e:
            logger.warning(f"Triton health check failed: {e}")
            return False
    
    def get_model_metadata(self, model_name: str, model_version: str = "") -> Dict:
        """Fetch model metadata from Triton."""
        cache_key = f"{model_name}:{model_version}"
        if cache_key in self.model_metadata:
            return self.model_metadata[cache_key]
        
        try:
            meta = self.http_client.get_model_metadata(model_name, model_version)
            self.model_metadata[cache_key] = meta
            return meta
        except Exception as e:
            logger.error(f"Failed to get metadata for {cache_key}: {e}")
            return {}
    
    def predict(
        self,
        inputs: np.ndarray,
        model_name: str,
        model_version: str = "",
    ) -> np.ndarray:
        """Run inference via gRPC (faster than HTTP)."""
        if not self.grpc_client:
            raise RuntimeError("gRPC client not initialized")
        
        # Build input tensor
        input_name = "input"  # Default; ideally fetched from model metadata
        triton_input = triton_grpc.InferInput(input_name, inputs.shape, "FP32")
        triton_input.set_data_from_numpy(inputs.astype(np.float32))
        
        # Run inference
        response = self.grpc_client.infer(
            model_name=model_name,
            inputs=[triton_input],
            model_version=model_version,
        )
        
        # Extract output (assuming single output named "output")
        output_name = "output"
        return response.as_numpy(output_name)


# Global client instance
triton = TritonClient()


# =============================================================================
# LIFESPAN (startup/shutdown)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("🚀 Gateway starting up...")
    connected = triton.connect()
    TRITON_HEALTH.set(1 if connected else 0)
    
    if connected:
        meta = triton.get_model_metadata(MODEL_NAME, MODEL_VERSION)
        logger.info(f"📋 Model metadata: {meta.get('name', MODEL_NAME)}")
    else:
        logger.warning("⚠️  Triton not available — gateway will retry on requests")
    
    yield
    
    # Shutdown
    logger.info("🛑 Gateway shutting down...")


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="Tensor Pipeline — Inference Gateway",
    description="Production-ready REST API for NVIDIA Triton Inference Server",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Catch-all exception handler."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).dict(),
    )


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Liveness probe — returns 200 if gateway is running."""
    triton_ok = triton.is_healthy()
    TRITON_HEALTH.set(1 if triton_ok else 0)
    
    return HealthResponse(
        status="healthy",
        triton_connected=triton_ok,
        model_loaded=triton_ok,
        timestamp=time.time(),
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe — returns 200 only if Triton is ready."""
    if not triton.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton inference server not ready"
        )
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from starlette.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model/info", tags=["Model"])
async def model_info(
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
):
    """Get model metadata from Triton."""
    name = model_name or MODEL_NAME
    version = model_version or MODEL_VERSION
    
    if not triton.is_healthy():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Triton not available"
        )
    
    meta = triton.get_model_metadata(name, version)
    return {
        "model_name": name,
        "model_version": version,
        "metadata": meta,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Single prediction endpoint.
    
    Example request:
    {
      "inputs": [[0.1, 0.2, 0.3, ...]],  // your feature vector
      "model_name": "resnet50",           // optional override
      "model_version": "1"                // optional override
    }
    """
    request_id = f"req_{int(time.time() * 1000)}"
    model_name = request.model_name or MODEL_NAME
    model_version = request.model_version or MODEL_VERSION
    batch_size = len(request.inputs)
    
    ACTIVE_REQUESTS.labels(model_name=model_name).inc()
    start_time = time.perf_counter()
    
    try:
        # Validate Triton connection
        if not triton.is_healthy():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Triton inference server is not ready"
            )
        
        # Convert inputs to numpy
        inputs_np = np.array(request.inputs, dtype=np.float32)
        
        # Run inference
        predictions = triton.predict(inputs_np, model_name, model_version)
        
        # Calculate latency
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Record metrics
        PREDICTION_LATENCY.labels(
            model_name=model_name,
            batch_size=str(batch_size)
        ).observe(inference_time / 1000)
        PREDICTION_COUNTER.labels(
            model_name=model_name,
            status="success"
        ).inc()
        
        logger.info(
            f"✅ Prediction | model={model_name} | batch={batch_size} | "
            f"latency={inference_time:.2f}ms | req={request_id}"
        )
        
        return PredictResponse(
            success=True,
            model_name=model_name,
            model_version=model_version,
            predictions=predictions.tolist(),
            inference_time_ms=round(inference_time, 2),
            batch_size=batch_size,
            request_id=request_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_COUNTER.labels(
            model_name=model_name,
            status="error"
        ).inc()
        logger.error(f"❌ Prediction failed | req={request_id} | error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )
    finally:
        ACTIVE_REQUESTS.labels(model_name=model_name).dec()


@app.post("/predict/batch", response_model=PredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest):
    """
    Batch prediction endpoint.
    
    Example request:
    {
      "inputs": [
        [[0.1, 0.2, ...]],   // sample 1
        [[0.3, 0.4, ...]],   // sample 2
        ...
      ]
    }
    """
    request_id = f"batch_{int(time.time() * 1000)}"
    model_name = request.model_name or MODEL_NAME
    model_version = request.model_version or MODEL_VERSION
    batch_size = len(request.inputs)
    
    ACTIVE_REQUESTS.labels(model_name=model_name).inc()
    start_time = time.perf_counter()
    
    try:
        if not triton.is_healthy():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Triton inference server is not ready"
            )
        
        # Stack batch into single numpy array
        inputs_np = np.array(request.inputs, dtype=np.float32)
        
        # Triton handles dynamic batching internally if configured
        predictions = triton.predict(inputs_np, model_name, model_version)
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        PREDICTION_LATENCY.labels(
            model_name=model_name,
            batch_size=str(batch_size)
        ).observe(inference_time / 1000)
        PREDICTION_COUNTER.labels(
            model_name=model_name,
            status="success"
        ).inc()
        
        logger.info(
            f"✅ Batch prediction | model={model_name} | batch={batch_size} | "
            f"latency={inference_time:.2f}ms | req={request_id}"
        )
        
        return PredictResponse(
            success=True,
            model_name=model_name,
            model_version=model_version,
            predictions=predictions.tolist(),
            inference_time_ms=round(inference_time, 2),
            batch_size=batch_size,
            request_id=request_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_COUNTER.labels(
            model_name=model_name,
            status="error"
        ).inc()
        logger.error(f"❌ Batch prediction failed | req={request_id} | error={e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )
    finally:
        ACTIVE_REQUESTS.labels(model_name=model_name).dec()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        log_level=LOG_LEVEL.lower(),
        reload=False,  # Never reload in production
        workers=1,     # Scale via replicas, not workers
    )
