# =============================================================================
# Tensor Pipeline — Optimization Service Dockerfile
# =============================================================================
FROM nvcr.io/nvidia/tensorrt:24.05-py3

WORKDIR /app

# Install ONNX dependencies
RUN pip install --no-cache-dir onnx onnxruntime-gpu tf2onnx

# Copy optimization scripts
COPY src/optimization/ .

CMD ["python", "optimize.py"]
