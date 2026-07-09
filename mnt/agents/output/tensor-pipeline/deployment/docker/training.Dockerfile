# =============================================================================
# Tensor Pipeline — Training Service Dockerfile
# =============================================================================
FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir mlflow tensorboard

# Copy training code
COPY src/training/ .

# Default: run training
CMD ["python", "train.py"]
