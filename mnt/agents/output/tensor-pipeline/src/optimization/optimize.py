#!/usr/bin/env python3
"""
Tensor Pipeline — Optimization Service
Converts SavedModel → ONNX/TensorRT for Triton Inference Server.
"""
import os
import sys
import shutil

SAVED_MODEL_DIR = os.getenv("SAVED_MODEL_DIR", "/app/artifacts/saved_model")
TRITON_REPO = os.getenv("TRITON_MODEL_REPO", "/app/triton_repo")
MODEL_NAME = os.getenv("MODEL_NAME", "resnet50")
OPTIMIZE_TRT = os.getenv("OPTIMIZE_TRT", "false").lower() == "true"
OPTIMIZE_ONNX = os.getenv("OPTIMIZE_ONNX", "true").lower() == "true"


def setup_triton_model_repo():
    """Create Triton model repository structure."""
    model_dir = os.path.join(TRITON_REPO, MODEL_NAME)
    version_dir = os.path.join(model_dir, "1")
    os.makedirs(version_dir, exist_ok=True)
    return model_dir, version_dir


def convert_to_onnx(saved_model_dir, output_path):
    """Convert SavedModel to ONNX."""
    import tf2onnx
    import tensorflow as tf
    
    print(f"🔄 Converting to ONNX: {output_path}")
    
    model = tf.saved_model.load(saved_model_dir)
    spec = [tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input")]
    
    tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=output_path,
    )
    print(f"✅ ONNX model saved: {output_path}")


def write_triton_config(model_dir, backend="onnxruntime"):
    """Write Triton model configuration."""
    config = f"""
name: "{MODEL_NAME}"
platform: "{backend}"
max_batch_size: 8
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [224, 224, 3]
  }}
]
output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [10]
  }}
]
dynamic_batching {{
  preferred_batch_size: [2, 4, 8]
  max_queue_delay_microseconds: 100
}}
"""
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config)
    print(f"✅ Triton config written: {config_path}")


def main():
    print("🔧 Starting optimization pipeline...")
    
    if not os.path.exists(SAVED_MODEL_DIR):
        print(f"❌ SavedModel not found at {SAVED_MODEL_DIR}")
        sys.exit(1)
    
    model_dir, version_dir = setup_triton_model_repo()
    
    if OPTIMIZE_ONNX:
        onnx_path = os.path.join(version_dir, "model.onnx")
        convert_to_onnx(SAVED_MODEL_DIR, onnx_path)
        write_triton_config(model_dir, platform="onnxruntime")
    
    if OPTIMIZE_TRT:
        print("⚠️  TensorRT optimization requires GPU — skipping in stub")
        # TODO: Add TensorRT conversion here
    
    print(f"✅ Optimization complete. Triton repo: {TRITON_REPO}")
    print(f"   Start Triton with: tritonserver --model-repository={TRITON_REPO}")


if __name__ == "__main__":
    main()
