#!/usr/bin/env python3
"""
Tensor Pipeline — TensorRT Optimizer
Converts TensorFlow SavedModel to TensorRT engine for NVIDIA GPU inference.
Supports FP16 and INT8 precision with calibration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import tensorflow as tf
import numpy as np

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("tensorrt")


def check_gpu_available():
    """Check if NVIDIA GPU is available and return compute capability."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.error("❌ No GPU found. TensorRT requires NVIDIA GPU.")
        return False, None
    
    # Get GPU details
    gpu_details = tf.config.experimental.get_device_details(gpus[0])
    compute_capability = gpu_details.get('compute_capability', (0, 0))
    
    logger.info(f"✅ GPU found: {gpu_details}")
    logger.info(f"   Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    
    # TensorRT requirements
    min_cc = (5, 3)  # sm_53 minimum
    if compute_capability < min_cc:
        logger.warning(f"⚠️  Compute Capability {compute_capability} < {min_cc}. TensorRT may not work.")
        return False, compute_capability
    
    return True, compute_capability


def convert_to_tensorrt_fp16(
    saved_model_dir: str,
    output_dir: str,
    input_shape: tuple = (1, 224, 224, 3),
    max_batch_size: int = 8,
    max_workspace_size: int = 1 << 30,  # 1GB
) -> str:
    """
    Convert SavedModel to TensorRT FP16.
    
    Args:
        saved_model_dir: Path to SavedModel directory
        output_dir: Output directory for TensorRT plan
        input_shape: Input tensor shape (batch, H, W, C)
        max_batch_size: Maximum batch size for dynamic batching
        max_workspace_size: Maximum workspace size in bytes
    
    Returns:
        Path to saved TensorRT plan
    """
    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
    except ImportError:
        logger.error("❌ TensorRT not available in this TensorFlow build.")
        logger.error("   Use NVIDIA TensorRT container: nvcr.io/nvidia/tensorflow:xx.xx-py3")
        sys.exit(1)
    
    logger.info("🔄 Starting TensorRT FP16 conversion...")
    logger.info(f"   Input: {saved_model_dir}")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Precision: FP16")
    logger.info(f"   Max batch size: {max_batch_size}")
    
    # Conversion parameters
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.FP16,
        max_workspace_size_bytes=max_workspace_size,
        maximum_cached_engines=100,
        minimum_segment_size=3,
        use_calibration=False,
    )
    
    # Convert
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_dir,
        conversion_params=params,
    )
    
    logger.info("⚙️  Building TensorRT engines (this may take a while)...")
    converter.convert()
    
    # Build with sample input for optimization
    def input_fn():
        for _ in range(1):
            yield [np.random.normal(size=input_shape).astype(np.float32)]
    
    converter.build(input_fn=input_fn)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    converter.save(output_dir)
    
    logger.info(f"✅ TensorRT FP16 model saved: {output_dir}")
    
    # Print model info
    saved_model = tf.saved_model.load(output_dir)
    logger.info(f"   Signatures: {list(saved_model.signatures.keys())}")
    
    return output_dir


def convert_to_tensorrt_int8(
    saved_model_dir: str,
    output_dir: str,
    calibration_data: np.ndarray,
    input_shape: tuple = (1, 224, 224, 3),
    max_batch_size: int = 8,
    max_workspace_size: int = 1 << 30,
) -> str:
    """
    Convert SavedModel to TensorRT INT8 with calibration.
    INT8 requires calibration dataset for accuracy.
    
    Args:
        saved_model_dir: Path to SavedModel directory
        output_dir: Output directory
        calibration_data: Representative dataset for calibration (N, H, W, C)
        input_shape: Input tensor shape
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size
    
    Returns:
        Path to saved TensorRT plan
    """
    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
    except ImportError:
        logger.error("❌ TensorRT not available.")
        sys.exit(1)
    
    logger.info("🔄 Starting TensorRT INT8 conversion...")
    logger.info(f"   Calibration samples: {len(calibration_data)}")
    
    # Calibration input function
    def calibration_input_fn():
        batch_size = 32
        for i in range(0, len(calibration_data), batch_size):
            batch = calibration_data[i:i + batch_size]
            yield [batch.astype(np.float32)]
    
    # Conversion parameters
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        precision_mode=trt.TrtPrecisionMode.INT8,
        max_workspace_size_bytes=max_workspace_size,
        maximum_cached_engines=100,
        minimum_segment_size=3,
        use_calibration=True,
    )
    
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=saved_model_dir,
        conversion_params=params,
    )
    
    logger.info("⚙️  Calibrating INT8 (this may take a while)...")
    converter.convert(calibration_input_fn=calibration_input_fn)
    
    # Build
    def input_fn():
        yield [calibration_data[:1].astype(np.float32)]
    
    converter.build(input_fn=input_fn)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    converter.save(output_dir)
    
    logger.info(f"✅ TensorRT INT8 model saved: {output_dir}")
    
    return output_dir


def generate_calibration_data(
    num_samples: int = 500,
    shape: tuple = (224, 224, 3),
) -> np.ndarray:
    """
    Generate synthetic calibration data.
    Replace with real representative data for production!
    
    Args:
        num_samples: Number of calibration samples
        shape: Image shape (H, W, C)
    
    Returns:
        Calibration data array (N, H, W, C)
    """
    logger.info(f"🎲 Generating {num_samples} synthetic calibration samples...")
    
    # Use normal distribution as placeholder
    # In production: use real validation data from your domain!
    data = np.random.normal(loc=0.5, scale=0.2, size=(num_samples, *shape))
    data = np.clip(data, 0.0, 1.0).astype(np.float32)
    
    return data


def benchmark_tensorrt(
    trt_model_dir: str,
    input_shape: tuple = (1, 224, 224, 3),
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """
    Benchmark TensorRT model inference speed.
    
    Returns:
        Dict with latency stats
    """
    logger.info(f"🏁 Benchmarking TensorRT model...")
    
    model = tf.saved_model.load(trt_model_dir)
    infer = model.signatures['serving_default']
    
    # Warmup
    dummy_input = tf.constant(np.random.normal(size=input_shape).astype(np.float32))
    for _ in range(num_warmup):
        _ = infer(dummy_input)
    
    # Benchmark
    import time
    latencies = []
    
    for _ in range(num_runs):
        dummy_input = tf.constant(np.random.normal(size=input_shape).astype(np.float32))
        
        start = time.perf_counter()
        _ = infer(dummy_input)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # ms
    
    stats = {
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "throughput_fps": 1000 / np.mean(latencies),
    }
    
    logger.info(f"   Mean: {stats['mean_ms']:.2f}ms")
    logger.info(f"   P95: {stats['p95_ms']:.2f}ms")
    logger.info(f"   P99: {stats['p99_ms']:.2f}ms")
    logger.info(f"   Throughput: {stats['throughput_fps']:.1f} FPS")
    
    return stats


def write_triton_tensorrt_config(
    model_dir: str,
    model_name: str = "resnet50",
    max_batch_size: int = 8,
    input_name: str = "input",
    output_name: str = "output",
):
    """Write Triton config for TensorRT model."""
    config = f"""
name: "{model_name}"
platform: "tensorflow_savedmodel"
max_batch_size: {max_batch_size}
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [224, 224, 3]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [10]
  }}
]
dynamic_batching {{
  preferred_batch_size: [2, 4, 8]
  max_queue_delay_microseconds: 100
}}
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator: [
      {{
        name: "tensorrt"
        parameters {{ key: "precision_mode" value: "FP16" }}
      }}
    ]
  }}
}}
"""
    config_path = os.path.join(model_dir, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config)
    logger.info(f"✅ Triton config written: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="TensorRT Optimization")
    parser.add_argument("--saved-model", required=True, help="Path to SavedModel")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--precision", choices=["FP16", "INT8"], default="FP16", help="Precision mode")
    parser.add_argument("--batch-size", type=int, default=8, help="Max batch size")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after conversion")
    parser.add_argument("--write-triton-config", action="store_true", help="Write Triton config.pbtxt")
    
    args = parser.parse_args()
    
    # Check GPU
    gpu_ok, cc = check_gpu_available()
    if not gpu_ok:
        logger.warning("⚠️  Continuing anyway, but TensorRT may fail...")
    
    # Convert
    if args.precision == "FP16":
        output = convert_to_tensorrt_fp16(
            args.saved_model,
            args.output,
            max_batch_size=args.batch_size,
        )
    else:
        # INT8 requires calibration data
        calib_data = generate_calibration_data(num_samples=500)
        output = convert_to_tensorrt_int8(
            args.saved_model,
            args.output,
            calibration_data=calib_data,
            max_batch_size=args.batch_size,
        )
    
    # Benchmark
    if args.benchmark:
        benchmark_tensorrt(output)
    
    # Triton config
    if args.write_triton_config:
        write_triton_tensorrt_config(
            args.output,
            max_batch_size=args.batch_size,
        )
    
    logger.info("✅ TensorRT optimization complete!")


if __name__ == "__main__":
    main()
