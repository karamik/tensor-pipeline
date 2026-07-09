#!/usr/bin/env python3
"""
Tensor Pipeline — ONNX Converter
Standalone module for converting TensorFlow SavedModel to ONNX format.
Supports both tf2onnx (recommended) and ONNX Runtime tools.
"""

import os
import sys
import argparse
from pathlib import Path

import tensorflow as tf


def convert_savedmodel_to_onnx(
    saved_model_dir: str,
    output_path: str,
    opset: int = 13,
    input_signature: tuple = (None, 224, 224, 3),
    input_name: str = "input",
    output_name: str = "output",
) -> str:
    """
    Convert TensorFlow SavedModel to ONNX using tf2onnx.
    
    Args:
        saved_model_dir: Path to SavedModel directory
        output_path: Output ONNX file path
        opset: ONNX opset version
        input_signature: Input tensor shape (batch, height, width, channels)
        input_name: Name of input tensor
        output_name: Name of output tensor
    
    Returns:
        Path to saved ONNX model
    """
    try:
        import tf2onnx
    except ImportError:
        print("❌ tf2onnx not installed. Run: pip install tf2onnx")
        sys.exit(1)
    
    print(f"🔄 Loading SavedModel from: {saved_model_dir}")
    model = tf.saved_model.load(saved_model_dir)
    
    # Infer batch dimension if None
    batch_size = input_signature[0] if input_signature[0] is not None else None
    shape = [batch_size] + list(input_signature[1:])
    
    spec = [tf.TensorSpec(shape, tf.float32, name=input_name)]
    
    print(f"📝 Input spec: {spec}")
    print(f"📝 ONNX opset: {opset}")
    print(f"📝 Output: {output_path}")
    
    # Convert
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=opset,
        output_path=output_path,
    )
    
    # Verify
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ ONNX model saved: {output_path} ({size_mb:.2f} MB)")
        
        # Print model info
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model validation passed")
        print(f"   Inputs: {[i.name for i in onnx_model.graph.input]}")
        print(f"   Outputs: {[o.name for o in onnx_model.graph.output]}")
    else:
        raise RuntimeError("ONNX model was not created")
    
    return output_path


def convert_with_ort_tools(
    saved_model_dir: str,
    output_path: str,
) -> str:
    """
    Alternative conversion using ONNX Runtime tools.
    Useful for advanced optimizations.
    """
    from onnxruntime.tools import optimizer
    
    # First convert with tf2onnx
    temp_onnx = output_path.replace(".onnx", "_temp.onnx")
    convert_savedmodel_to_onnx(saved_model_dir, temp_onnx)
    
    # Then optimize with ONNX Runtime
    print("🔧 Applying ONNX Runtime optimizations...")
    optimized_model = optimizer.optimize_model(
        temp_onnx,
        model_type="bert" if "bert" in saved_model_dir else "onnx",
        use_gpu=True,
    )
    optimized_model.convert_model_float32_to_float16()
    optimized_model.save_model_to_file(output_path)
    
    # Cleanup
    os.remove(temp_onnx)
    print(f"✅ Optimized ONNX saved: {output_path}")
    
    return output_path


def validate_onnx_model(onnx_path: str) -> dict:
    """
    Validate ONNX model and return metadata.
    
    Returns:
        Dict with model info
    """
    import onnx
    
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    
    info = {
        "ir_version": model.ir_version,
        "opset_import": [(domain, version) for domain, version in model.opset_import],
        "inputs": [
            {
                "name": i.name,
                "shape": [d.dim_value if d.dim_value else str(d.dim_param) for d in i.type.tensor_type.shape.dim],
                "dtype": onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type),
            }
            for i in model.graph.input
        ],
        "outputs": [
            {
                "name": o.name,
                "shape": [d.dim_value if d.dim_value else str(d.dim_param) for d in o.type.tensor_type.shape.dim],
                "dtype": onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type),
            }
            for o in model.graph.output
        ],
        "num_nodes": len(model.graph.node),
        "num_initializers": len(model.graph.initializer),
    }
    
    return info


def main():
    parser = argparse.ArgumentParser(description="Convert TensorFlow SavedModel to ONNX")
    parser.add_argument("--saved-model", required=True, help="Path to SavedModel directory")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (None for dynamic)")
    parser.add_argument("--input-shape", type=int, nargs=3, default=[224, 224, 3], help="Input shape H W C")
    parser.add_argument("--validate", action="store_true", help="Validate output model")
    parser.add_argument("--optimize", action="store_true", help="Use ONNX Runtime optimizer")
    
    args = parser.parse_args()
    
    input_signature = (args.batch_size, *args.input_shape)
    
    if args.optimize:
        convert_with_ort_tools(args.saved_model, args.output)
    else:
        convert_savedmodel_to_onnx(
            args.saved_model,
            args.output,
            opset=args.opset,
            input_signature=input_signature,
        )
    
    if args.validate:
        print("\n📋 Validating ONNX model...")
        info = validate_onnx_model(args.output)
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
