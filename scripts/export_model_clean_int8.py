"""Export embedding model with clean INT8 (no duplicate weights)."""

import os
import sys
import shutil
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "embedding-model"


def analyze_model(model_path: str, name: str):
    """Analyze ONNX model data types."""
    import onnx
    m = onnx.load(model_path)
    types = Counter([init.data_type for init in m.graph.initializer])
    
    type_names = {1: "FLOAT32", 10: "FLOAT16", 3: "INT8", 2: "UINT8", 6: "INT32", 7: "INT64"}
    
    total_bytes = 0
    for init in m.graph.initializer:
        import numpy as np
        total_bytes += np.prod(init.dims) * {1: 4, 10: 2, 3: 1, 2: 1, 6: 4, 7: 8}.get(init.data_type, 4)
    
    logger.info(f"{name} - Initializers: {total_bytes / (1024*1024):.1f} MB of weights")
    for dtype, count in types.items():
        type_name = type_names.get(dtype, f"TYPE_{dtype}")
        logger.info(f"  {type_name}: {count} tensors")


def export_clean_int8():
    """Export with proper INT8 quantization."""
    
    logger.info(f"Exporting clean INT8 model: {MODEL_NAME}")
    
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import onnx
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        logger.info("Exporting to ONNX...")
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
        
        temp_dir = OUTPUT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        model.save_pretrained(temp_dir)
        shutil.copy(temp_dir / "config.json", OUTPUT_DIR / "config.json")
        
        fp32_path = temp_dir / "model.onnx"
        fp32_size = fp32_path.stat().st_size / (1024 * 1024)
        logger.info(f"FP32: {fp32_size:.1f} MB")
        analyze_model(str(fp32_path), "FP32")
        
        # Quantize with QInt8 (signed, often better compression)
        logger.info("Quantizing to INT8...")
        int8_path = OUTPUT_DIR / "model.onnx"
        
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,  # Signed INT8
            extra_options={
                'DefaultTensorType': 1,  # FLOAT
            }
        )
        
        int8_size = int8_path.stat().st_size / (1024 * 1024)
        logger.info(f"INT8: {int8_size:.1f} MB")
        analyze_model(str(int8_path), "INT8")
        
        # Check for duplicate weights
        m = onnx.load(str(int8_path))
        float_count = sum(1 for i in m.graph.initializer if i.data_type == 1)
        int8_count = sum(1 for i in m.graph.initializer if i.data_type == 3)
        logger.info(f"Remaining FLOAT32 tensors: {float_count}")
        logger.info(f"INT8 tensors: {int8_count}")
        
        shutil.rmtree(temp_dir)
        
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
        
        logger.info(f"="*50)
        logger.info(f"Model file: {int8_size:.1f} MB")
        logger.info(f"Total folder: {total_size / (1024*1024):.1f} MB")
        
        if int8_size < 100:
            logger.success(f"✅ Under 100MB!")
        else:
            logger.warning(f"⚠️ Still {int8_size:.1f} MB - model has 118M params")
            logger.info("Options: Git LFS or smaller model")
        
        return int8_size < 100
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")
    export_clean_int8()
