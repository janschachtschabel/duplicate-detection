"""Export embedding model with proper FP16 conversion (<100MB for GitHub)."""

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
    
    # ONNX data type mapping
    type_names = {
        1: "FLOAT32",
        10: "FLOAT16", 
        3: "INT8",
        2: "UINT8",
        11: "DOUBLE",
        6: "INT32",
        7: "INT64"
    }
    
    logger.info(f"{name} - Data types in initializers:")
    for dtype, count in types.items():
        type_name = type_names.get(dtype, f"TYPE_{dtype}")
        logger.info(f"  {type_name}: {count} tensors")


def export_fp16_model():
    """Export the embedding model with FP16 conversion."""
    
    logger.info(f"Exporting FP16 model: {MODEL_NAME}")
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import onnx
        from onnxconverter_common import float16
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        logger.info("Loading and converting model to ONNX (FP32)...")
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
        
        # Save FP32 model
        temp_dir = OUTPUT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        model.save_pretrained(temp_dir)
        shutil.copy(temp_dir / "config.json", OUTPUT_DIR / "config.json")
        
        fp32_model_path = temp_dir / "model.onnx"
        fp32_size = fp32_model_path.stat().st_size / (1024 * 1024)
        logger.info(f"FP32 model size: {fp32_size:.1f} MB")
        analyze_model(str(fp32_model_path), "FP32")
        
        # Convert to FP16
        logger.info("Converting to FP16...")
        model_fp32 = onnx.load(str(fp32_model_path))
        model_fp16 = float16.convert_float_to_float16(
            model_fp32,
            keep_io_types=True  # Keep inputs/outputs as FP32 for compatibility
        )
        
        fp16_model_path = OUTPUT_DIR / "model.onnx"
        onnx.save(model_fp16, str(fp16_model_path))
        
        fp16_size = fp16_model_path.stat().st_size / (1024 * 1024)
        logger.info(f"FP16 model size: {fp16_size:.1f} MB")
        analyze_model(str(fp16_model_path), "FP16")
        
        # Clean up temp
        shutil.rmtree(temp_dir)
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        logger.info(f"="*50)
        logger.info(f"RESULTS:")
        logger.info(f"  FP32 → FP16: {fp32_size:.1f} MB → {fp16_size:.1f} MB")
        logger.info(f"  Reduction: {(1 - fp16_size/fp32_size)*100:.1f}%")
        logger.info(f"  Total folder size: {size_mb:.1f} MB")
        logger.info(f"="*50)
        
        if fp16_size < 100:
            logger.success(f"✅ Model is {fp16_size:.1f} MB - GitHub compatible!")
        else:
            logger.warning(f"⚠️ Model is still {fp16_size:.1f} MB")
        
        logger.info("Files:")
        for f in sorted(OUTPUT_DIR.rglob("*")):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.name}: {size:.1f} MB")
        
        return fp16_size < 100
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")
    
    success = export_fp16_model()
    sys.exit(0 if success else 1)
