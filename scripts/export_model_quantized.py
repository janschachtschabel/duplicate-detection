"""Export embedding model to quantized ONNX format (smaller size for Vercel)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "embedding-model"


def export_quantized_model():
    """Export the embedding model to quantized ONNX format."""
    
    logger.info(f"Exporting quantized model: {MODEL_NAME}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoTokenizer
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        logger.info("Loading model and converting to ONNX...")
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
        
        # Save unquantized first
        temp_dir = OUTPUT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        
        logger.info("Applying dynamic quantization (int8)...")
        quantizer = ORTQuantizer.from_pretrained(temp_dir)
        
        # Use dynamic quantization for smaller size
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        
        quantizer.quantize(
            save_dir=OUTPUT_DIR,
            quantization_config=qconfig,
        )
        
        # Copy tokenizer files
        logger.info("Copying tokenizer files...")
        for f in temp_dir.glob("*.json"):
            if "model" not in f.name.lower():
                import shutil
                shutil.copy(f, OUTPUT_DIR / f.name)
        
        # Clean up temp
        import shutil
        shutil.rmtree(temp_dir)
        
        # Rename quantized model
        quantized_model = OUTPUT_DIR / "model_quantized.onnx"
        if quantized_model.exists():
            target = OUTPUT_DIR / "model.onnx"
            if target.exists():
                target.unlink()
            quantized_model.rename(target)
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        logger.success(f"Quantized model exported successfully!")
        logger.info(f"Location: {OUTPUT_DIR}")
        logger.info(f"Total size: {size_mb:.1f} MB")
        
        for f in sorted(OUTPUT_DIR.rglob("*")):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.name}: {size:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to export quantized model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")
    
    success = export_quantized_model()
    sys.exit(0 if success else 1)
