"""Export the final model for production use."""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# New smaller model (de, en, es, fr, it, nl, pl, pt)
MODEL_NAME = "h4g3n/multilingual-MiniLM-L12-de-en-es-fr-it-nl-pl-pt"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "embedding-model"


def export_model():
    """Export the model with INT8 quantization."""
    
    logger.info(f"Exporting model: {MODEL_NAME}")
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        logger.info(f"Removing existing model at {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
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
        
        # Quantize to INT8
        logger.info("Quantizing to INT8...")
        int8_path = OUTPUT_DIR / "model.onnx"
        
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8
        )
        
        int8_size = int8_path.stat().st_size / (1024 * 1024)
        
        # Cleanup temp
        shutil.rmtree(temp_dir)
        
        # Save model info
        import json
        model_info = {
            "model_name": MODEL_NAME.split("/")[-1],
            "model_id": MODEL_NAME,
            "quantization": "INT8",
            "exported_by": "export_final_model.py"
        }
        with open(OUTPUT_DIR / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"Saved model_info.json")
        
        # Show results
        total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
        
        logger.info("="*50)
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"INT8 ONNX: {int8_size:.1f} MB")
        logger.info(f"Total folder: {total/(1024*1024):.1f} MB")
        logger.info("="*50)
        
        logger.info("Files:")
        for f in sorted(OUTPUT_DIR.rglob("*")):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.name}: {size:.2f} MB")
        
        if int8_size < 100:
            logger.success(f"✅ Ready for GitHub (model.onnx = {int8_size:.1f} MB)")
        else:
            logger.warning(f"⚠️ Still over 100 MB")
        
        return int8_size < 100
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")
    export_model()
