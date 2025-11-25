"""Export embedding model to ONNX format for faster loading and Vercel compatibility."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Model configuration
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "embedding-model"


def export_model():
    """Export the embedding model to ONNX format."""
    
    logger.info(f"Exporting model: {MODEL_NAME}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        logger.info("Loading and converting model to ONNX (this may take a few minutes)...")
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
        
        logger.info("Saving tokenizer...")
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        logger.info("Saving ONNX model...")
        model.save_pretrained(OUTPUT_DIR)
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        logger.success(f"Model exported successfully!")
        logger.info(f"Location: {OUTPUT_DIR}")
        logger.info(f"Total size: {size_mb:.1f} MB")
        
        # List files
        logger.info("Files:")
        for f in sorted(OUTPUT_DIR.rglob("*")):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.name}: {size:.1f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        return False


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")
    
    success = export_model()
    sys.exit(0 if success else 1)
