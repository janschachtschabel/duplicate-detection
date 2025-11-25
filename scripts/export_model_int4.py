"""Export embedding model with INT4 quantization (<100MB for GitHub)."""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "embedding-model"


def export_int4_model():
    """Export the embedding model with INT4 quantization."""
    
    logger.info(f"Exporting INT4 quantized model: {MODEL_NAME}")
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        logger.info("Loading and converting model to ONNX...")
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
        
        # Save unquantized first
        temp_dir = OUTPUT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Copy config
        shutil.copy(temp_dir / "config.json", OUTPUT_DIR / "config.json")
        
        input_model = temp_dir / "model.onnx"
        output_model = OUTPUT_DIR / "model.onnx"
        
        # Try different quantization approaches
        logger.info("Attempting INT4 quantization...")
        
        try:
            # Method 1: Try onnxruntime MatMul4BitsQuantizer (newest)
            from onnxruntime.quantization import matmul_4bits_quantizer
            
            logger.info("Using MatMul 4-bit quantizer...")
            quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
                model=str(input_model),
                accuracy_level=4,
                block_size=32,
            )
            quant.process()
            quant.model.save_model_to_file(str(output_model), use_external_data_format=False)
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"MatMul4BitsQuantizer not available: {e}")
            
            try:
                # Method 2: Try quantize_dynamic with QInt4
                from onnxruntime.quantization import quantize_dynamic, QuantType
                
                logger.info("Trying QInt4 dynamic quantization...")
                quantize_dynamic(
                    model_input=str(input_model),
                    model_output=str(output_model),
                    weight_type=QuantType.QInt4
                )
            except Exception as e2:
                logger.warning(f"QInt4 not available: {e2}")
                
                # Method 3: Fallback to INT8 with optimization
                logger.info("Falling back to optimized INT8...")
                from onnxruntime.quantization import quantize_dynamic, QuantType
                import onnx
                from onnxruntime.transformers import optimizer
                
                # First optimize the model
                logger.info("Optimizing model graph...")
                optimized_model = temp_dir / "model_optimized.onnx"
                
                opt_model = optimizer.optimize_model(
                    str(input_model),
                    model_type='bert',
                    opt_level=2,
                    use_gpu=False
                )
                opt_model.save_model_to_file(str(optimized_model))
                
                # Then quantize
                logger.info("Quantizing optimized model...")
                quantize_dynamic(
                    model_input=str(optimized_model),
                    model_output=str(output_model),
                    weight_type=QuantType.QUInt8
                )
        
        # Clean up temp
        shutil.rmtree(temp_dir)
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        model_size = (OUTPUT_DIR / "model.onnx").stat().st_size / (1024 * 1024)
        
        logger.success(f"Model exported!")
        logger.info(f"Total size: {size_mb:.1f} MB")
        logger.info(f"Model file: {model_size:.1f} MB")
        
        if model_size < 100:
            logger.success(f"✅ Model is under 100MB - GitHub compatible!")
        else:
            logger.warning(f"⚠️ Model is still {model_size:.1f}MB (over 100MB)")
        
        for f in sorted(OUTPUT_DIR.rglob("*")):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.name}: {size:.1f} MB")
        
        return model_size < 100
        
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")
    
    success = export_int4_model()
    sys.exit(0 if success else 1)
