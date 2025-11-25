"""Export embedding model with heavy optimization (<100MB for GitHub)."""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "embedding-model"


def export_optimized_model():
    """Export heavily optimized model."""
    
    logger.info(f"Exporting optimized model: {MODEL_NAME}")
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        logger.info("Loading and converting model to ONNX...")
        model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
        
        # Save unquantized first
        temp_dir = OUTPUT_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(OUTPUT_DIR)
        shutil.copy(temp_dir / "config.json", OUTPUT_DIR / "config.json")
        
        input_model = temp_dir / "model.onnx"
        optimized_model = temp_dir / "model_opt.onnx"
        output_model = OUTPUT_DIR / "model.onnx"
        
        # Step 1: Optimize the graph
        logger.info("Step 1: Optimizing model graph (fusion, pruning)...")
        
        fusion_options = FusionOptions('bert')
        fusion_options.enable_skip_layer_norm = True
        fusion_options.enable_bias_skip_layer_norm = True
        fusion_options.enable_gelu_approximation = True
        
        opt_model = optimizer.optimize_model(
            str(input_model),
            model_type='bert',
            opt_level=99,  # Maximum optimization
            use_gpu=False,
            only_onnxruntime=True,
            optimization_options=fusion_options
        )
        opt_model.save_model_to_file(str(optimized_model))
        
        opt_size = optimized_model.stat().st_size / (1024 * 1024)
        logger.info(f"After optimization: {opt_size:.1f} MB")
        
        # Step 2: Quantize to INT8
        logger.info("Step 2: Quantizing to INT8...")
        
        quantize_dynamic(
            model_input=str(optimized_model),
            model_output=str(output_model),
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=True  # More aggressive quantization
        )
        
        # Clean up temp
        shutil.rmtree(temp_dir)
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        model_size = (OUTPUT_DIR / "model.onnx").stat().st_size / (1024 * 1024)
        
        logger.info(f"="*50)
        logger.info(f"Total size: {size_mb:.1f} MB")
        logger.info(f"Model file: {model_size:.1f} MB")
        
        if model_size < 100:
            logger.success(f"✅ Model is under 100MB - GitHub compatible!")
        else:
            logger.warning(f"⚠️ Model is still {model_size:.1f}MB")
            logger.info("The model cannot be reduced below 100MB with standard quantization.")
            logger.info("Consider using Git LFS or HuggingFace Inference API.")
        
        for f in sorted(OUTPUT_DIR.rglob("*")):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                logger.info(f"  {f.name}: {size:.1f} MB")
        
        return model_size < 100
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")
    
    success = export_optimized_model()
    sys.exit(0 if success else 1)
