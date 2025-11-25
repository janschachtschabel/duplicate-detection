"""Embedding-based duplicate detection using ONNX runtime (Vercel-compatible)."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from app.config import detection_config
from app.models import ContentMetadata, DuplicateCandidate

# Check if ONNX runtime is available
ONNX_AVAILABLE = False
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer
    ONNX_AVAILABLE = True
except ImportError:
    logger.warning("optimum/onnxruntime not installed. Embedding detection will be disabled.")

# Local model path (exported via scripts/export_model.py)
LOCAL_MODEL_PATH = Path(__file__).parent.parent / "models" / "embedding-model"

# Lazy load model to avoid startup delay
_model = None
_tokenizer = None
_model_name = None


def is_embedding_available() -> bool:
    """Check if embedding detection is available."""
    return ONNX_AVAILABLE


def is_local_model_available() -> bool:
    """Check if local ONNX model is available."""
    return LOCAL_MODEL_PATH.exists() and (LOCAL_MODEL_PATH / "model.onnx").exists()


def get_current_model_name() -> str:
    """Get the name of the currently configured/loaded model."""
    if _model_name:
        # Check if it's a local path (contains backslash or models folder)
        if str(LOCAL_MODEL_PATH) in _model_name or "models" in _model_name or "\\" in _model_name:
            # For local models, read the model name from model_info.json
            try:
                import json
                info_path = LOCAL_MODEL_PATH / "model_info.json"
                if info_path.exists():
                    with open(info_path) as f:
                        info = json.load(f)
                        return info.get("model_name", "local-embedding-model")
            except:
                pass
            # Fallback: use vercel model name from config (that's what's exported locally)
            return detection_config.embedding_config.vercel_model.split("/")[-1]
        # Return loaded model name (HuggingFace path)
        return _model_name.split("/")[-1] if "/" in _model_name else _model_name
    # Return configured model name
    return detection_config.embedding_model_name


def get_embedding_model():
    """Get or load the ONNX embedding model (lazy initialization)."""
    global _model, _tokenizer, _model_name
    
    if not ONNX_AVAILABLE:
        raise RuntimeError(
            "Embedding detection is not available. "
            "optimum/onnxruntime is not installed. "
            "Use hash-based detection instead."
        )
    
    # Check if local model exists (faster loading, used for Vercel deployment)
    if is_local_model_available():
        model_path = str(LOCAL_MODEL_PATH)
        model_source = "local"
    else:
        # Use configured model (depends on environment: Vercel vs local)
        model_path = detection_config.embedding_model
        model_source = "huggingface"
    
    if _model is None or _model_name != model_path:
        try:
            logger.info(f"Loading ONNX embedding model from {model_source}: {model_path}")
            logger.info(f"Configured model: {detection_config.embedding_model_name}")
            _tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if model_source == "local":
                # Load from local without export
                _model = ORTModelForFeatureExtraction.from_pretrained(model_path)
            else:
                # Download and export from HuggingFace
                _model = ORTModelForFeatureExtraction.from_pretrained(model_path, export=True)
            
            _model_name = model_path
            logger.info(f"ONNX embedding model loaded successfully ({model_source})")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model {model_path}: {e}")
    
    return _model, _tokenizer


def is_model_loaded() -> bool:
    """Check if the embedding model is loaded."""
    return _model is not None


class EmbeddingDetector:
    """Embedding-based duplicate detection using ONNX runtime."""
    
    def __init__(self):
        """Initialize embedding detector."""
        self.model = None
        self.tokenizer = None
        logger.info("Embedding detector initialized (ONNX model will be loaded on first use)")
    
    def _ensure_model(self):
        """Ensure the model is loaded."""
        if self.model is None:
            self.model, self.tokenizer = get_embedding_model()
    
    def _mean_pooling(self, model_output, attention_mask) -> np.ndarray:
        """Apply mean pooling to get sentence embedding."""
        token_embeddings = model_output[0]  # First element is token embeddings
        
        # Convert to numpy if needed
        if hasattr(token_embeddings, 'numpy'):
            token_embeddings = token_embeddings.numpy()
        if hasattr(attention_mask, 'numpy'):
            attention_mask = attention_mask.numpy()
        
        input_mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1),
            token_embeddings.shape
        ).astype(float)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        
        return sum_embeddings / sum_mask
    
    def compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Compute embedding for text using ONNX model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if text is empty
        """
        if not text or not text.strip():
            return None
        
        self._ensure_model()
        
        try:
            # Truncate very long texts
            if len(text) > 10000:
                text = text[:10000]
            
            # Tokenize
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="np")
            
            # Run through ONNX model
            outputs = self.model(**inputs)
            
            # Mean pooling
            embedding = self._mean_pooling(outputs, inputs["attention_mask"])
            
            return embedding[0]  # Return first (and only) embedding
        except Exception as e:
            logger.error(f"Failed to compute embedding: {e}")
            return None
    
    def compute_metadata_embedding(self, metadata: ContentMetadata) -> Optional[np.ndarray]:
        """
        Compute embedding from content metadata.
        
        Args:
            metadata: Content metadata
            
        Returns:
            Embedding vector or None if no content
        """
        text = metadata.get_searchable_text()
        return self.compute_embedding(text)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Reshape for sklearn
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def batch_compute_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Compute embeddings for multiple texts using ONNX model.
        
        Args:
            texts: List of texts
            
        Returns:
            List of embeddings (None for empty texts)
        """
        self._ensure_model()
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                # Truncate very long texts
                if len(text) > 10000:
                    text = text[:10000]
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return [None] * len(texts)
        
        try:
            # Tokenize all texts at once
            inputs = self.tokenizer(valid_texts, padding=True, truncation=True, max_length=512, return_tensors="np")
            
            # Run through ONNX model
            outputs = self.model(**inputs)
            
            # Mean pooling
            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
            
            # Build result list
            result = [None] * len(texts)
            for idx, emb in zip(valid_indices, embeddings):
                result[idx] = emb
            
            return result
        except Exception as e:
            logger.error(f"Failed to compute batch embeddings: {e}")
            return [None] * len(texts)
    
    def _is_valid_field(self, value) -> bool:
        """Check if a field value is valid (not empty or placeholder)."""
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() not in {"string", ""} and len(value.strip()) > 0
        if isinstance(value, list):
            valid = [v for v in value if v and str(v).strip().lower() != "string"]
            return len(valid) > 0
        return False
    
    def find_duplicates(
        self,
        source_metadata: ContentMetadata,
        candidates: Dict[str, List[Dict[str, Any]]],
        threshold: float = None
    ) -> tuple[List[DuplicateCandidate], Dict[str, float]]:
        """
        Find duplicates among candidates using embeddings.
        Compares only the same fields that are present in source_metadata.
        
        Args:
            source_metadata: Source content metadata
            candidates: Dict of search_field -> candidate nodes
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple of:
            - List of duplicate candidates above threshold
            - Dict of field -> highest similarity score
        """
        threshold = threshold or detection_config.default_embedding_threshold
        
        # Determine which fields are available in source
        has_title = self._is_valid_field(source_metadata.title)
        has_description = self._is_valid_field(source_metadata.description)
        has_keywords = self._is_valid_field(source_metadata.keywords)
        
        # Build source text from available fields only
        source_parts = []
        if has_title:
            source_parts.append(source_metadata.title)
        if has_description:
            source_parts.append(source_metadata.description)
        if has_keywords:
            valid_kw = [k for k in source_metadata.keywords if k and k.strip().lower() != "string"]
            source_parts.extend(valid_kw)
        
        source_text = " ".join(source_parts)
        source_emb = self.compute_embedding(source_text)
        
        if source_emb is None:
            logger.warning("Could not compute embedding for source metadata")
            return [], {}
        
        logger.info(f"Source fields: title={has_title}, description={has_description}, keywords={has_keywords}")
        
        # Collect all candidates per field
        field_candidates_data: Dict[str, list] = {}
        seen_ids_global = set()
        
        for search_field, field_candidates in candidates.items():
            field_candidates_data[search_field] = []
            
            for candidate in field_candidates:
                node_id = candidate.get("ref", {}).get("id")
                if not node_id:
                    continue
                
                properties = candidate.get("properties", {})
                
                # Extract all fields for output
                title = None
                for key in ["cclom:title", "cm:name"]:
                    if key in properties:
                        val = properties[key]
                        title = val[0] if isinstance(val, list) else val
                        break
                
                description = None
                for key in ["cclom:general_description"]:
                    if key in properties:
                        val = properties[key]
                        description = val[0] if isinstance(val, list) else val
                        break
                
                keywords = None
                if "cclom:general_keyword" in properties:
                    kw = properties["cclom:general_keyword"]
                    keywords = kw if isinstance(kw, list) else [kw]
                
                url = None
                for key in ["ccm:wwwurl", "cclom:location"]:
                    if key in properties:
                        val = properties[key]
                        url = val[0] if isinstance(val, list) else val
                        break
                
                # Build candidate text from SAME fields as source
                candidate_parts = []
                if has_title and title:
                    candidate_parts.append(title)
                if has_description and description:
                    candidate_parts.append(description)
                if has_keywords and keywords:
                    candidate_parts.extend(keywords)
                
                candidate_text = " ".join(candidate_parts)
                
                field_candidates_data[search_field].append((node_id, title, description, keywords, url, candidate_text))
        
        # Collect all unique texts for batch embedding
        all_texts = []
        text_to_idx = {}
        for field, items in field_candidates_data.items():
            for node_id, title, description, keywords, url, text in items:
                if text not in text_to_idx:
                    text_to_idx[text] = len(all_texts)
                    all_texts.append(text)
        
        if not all_texts:
            return [], {}
        
        # Batch compute embeddings
        logger.info(f"Computing embeddings for {len(all_texts)} unique candidates")
        all_embeddings = self.batch_compute_embeddings(all_texts)
        
        # Compute per-field max similarity and collect duplicates
        duplicates = []
        field_max_similarity: Dict[str, float] = {}
        
        for search_field, items in field_candidates_data.items():
            field_max = 0.0
            
            for node_id, title, description, keywords, url, text in items:
                idx = text_to_idx.get(text)
                if idx is None:
                    continue
                emb = all_embeddings[idx]
                if emb is None:
                    continue
                
                similarity = self.compute_similarity(source_emb, emb)
                
                # Track max for this field
                if similarity > field_max:
                    field_max = similarity
                
                # Only add to duplicates if not already seen
                if node_id not in seen_ids_global and similarity >= threshold:
                    seen_ids_global.add(node_id)
                    duplicates.append(DuplicateCandidate(
                        node_id=node_id,
                        title=title,
                        description=description,
                        keywords=keywords,
                        url=url,
                        similarity_score=round(similarity, 4),
                        match_source=search_field
                    ))
            
            # Store max similarity for this field
            if items:
                field_max_similarity[search_field] = round(field_max, 4)
        
        # Sort by similarity (highest first)
        duplicates.sort(key=lambda x: x.similarity_score, reverse=True)
        
        logger.info(f"Found {len(duplicates)} embedding-based duplicates above threshold {threshold}")
        return duplicates, field_max_similarity


# Global instance
embedding_detector = EmbeddingDetector()
