"""Configuration for WLO Duplicate Detection API."""

import os
from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field


def is_vercel() -> bool:
    """Check if running on Vercel."""
    return os.environ.get("VERCEL") == "1" or os.environ.get("VERCEL_ENV") is not None


class Environment(str, Enum):
    """WLO API environments."""
    PRODUCTION = "production"
    STAGING = "staging"


class WLOConfig(BaseModel):
    """WLO API configuration."""
    
    base_urls: Dict[Environment, str] = Field(
        default={
            Environment.PRODUCTION: "https://redaktion.openeduhub.net/edu-sharing/rest",
            Environment.STAGING: "https://repository.staging.openeduhub.net/edu-sharing/rest"
        }
    )
    default_repository: str = Field(default="-home-")
    default_timeout: int = Field(default=60)
    max_retries: int = Field(default=3)
    
    def get_base_url(self, environment: Environment) -> str:
        """Get base URL for environment."""
        return self.base_urls[environment]


class EmbeddingModelConfig(BaseModel):
    """Configuration for embedding models."""
    
    # Model for Vercel deployment (must be <100MB when quantized)
    vercel_model: str = Field(
        default="h4g3n/multilingual-MiniLM-L12-de-en-es-fr-it-nl-pl-pt",
        description="Smaller model for Vercel (<100MB). Supports: de, en, es, fr, it, nl, pl, pt"
    )
    
    # Model for local deployment (can be larger, better quality)
    local_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Full multilingual model for local deployment. Supports 50+ languages"
    )
    
    # Override via environment variable (optional)
    # Set EMBEDDING_MODEL=sentence-transformers/your-model to override
    env_override: Optional[str] = Field(default=None)
    
    def get_model_id(self) -> str:
        """Get the appropriate model ID based on environment."""
        # 1. Check environment variable override
        env_model = os.environ.get("EMBEDDING_MODEL")
        if env_model:
            return env_model
        
        # 2. Check if running on Vercel
        if is_vercel():
            return self.vercel_model
        
        # 3. Default to local model
        return self.local_model
    
    def get_model_name(self) -> str:
        """Get short model name for display."""
        model_id = self.get_model_id()
        # Extract name from full path (e.g., "sentence-transformers/model-name" -> "model-name")
        return model_id.split("/")[-1] if "/" in model_id else model_id


class DetectionConfig(BaseModel):
    """Detection configuration defaults."""
    
    # Hash-based detection
    default_hash_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    num_hashes: int = Field(default=100)
    
    # Embedding-based detection
    default_embedding_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    embedding_config: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    
    # Candidate search
    max_candidates_per_search: int = Field(default=100)
    default_search_fields: list[str] = Field(
        default=["title", "description", "keywords", "url"]
    )
    
    @property
    def embedding_model(self) -> str:
        """Get current embedding model ID."""
        return self.embedding_config.get_model_id()
    
    @property
    def embedding_model_name(self) -> str:
        """Get current embedding model display name."""
        return self.embedding_config.get_model_name()


# Global config instances
wlo_config = WLOConfig()
detection_config = DetectionConfig()
