"""Configuration for WLO Duplicate Detection API."""

from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field


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


class DetectionConfig(BaseModel):
    """Detection configuration defaults."""
    
    # Hash-based detection
    default_hash_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    num_hashes: int = Field(default=100)
    
    # Embedding-based detection
    default_embedding_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    embedding_model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Candidate search
    max_candidates_per_search: int = Field(default=100)
    default_search_fields: list[str] = Field(
        default=["title", "description", "keywords", "url"]
    )


# Global config instances
wlo_config = WLOConfig()
detection_config = DetectionConfig()
