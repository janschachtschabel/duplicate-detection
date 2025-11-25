"""Pydantic models for WLO Duplicate Detection API."""

from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, urlunparse
from pydantic import BaseModel, Field
from enum import Enum

from app.config import Environment


def normalize_title(title: Optional[str]) -> Optional[str]:
    """
    Normalize title for better duplicate matching.
    
    Normalizations:
    - Remove common suffixes (- Wikipedia, | Klexikon, etc.)
    - Remove publisher names in brackets
    - Strip extra whitespace
    
    Examples:
        "Islam - Wikipedia" -> "Islam"
        "Mathematik | Klexikon" -> "Mathematik"
        "Geschichte (planet-schule.de)" -> "Geschichte"
    """
    if not title or not title.strip():
        return None
    
    title = title.strip()
    
    # Common suffixes to remove (case-insensitive patterns)
    suffixes = [
        r'\s*[-–—|:]\s*Wikipedia.*$',
        r'\s*[-–—|:]\s*Klexikon.*$',
        r'\s*[-–—|:]\s*Wikibooks.*$',
        r'\s*[-–—|:]\s*Wikiversity.*$',
        r'\s*[-–—|:]\s*planet-schule.*$',
        r'\s*[-–—|:]\s*Planet Schule.*$',
        r'\s*[-–—|:]\s*Lehrer-Online.*$',
        r'\s*[-–—|:]\s*Lernhelfer.*$',
        r'\s*[-–—|:]\s*sofatutor.*$',
        r'\s*[-–—|:]\s*learningapps.*$',
        r'\s*[-–—|:]\s*serlo.*$',
        r'\s*\([^)]*\.(de|com|org|net|edu)\)$',  # (example.de)
        r'\s*\|\s*[^|]+$',  # | anything at the end
    ]
    
    import re
    normalized = title
    for pattern in suffixes:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
    
    normalized = normalized.strip()
    
    # Return None if normalized is empty or same as original
    if not normalized:
        return None
    
    return normalized if normalized != title else None


def normalize_url(url: Optional[str]) -> Optional[str]:
    """
    Normalize URL for better duplicate matching.
    
    Normalizations:
    - Lowercase
    - Remove protocol (http/https)
    - Remove www. prefix
    - Remove trailing slashes
    - Remove common tracking parameters
    
    Examples:
        https://www.Example.com/Page/ -> example.com/page
        http://example.com/page?utm_source=x -> example.com/page
    """
    if not url or not url.strip():
        return None
    
    url = url.strip().lower()
    
    try:
        parsed = urlparse(url)
        
        # Get host without www.
        host = parsed.netloc
        if host.startswith('www.'):
            host = host[4:]
        
        # Get path without trailing slash
        path = parsed.path.rstrip('/')
        
        # Remove common tracking query parameters
        # Keep the path but drop query params for matching
        # (exact URL match is still possible via original URL)
        
        # Reconstruct normalized URL (no protocol, no query)
        normalized = host + path
        
        return normalized if normalized else None
        
    except Exception:
        # If parsing fails, return lowercase stripped version
        return url.strip().lower()


class SearchField(str, Enum):
    """Available metadata fields for candidate search."""
    TITLE = "title"
    DESCRIPTION = "description"
    KEYWORDS = "keywords"
    URL = "url"


class ContentMetadata(BaseModel):
    """Content metadata for duplicate detection."""
    title: Optional[str] = Field(default=None, description="Title of the content")
    description: Optional[str] = Field(default=None, description="Description text")
    keywords: Optional[List[str]] = Field(default=None, description="List of keywords")
    url: Optional[str] = Field(default=None, description="Content URL (ccm:wwwurl)")
    
    @property
    def normalized_url(self) -> Optional[str]:
        """Get normalized URL for duplicate matching."""
        return normalize_url(self.url)
    
    def get_searchable_text(self) -> str:
        """Get combined searchable text from all fields."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.description:
            parts.append(self.description)
        if self.keywords:
            parts.append(" ".join(self.keywords))
        return " ".join(parts)
    
    def has_content(self) -> bool:
        """Check if there is any content to search with."""
        return bool(self.title or self.description or self.keywords or self.url)


class DuplicateCandidate(BaseModel):
    """A potential duplicate candidate."""
    node_id: str = Field(..., description="Node ID of the candidate")
    title: Optional[str] = Field(default=None, description="Title of the candidate")
    description: Optional[str] = Field(default=None, description="Description of the candidate")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords of the candidate")
    url: Optional[str] = Field(default=None, description="URL of the candidate")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    match_source: str = Field(..., description="Which search field found this candidate")


class DetectionRequest(BaseModel):
    """Base request for duplicate detection."""
    environment: Environment = Field(
        default=Environment.PRODUCTION,
        description="WLO environment (production or staging)"
    )
    search_fields: List[SearchField] = Field(
        default=[SearchField.TITLE, SearchField.DESCRIPTION, SearchField.KEYWORDS, SearchField.URL],
        description="Metadata fields to use for candidate search"
    )
    max_candidates: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum candidates per search field (pagination used if > 100)"
    )


class NodeIdRequest(DetectionRequest):
    """Request for detection by Node ID."""
    node_id: str = Field(..., description="Node ID of the content to check")


class MetadataRequest(DetectionRequest):
    """Request for detection by direct metadata input."""
    metadata: ContentMetadata = Field(..., description="Content metadata to check for duplicates")


class HashDetectionRequest(NodeIdRequest):
    """Request for hash-based duplicate detection by Node ID."""
    similarity_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for hash-based matching (0-1)"
    )


class HashMetadataRequest(MetadataRequest):
    """Request for hash-based detection with direct metadata."""
    similarity_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for hash-based matching (0-1)"
    )


class EmbeddingDetectionRequest(NodeIdRequest):
    """Request for embedding-based duplicate detection by Node ID."""
    similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for embedding matching (0-1)"
    )


class EmbeddingMetadataRequest(MetadataRequest):
    """Request for embedding-based detection with direct metadata."""
    similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for embedding matching (0-1)"
    )


class CandidateStats(BaseModel):
    """Statistics about candidate search per field."""
    field: str = Field(..., description="Search field name")
    search_value: Optional[str] = Field(default=None, description="Value used for search (truncated)")
    candidates_found: int = Field(default=0, description="Total number of candidates found")
    highest_similarity: Optional[float] = Field(default=None, description="Highest similarity score among candidates (0-1)")
    # Detailed breakdown for normalized searches
    original_count: Optional[int] = Field(default=None, description="Candidates from original search")
    normalized_search: Optional[str] = Field(default=None, description="Normalized search value (if different)")
    normalized_count: Optional[int] = Field(default=None, description="Additional candidates from normalized search")


class DetectionResponse(BaseModel):
    """Response from duplicate detection."""
    success: bool = Field(default=True)
    source_node_id: Optional[str] = Field(default=None, description="Node ID of source content (if provided)")
    source_metadata: Optional[ContentMetadata] = Field(default=None, description="Metadata used for detection")
    method: str = Field(..., description="Detection method used (hash or embedding)")
    threshold: float = Field(..., description="Similarity threshold used")
    candidate_search_results: List[CandidateStats] = Field(
        default_factory=list, 
        description="Candidates found per search field"
    )
    total_candidates_checked: int = Field(default=0, description="Total unique candidates checked")
    duplicates: List[DuplicateCandidate] = Field(default_factory=list, description="List of potential duplicates")
    error: Optional[str] = Field(default=None, description="Error message if any")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    hash_detection_available: bool = Field(default=True, description="Hash-based detection is always available")
    embedding_detection_available: bool = Field(default=False, description="Whether embedding detection is available")
    embedding_model_loaded: bool = Field(default=False, description="Whether embedding model is loaded")
    embedding_model_name: str = Field(default="", description="Name of the configured embedding model")
    version: str = Field(default="1.0.0")


class EmbeddingRequest(BaseModel):
    """Request for text embedding."""
    text: str = Field(..., description="Text to embed", min_length=1)
    

class EmbeddingBatchRequest(BaseModel):
    """Request for batch text embedding."""
    texts: List[str] = Field(..., description="List of texts to embed", min_length=1)


class EmbeddingResponse(BaseModel):
    """Response with embedding vector."""
    success: bool = Field(default=True)
    text: str = Field(..., description="Input text")
    embedding: List[float] = Field(..., description="Embedding vector")
    dimensions: int = Field(..., description="Number of dimensions")
    model: str = Field(..., description="Model used for embedding")
    error: Optional[str] = Field(default=None)


class EmbeddingBatchResponse(BaseModel):
    """Response with multiple embedding vectors."""
    success: bool = Field(default=True)
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    dimensions: int = Field(..., description="Number of dimensions per embedding")
    count: int = Field(..., description="Number of embeddings returned")
    model: str = Field(..., description="Model used for embedding")
    error: Optional[str] = Field(default=None)
