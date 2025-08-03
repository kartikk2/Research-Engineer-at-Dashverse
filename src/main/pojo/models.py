"""Data models for the multimodal AI application."""

import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ArtCategory(str, Enum):
    """Enum for art categories."""
    DRAWING = "drawing"
    ENGRAVING = "engraving"
    ICONOGRAPHY = "iconography"
    PAINTING = "painting"
    SCULPTURE = "sculpture"


class ImageMetadata(BaseModel):
    """Metadata for an image with Pydantic validation."""
    image_name: str = Field(..., description="Name of the image file")
    image_path: str = Field(..., description="Full path to the image file")
    category: ArtCategory = Field(..., description="Art category of the image")
    embedding: Optional[List[float]] = Field(None, description="CLIP embedding vector")
    description: Optional[str] = Field(None, description="Description of the image")
    
    @field_validator('image_path')
    @classmethod
    def validate_image_path(cls, v):
        """Validate that the image path exists."""
        if not os.path.exists(v):
            raise ValueError(f"Image path does not exist: {v}")
        return v
    
    @field_validator('image_name')
    @classmethod
    def validate_image_name(cls, v):
        """Validate image name is not empty."""
        if not v.strip():
            raise ValueError("Image name cannot be empty")
        return v.strip()
    
    model_config = {
        "use_enum_values": True,
        "validate_assignment": True
    } 