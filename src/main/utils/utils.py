"""Utility functions for the multimodal AI application."""

import hashlib
from typing import Optional


def generate_image_id(image_path: str) -> str:
    """Generate a unique ID for an image based on its path.
    
    Args:
        image_path: Full path to the image file
        
    Returns:
        MD5 hash of the image path as a hexadecimal string
    """
    return hashlib.md5(image_path.encode('utf-8')).hexdigest()


def validate_image_path(image_path: str) -> bool:
    """Validate that an image path exists and is accessible.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if the path exists and is accessible, False otherwise
    """
    import os
    return os.path.exists(image_path) and os.access(image_path, os.R_OK)

