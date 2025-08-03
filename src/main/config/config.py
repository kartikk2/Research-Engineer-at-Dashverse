"""Configuration settings for the embedding workflow."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Import torch for device detection
try:
    import torch
except ImportError:
    torch = None

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the application."""
    
    # Pinecone settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "image-embeddings")
    PINECONE_HOST = os.getenv("PINECONE_HOST", "https://image-embeddings-mxyhas5.svc.aped-4627-b74a.pinecone.io")
    PINECONE_DIMENSION = int(os.getenv("PINECONE_DIMENSION", "512"))
    
    # CLIP model settings
    CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    
    # BLIP-2 model settings
    BLIP2_MODEL_NAME = os.getenv("BLIP2_MODEL_NAME", "Salesforce/blip2-flan-t5-xl")
    CAPTION_MAX_LENGTH = int(os.getenv("CAPTION_MAX_LENGTH", "50"))
    
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Dataset settings
    DATASET_PATH = os.getenv("DATASET_PATH", "data/dataset/dataset_updated")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    
    # Processing settings
    if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration.
        
        Returns:
            True if all required config is present
        """
        if not cls.PINECONE_API_KEY:
            print("Error: PINECONE_API_KEY not found in environment variables")
            return False
        
        if not Path(cls.DATASET_PATH).exists():
            print(f"Error: Dataset path {cls.DATASET_PATH} does not exist")
            return False
        
        return True 