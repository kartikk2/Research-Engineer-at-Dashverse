"""Vector database ingestion for image embeddings."""

import uuid
import base64
import hashlib
from typing import List, Dict, Optional
from pinecone import Pinecone
from ..config.config import Config
from ..pojo.models import ImageMetadata, ArtCategory
from ..utils.utils import generate_image_id


class VectorDBIngestor:
    """Handle vector database ingestion operations."""
    
    def __init__(self, api_key: Optional[str] = None, index_name: str = None):
        """Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key (from config if not provided)
            index_name: Name of the Pinecone index (from config if not provided)
        """
        self.api_key = api_key or Config.PINECONE_API_KEY
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.pc = Pinecone(api_key=self.api_key)
        
        # Connect to existing index
        self.index = self.pc.Index(self.index_name, host=Config.PINECONE_HOST)
        print(f"🔗 Connected to Pinecone index: {self.index_name}")
    
    def prepare_vector_data(self, image_metadata: ImageMetadata) -> Optional[Dict]:
        """Prepare image metadata for Pinecone ingestion.
        
        Args:
            image_metadata: Validated ImageMetadata object
            
        Returns:
            Dictionary with vector data or None if invalid
        """
        if not image_metadata.embedding:
            print(f"❌ No embedding found for {image_metadata.image_name}")
            return None
        
        # Category validation is handled by Pydantic in ImageMetadata
        # No need for manual validation here
        
        # Create ID using utility function
        image_id = generate_image_id(image_metadata.image_path)
        
        vector_data = {
            'id': image_id,
            'values': image_metadata.embedding,
            'metadata': {
                'image_name': image_metadata.image_name,
                'image_path': image_metadata.image_path,
                'category': image_metadata.category
            }
        }
        
        return vector_data
    
    def ingest_single_image(self, image_metadata: ImageMetadata) -> Dict:
        """Ingest a single image into Pinecone.
        
        Args:
            image_metadata: ImageMetadata object
            
        Returns:
            Dictionary with ingestion results
        """
        vector_data = self.prepare_vector_data(image_metadata)
        
        if not vector_data:
            return {'success': 0, 'failed': 1}
        
        try:
            self.index.upsert(vectors=[vector_data])
            return {'success': 1, 'failed': 0}
        except Exception as e:
            print(f"❌ Failed to ingest {image_metadata.image_name}: {e}")
            return {'success': 0, 'failed': 1}
    
    def ingest_batch(self, image_metadata_list: List[ImageMetadata]) -> Dict:
        """Ingest a batch of images into Pinecone.
        
        Args:
            image_metadata_list: List of ImageMetadata objects
            
        Returns:
            Dictionary with ingestion results
        """
        if not image_metadata_list:
            return {'success': 0, 'failed': 0}
        
        # Prepare all vector data
        vectors_to_upsert = []
        failed_count = 0
        
        for metadata in image_metadata_list:
            vector_data = self.prepare_vector_data(metadata)
            if vector_data:
                vectors_to_upsert.append(vector_data)
            else:
                failed_count += 1
        
        if not vectors_to_upsert:
            return {'success': 0, 'failed': len(image_metadata_list)}
        
        try:
            # Upsert all vectors at once
            self.index.upsert(vectors=vectors_to_upsert)
            
            # Print summary by category
            category_counts = {}
            for metadata in image_metadata_list:
                if metadata.embedding:  # Only count successful ones
                    category = metadata.category
                    category_counts[category] = category_counts.get(category, 0) + 1
            
            print("📊 Ingestion summary by category:")
            for category, count in category_counts.items():
                print(f"   • {category}: {count} images")
            
            return {
                'success': len(vectors_to_upsert),
                'failed': failed_count
            }
            
        except Exception as e:
            print(f"❌ Batch ingestion failed: {e}")
            return {'success': 0, 'failed': len(image_metadata_list)}
    
    def query_similar_images(self, embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Query for similar images.
        
        Args:
            embedding: Query embedding vector
            top_k: Number of similar results to return
            
        Returns:
            List of similar image results
        """
        try:
            results = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []
    
    def fetch_by_id(self, image_id: str) -> Optional[Dict]:
        """Fetch a specific image entry from Pinecone by ID.
        
        Args:
            image_id: The ID of the image to fetch
            
        Returns:
            Dictionary containing the vector data if found, None otherwise
        """
        try:
            # Fetch the vector from Pinecone
            fetch_response = self.index.fetch(ids=[image_id])
            
            # Handle the new Pinecone response format
            if hasattr(fetch_response, 'vectors') and image_id in fetch_response.vectors:
                vector_data = fetch_response.vectors[image_id]
                print(f"✅ Found image in Pinecone: {vector_data.metadata.get('image_name', 'Unknown')}")
                return {
                    'id': vector_data.id,
                    'values': vector_data.values,
                    'metadata': vector_data.metadata
                }
            elif isinstance(fetch_response, dict) and 'vectors' in fetch_response:
                # Fallback for older format
                if image_id in fetch_response['vectors']:
                    vector_data = fetch_response['vectors'][image_id]
                    print(f"✅ Found image in Pinecone: {vector_data['metadata']['image_name']}")
                    return vector_data
            else:
                print(f"❌ Image with ID {image_id} not found in Pinecone")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching from Pinecone: {e}")
            return None
    
    def update_captions_batch(self, metadata_updates: List[Dict[str, any]]) -> Dict:
        """Update metadata for multiple images in Pinecone.
        
        Args:
            metadata_updates: List of dictionaries with 'id' and various metadata fields
            
        Returns:
            Dictionary with update results
        """
        if not metadata_updates:
            return {'success': 0, 'failed': 0}
        
        try:
            success_count = 0
            failed_count = 0
            
            # Process each metadata update
            for update in metadata_updates:
                try:
                    image_id = update['id']
                    
                    # Extract all metadata fields except 'id'
                    metadata_fields = {k: v for k, v in update.items() if k != 'id'}
                    
                    if metadata_fields:
                        # Update the metadata for this image
                        self.index.update(
                            id=image_id,
                            set_metadata=metadata_fields
                        )
                        success_count += 1
                        print(f"✅ Updated metadata for {image_id}: {list(metadata_fields.keys())}")
                    else:
                        print(f"⚠️  No metadata fields found for {image_id}")
                        failed_count += 1
                        
                except Exception as e:
                    print(f"❌ Failed to update metadata for {update.get('id', 'unknown')}: {e}")
                    failed_count += 1
            
            print(f"📊 Metadata update summary: {success_count} success, {failed_count} failed")
            return {
                'success': success_count,
                'failed': failed_count
            }
            
        except Exception as e:
            print(f"❌ Batch metadata update failed: {e}")
            return {'success': 0, 'failed': len(metadata_updates)}
    
    def get_index_stats(self) -> None:
        """Get and display index statistics."""
        try:
            stats = self.index.describe_index_stats()
            print("📊 Index statistics:")
            print(f"   • Total vectors: {stats.total_vector_count}")
            print(f"   • Index fullness: {stats.dimension}")
            print(f"   • Namespaces: {list(stats.namespaces.keys()) if stats.namespaces else []}")
        except Exception as e:
            print(f"❌ Failed to get index stats: {e}")


def ingest_image_metadata(image_metadata_list: List[ImageMetadata], 
                         api_key: Optional[str] = None,
                         index_name: Optional[str] = None) -> Dict[str, int]:
    """Convenience function to ingest image metadata.
    
    Args:
        image_metadata_list: List of ImageMetadata objects with embeddings
        api_key: Optional Pinecone API key
        index_name: Optional Pinecone index name
        
    Returns:
        Dictionary with success/failure counts
    """
    ingestor = VectorDBIngestor(api_key=api_key, index_name=index_name)
    return ingestor.ingest_batch(image_metadata_list)


def main():
    """Main function for testing vector database ingestion."""
    print("🗄️  Vector Database Ingestion Test")
    print("=" * 50)
    
    # This is a test function - in practice, you would call this from other modules
    print("This module provides VectorDBIngestor class for Pinecone operations.")
    print("Import and use VectorDBIngestor or ingest_image_metadata function.")


if __name__ == "__main__":
    main() 