"""Multi-threaded description insertion workflow for adding descriptions to images in Pinecone."""

import sys
from pathlib import Path

# Add the src directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import threading
import queue
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from main.config.config import Config
from main.pojo.models import ArtCategory, ImageMetadata
from main.db.vector_dao import VectorDBIngestor
from main.generation.caption_generator import BLIP2CaptionGenerator
from main.utils.utils import generate_image_id


class ThreadSafeCaptionGenerator:
    """Thread-safe wrapper for BLIP2CaptionGenerator."""
    
    def __init__(self):
        """Initialize the thread-safe caption generator."""
        self.caption_generator = BLIP2CaptionGenerator()
        self.lock = threading.Lock()
    
    def generate_caption(self, image_path: str, prompt: str) -> str:
        """Generate caption with thread safety.
        
        Args:
            image_path: Path to the image
            prompt: Text prompt for caption generation
            
        Returns:
            Generated caption string
        """
        with self.lock:
            return self.caption_generator.generate_caption(image_path, prompt)


def process_single_image(args: Tuple[str, str, ArtCategory, ThreadSafeCaptionGenerator, VectorDBIngestor]) -> Dict:
    """Process a single image with multiple prompts.
    
    Args:
        args: Tuple of (image_path, image_name, category, caption_generator, vector_ingestor)
        
    Returns:
        Dictionary with processing results
    """
    image_path, image_name, category, caption_generator, vector_ingestor = args
    
    try:
        print(f"🔄 Processing: {image_name}")
        
        # Generate image ID
        image_id = generate_image_id(image_path)
        
        # Define prompts for different metadata fields
        prompts = [
            ("Describe the composition and colors in this image.", "composition_colors"),
            (f"Question: What does this {category.value} artwork depict in terms of the objects, people, setting, the era, the artistic style, and the background elements?\nAnswer: ", "description"),
            ("Question: What are the distinct objects visible in this image?\nAnswer: ", "objects"),
            ("Question: Who are the people or characters present in this image?\nAnswer: ", "people_characters"),
            ("Question: What is the setting or location depicted in the image?\nAnswer: ", "setting_location")
        ]
        
        # Generate captions for each prompt
        metadata_updates = {}
        for prompt_text, metadata_key in prompts:
            try:
                print(f"      🔍 Generating {metadata_key} for {image_name}...")
                caption = caption_generator.generate_caption(image_path, prompt_text)
                
                # Clean and store the caption
                caption = caption.strip()
                metadata_updates[metadata_key] = caption
                
                print(f"      ✅ {metadata_key}: {caption[:100]}{'...' if len(caption) > 100 else ''}")
                
            except Exception as e:
                print(f"      ❌ Error generating {metadata_key} for {image_name}: {e}")
                metadata_updates[metadata_key] = f"Error generating {metadata_key}"
        
        # Update metadata in Pinecone
        update_result = vector_ingestor.update_captions_batch([{
            'id': image_id,
            **metadata_updates
        }])
        
        if update_result['success'] > 0:
            print(f"✅ Generated and updated {len(metadata_updates)} descriptions for {image_name}")
            return {'success': 1, 'failed': 0, 'image_name': image_name}
        else:
            print(f"❌ Failed to update descriptions for {image_name}")
            return {'success': 0, 'failed': 1, 'image_name': image_name}
            
    except Exception as e:
        print(f"❌ Error processing {image_name}: {e}")
        return {'success': 0, 'failed': 1, 'image_name': image_name}


def process_category_descriptions_mt(category: ArtCategory, num_threads: int = 4) -> Dict:
    """Process descriptions for a category using multiple threads.
    
    Args:
        category: Art category to process
        num_threads: Number of worker threads
        
    Returns:
        Dictionary with processing results
    """
    print(f"🎨 Starting multi-threaded description processing for category: {category.value}")
    print(f"🧵 Using {num_threads} worker threads")
    
    try:
        # Initialize shared resources
        print("🔄 Initializing shared resources...")
        caption_generator = ThreadSafeCaptionGenerator()
        vector_ingestor = VectorDBIngestor()
        
        # Get image files
        if "training_set" in str(Config.DATASET_PATH):
            category_path = Path(Config.DATASET_PATH) / category.value
        else:
            category_path = Path(Config.DATASET_PATH) / "training_set" / category.value
        
        print(f"🔍 Looking for images in: {category_path}")
        
        if not category_path.exists():
            print(f"❌ Category path does not exist: {category_path}")
            return {'success': 0, 'failed': 0, 'category': category.value}
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in category_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"❌ No image files found in category: {category.value}")
            return {'success': 0, 'failed': 0, 'category': category.value}
        
        print(f"📁 Found {len(image_files)} images to process")
        
        # # Limit to 100 images for testing
        # test_limit = 100
        # if len(image_files) > test_limit:
        #     print(f"🧪 TEST MODE: Limiting to first {test_limit} images")
        #     image_files = image_files[:test_limit]
        
        # Prepare work items
        work_items = []
        for image_file in image_files:
            image_path = str(image_file)
            image_name = image_file.name
            work_items.append((image_path, image_name, category, caption_generator, vector_ingestor))
        
        # Process images using thread pool
        total_success = 0
        total_failed = 0
        
        print(f"🚀 Starting multi-threaded processing with {num_threads} threads...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all work items
            future_to_image = {
                executor.submit(process_single_image, work_item): work_item[1] 
                for work_item in work_items
            }
            
            # Process completed tasks
            for future in as_completed(future_to_image):
                image_name = future_to_image[future]
                try:
                    result = future.result()
                    total_success += result['success']
                    total_failed += result['failed']
                    
                    # Progress update
                    processed = total_success + total_failed
                    progress = (processed / len(image_files)) * 100
                    print(f"📊 Progress: {processed}/{len(image_files)} ({progress:.1f}%) - {image_name}")
                    
                except Exception as e:
                    print(f"❌ Exception for {image_name}: {e}")
                    total_failed += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"🎉 Category {category.value} complete: {total_success} success, {total_failed} failed")
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        print(f"📈 Average time per image: {processing_time / len(image_files):.2f} seconds")
        
        return {
            'success': total_success,
            'failed': total_failed,
            'category': category.value,
            'processing_time': processing_time
        }
        
    except Exception as e:
        print(f"❌ Error processing category {category.value}: {e}")
        return {'success': 0, 'failed': 0, 'category': category.value}


def main():
    """Main function for multi-threaded description insertion workflow."""
    print("🧪 TEST MODE: Multi-Threaded Description Insertion Workflow")
    print("=" * 60)
    print("Processing 100 images using multiple threads with shared BLIP-2 model")
    print("• Multi-threaded processing for better performance")
    print("• Shared BLIP-2 model with thread-safe locks")
    print("• Parallel image processing and I/O operations")
    print("• Reduced memory usage (single model instance)")
    print("• Generating 5 descriptions per image")
    print("• Updating Pinecone metadata")
    print("• TEST MODE: Limited to 100 images")
    print()
    
    # Configuration
    category = ArtCategory.ICONOGRAPHY  # Change this as needed
    num_threads = 4  # Adjust based on your system capabilities
    
    print(f"🎯 Target Category: {category.value}")
    print(f"🧵 Number of Threads: {num_threads}")
    print()
    
    start_time = time.time()
    
    # Process the category
    result = process_category_descriptions_mt(category, num_threads)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    print(f"🎨 {result['category'].upper()}: {result['success']} success, {result['failed']} failed")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📈 Processing time: {result.get('processing_time', 0):.2f} seconds")
    
    if result['success'] > 0:
        print(f"✅ Successfully processed {result['success']} images with descriptions!")
        print(f"🚀 Performance: {result['success'] / result.get('processing_time', 1):.2f} images/second")
    else:
        print("❌ No images were successfully processed")


if __name__ == "__main__":
    main() 