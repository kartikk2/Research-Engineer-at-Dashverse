"""Description insertion workflow for adding descriptions to images in Pinecone."""

import sys
from pathlib import Path

# Add the src directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import multiprocessing as mp
from typing import List, Dict, Optional
import time

from main.config.config import Config
from main.pojo.models import ArtCategory, ImageMetadata
from main.db.vector_dao import VectorDBIngestor
from main.generation.caption_generator import BLIP2CaptionGenerator
from main.utils.utils import generate_image_id


def process_category_descriptions_worker(category: ArtCategory, caption_generator: BLIP2CaptionGenerator, test_mode: bool = False) -> Dict:
    """Worker function to process descriptions for a single category.
    
    Args:
        category: Art category to process
        caption_generator: Pre-loaded BLIP2CaptionGenerator instance
        
    Returns:
        Dictionary with processing results
    """
    print(f"🎨 Starting description processing for category: {category.value}")
    
    try:
        # Initialize vector ingestor only
        vector_ingestor = VectorDBIngestor()
        
        # Get all images in the category
        # Check if DATASET_PATH already includes training_set
        if "training_set" in str(Config.DATASET_PATH):
            category_path = Path(Config.DATASET_PATH) / category.value
        else:
            category_path = Path(Config.DATASET_PATH) / "training_set" / category.value
        
        # Debug: Print the actual path being used
        print(f"🔍 Looking for images in: {category_path}")
        print(f"🔍 Config.DATASET_PATH: {Config.DATASET_PATH}")
        print(f"🔍 Device being used: {Config.DEVICE}")
        
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
        
        # Process images (all or just one for testing)
        if test_mode:
            print(f"🧪 TEST MODE: Processing only 1 image in {category.value}")
            image_files = image_files[:1]  # Take only the first image
        else:
            print(f"📁 Processing all {len(image_files)} images in {category.value}")
        
        # Process images directly without batching
        total_success = 0
        total_failed = 0
        
        for i, image_file in enumerate(image_files, 1):
            try:
                image_path = str(image_file)
                image_name = image_file.name
                
                print(f"🔄 Processing image {i}/{len(image_files)}: {image_name}")
                
                # Generate image ID
                image_id = generate_image_id(image_path)
                
                # Generate multiple descriptions using different prompts
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
                        # Customize prompt for the specific category
                        if "sculpture artwork" in prompt_text:
                            prompt_text = prompt_text.replace("sculpture artwork", f"{category.value} artwork")
                        
                        print(f"      🔍 Generating {metadata_key} for {image_name}...")
                        caption = caption_generator.generate_caption(image_path, prompt_text)
                        
                        # Clean and store the caption
                        caption = caption.strip()
                        metadata_updates[metadata_key] = caption
                        
                        print(f"      ✅ {metadata_key}: {caption[:100]}{'...' if len(caption) > 100 else ''}")
                        
                    except Exception as e:
                        print(f"      ❌ Error generating {metadata_key} for {image_name}: {e}")
                        metadata_updates[metadata_key] = f"Error generating {metadata_key}"
                
                # Update all descriptions in Pinecone
                update_result = vector_ingestor.update_captions_batch([{
                    'id': image_id,
                    **metadata_updates  # This spreads all the metadata keys and values
                }])
                
                if update_result['success'] > 0:
                    total_success += 1
                    print(f"✅ Generated and updated {len(metadata_updates)} descriptions for {image_name} (ID: {image_id})")
                    print(f"   📝 Metadata keys: {list(metadata_updates.keys())}")
                else:
                    total_failed += 1
                    print(f"❌ Failed to update descriptions for {image_name}")
                
            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")
                total_failed += 1
        
        print(f"🎉 Category {category.value} complete: {total_success} total success, {total_failed} total failed")
        return {
            'success': total_success,
            'failed': total_failed,
            'category': category.value
        }
        
    except Exception as e:
        print(f"❌ Error processing category {category.value}: {e}")
        return {'success': 0, 'failed': 0, 'category': category.value}


def process_all_categories_parallel() -> List[Dict]:
    """Process drawing category with all images.
    
    Returns:
        List of results from the drawing category worker
    """
    print("🚀 Starting description processing for drawing category")
    print("🔧 Processing ALL images in the drawing subdirectory")
    print("🔧 BLIP-2 model loaded once and reused")
    print("=" * 60)
    
    # Load BLIP-2 model once
    print("🔄 Loading BLIP-2 model once...")
    caption_generator = BLIP2CaptionGenerator()
    
    # Process drawing category with all images
    result = process_category_descriptions_worker(ArtCategory.ENGRAVING, caption_generator, test_mode=False)
    
    return [result]


def main():
    """Main function for description insertion workflow."""
    print("🎨 Description Insertion Workflow")
    print("=" * 50)
    print("Processing ALL images from drawing category in training dataset...")
    print("• Sequential processing (no multiprocessing)")
    print("• BLIP-2 model loaded once and reused")
    print("• Processing all images in drawing subdirectory")
    print("• Simple processing: one image at a time")
    print("• Generating descriptions with BLIP-2")
    print("• Updating Pinecone metadata")
    print()
    
    start_time = time.time()
    
    # Process drawing category only
    results = process_all_categories_parallel()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    total_success = 0
    total_failed = 0
    
    for result in results:
        category = result['category']
        success = result['success']
        failed = result['failed']
        
        total_success += success
        total_failed += failed
        
        print(f"🎨 {category.upper()}: {success} success, {failed} failed")
    
    print(f"\n🎉 TOTAL: {total_success} success, {total_failed} failed")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    
    # Calculate average time per image only if images were processed
    if total_success + total_failed > 0:
        print(f"📈 Average time per image: {total_time / (total_success + total_failed):.2f} seconds")
    else:
        print("📈 No images processed - cannot calculate average time")
    
    if total_success > 0:
        print(f"✅ Successfully processed {total_success} images with descriptions!")
    else:
        print("❌ No images were successfully processed")


if __name__ == "__main__":
    main() 