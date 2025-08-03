"""Test script for the simplified BLIP-2 caption generator."""

import sys
import os
sys.path.append('src')

from main.generation.caption_generator import BLIP2CaptionGenerator


def test_single_caption():
    """Test single caption generation with different prompts."""
    print("🎨 Testing Single Caption Generation")
    print("=" * 50)
    
    # Initialize caption generator (model loaded once)
    generator = BLIP2CaptionGenerator()
    
    # Test image path
    test_image_path = "data/dataset/dataset_updated/training_set/sculpture/i - 788.jpeg"
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    # Test different prompts
    prompts = [
        # "List all distinct objects visible in this image.",
        # "Who are the people or characters in this image?",
        # "What is the setting or place shown in the image?",
        # "What is the emotional mood or tone of this image?",
        "Describe the composition and colors in this image.", #composition/colors
        # "Be brief and specific. What feelings does this image evoke?", #feelings/emptions
        "Question: What does this sculpture artwork depict in terms of the objects, people, setting, the era, the artistic style, and the background elements?\nAnswer: ", #description
        "Question: What are the distinct objects visible in this image?\nAnswer: ", #objects
        "Question: Who are the people or characters present in this image?\nAnswer: ", #people/chractaers
        "Question: What is the setting or location depicted in the image?\nAnswer: ", #setting/location
        # "Question: What is the emotional mood or tone conveyed by this image?\nAnswer: ",
        # "Question: How would you describe the composition and use of colors in this image?\nAnswer: ",
        # "Question: What specific feelings or emotions does this image evoke?\nAnswer: "
    ]

    
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        print("-" * 40)
        
        # Generate caption
        caption = generator.generate_caption(test_image_path, prompt)
        
        print(f"✅ Caption: {caption}")


def test_multiple_images():
    """Test caption generation for multiple images."""
    print("\n🎨 Testing Multiple Images")
    print("=" * 50)
    
    # Initialize caption generator (model loaded once)
    generator = BLIP2CaptionGenerator()
    
    # Find test images from different categories
    test_images = []
    dataset_path = "data/dataset/dataset_updated/training_set"
    categories = ['painting', 'sculpture', 'drawing']
    
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(category_path, file))
                    if len(test_images) >= 3:  # Test with 3 images
                        break
            if len(test_images) >= 3:
                break
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📁 Found {len(test_images)} test images")
    
    # Generate captions for each image
    for i, image_path in enumerate(test_images, 1):
        print(f"\n📝 Image {i}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        # Generate caption with a descriptive prompt
        prompt = "Describe this artwork in detail"
        caption = generator.generate_caption(image_path, prompt)
        
        print(f"✅ Caption: {caption}")


def main():
    """Main test function."""
    print("🎨 Simplified BLIP-2 Caption Generator Test")
    print("=" * 60)
    
    # Test single caption generation
    test_single_caption()
    
    # Test multiple images
    # test_multiple_images()
    
    print("\n🎉 Test completed!")


if __name__ == "__main__":
    main() 