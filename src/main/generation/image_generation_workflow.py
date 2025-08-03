"""LangGraph workflow for image generation using seed descriptions and similar images."""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END

from ..config.config import Config
from ..pojo.models import ArtCategory
from ..db.vector_dao import VectorDBIngestor
from ..generation.seed_generator import SeedGenerator
from ..generation.image_generator import StableDiffusionGenerator
from ..generation.gpt4o_caption_generator import GPT4OCaptionGenerator
from ..generation.gpt4o_prompt_generator import generate_sdxl_prompt_from_seed



class WorkflowState(TypedDict):
    """State for the image generation workflow."""
    seed_description: Optional[str]
    seed_embedding: Optional[List[float]]
    similar_images: Optional[List[Dict]]
    image_id: Optional[str]
    image_name: Optional[str]
    similar_image_objects: Optional[List[Dict]]
    generated_image: Optional[Any]  # PIL Image
    generated_caption: Optional[str]
    output_path: Optional[str]
    workflow_id: str
    timestamp: str
    status: str
    error: Optional[str]


def create_workflow() -> StateGraph:
    """Create the LangGraph workflow for image generation.
    
    Returns:
        Configured StateGraph for the workflow
    """
    # Create the workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("generate_seed", generate_seed_node)
    workflow.add_node("embed_seed", embed_seed_node)
    workflow.add_node("find_similar_images", find_similar_images_node)
    workflow.add_node("generate_image_and_caption", generate_image_and_caption_node)
    workflow.add_node("save_results", save_results_node)
    
    # Define the workflow edges
    workflow.set_entry_point("generate_seed")
    workflow.add_edge("generate_seed", "embed_seed")
    workflow.add_edge("embed_seed", "find_similar_images")
    workflow.add_edge("find_similar_images", "generate_image_and_caption")
    workflow.add_edge("generate_image_and_caption", "save_results")
    workflow.add_edge("save_results", END)
    
    return workflow


def generate_seed_node(state: WorkflowState) -> WorkflowState:
    """Generate a random seed description using existing SeedGenerator.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with seed description, image_id, and image_name
    """
    print("🎲 Generating random seed...")
    
    try:
        # Initialize seed generator
        seed_generator = SeedGenerator()
        
        # Generate seed and get image info
        seed_result = seed_generator.generate_seed()
        
        if seed_result:
            seed_description, image_id = seed_result
            state['seed_description'] = seed_description
            state['image_id'] = image_id
            state['status'] = "seed_generated"
            print(f"✅ Generated seed: {seed_description}")
            print(f"🆔 Source image ID: {image_id}")
        else:
            # Fallback seed if no data found
            state['seed_description'] = "a beautiful artwork with vibrant colors and detailed composition"
            state['image_id'] = "fallback"
            state['status'] = "seed_generated_fallback"
            print("⚠️  Using fallback seed description")
        
    except Exception as e:
        state['error'] = f"Failed to generate seed: {str(e)}"
        state['status'] = "error"
        print(f"❌ Error generating seed: {e}")
    
    return state


def embed_seed_node(state: WorkflowState) -> WorkflowState:
    """Embed the seed description using CLIP text encoder.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with seed embedding
    """
    print("🔤 Embedding seed description...")
    
    try:
        # Generate text embedding for the seed description
        seed_description = state.get('seed_description', '')
        if not seed_description:
            raise ValueError("No seed description available")
        
        # Use CLIP text encoder to embed the description
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        model_name = Config.CLIP_MODEL_NAME
        device = torch.device(Config.DEVICE)
        
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        
        # Process text input
        inputs = processor(text=seed_description, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text embedding
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_embedding = text_features.cpu().numpy().flatten().tolist()
        
        state['seed_embedding'] = text_embedding
        state['status'] = "seed_embedded"
        print(f"✅ Embedded seed description (dimensions: {len(text_embedding)})")
        
    except Exception as e:
        state['error'] = f"Failed to embed seed: {str(e)}"
        state['status'] = "error"
        print(f"❌ Error embedding seed: {e}")
    
    return state


def find_similar_images_node(state: WorkflowState) -> WorkflowState:
    """Find similar images from Pinecone using the seed embedding.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with similar images, similar_descriptions, image_id, and image_name
    """
    print("🔍 Finding similar images...")
    
    try:
        # Initialize vector database ingestor
        vector_ingestor = VectorDBIngestor()
        
        # Get seed embedding
        seed_embedding = state.get('seed_embedding')
        if not seed_embedding:
            raise ValueError("No seed embedding available")
        
        # Query for similar images (top 5)
        similar_images = vector_ingestor.query_similar_images(
            embedding=seed_embedding,
            top_k=5
        )
        
        if similar_images:
            state['similar_images'] = similar_images
            state['status'] = "similar_images_found"
            print(f"✅ Found {len(similar_images)} similar images")
            
            # Create list of similar image objects with all metadata
            similar_image_objects = []
            
            for i, img in enumerate(similar_images, 1):
                metadata = img.metadata
                
                # Create similar image object with all metadata fields
                similar_image = {
                    'image_name': metadata.get('image_name', 'unknown'),
                    'category': metadata.get('category', 'artwork'),
                    'description': metadata.get('description', ''),
                    'objects': metadata.get('objects', ''),
                    'people_characters': metadata.get('people_characters', ''),
                    'setting_location': metadata.get('setting_location', ''),
                    'composition_colors': metadata.get('composition_colors', '')
                }
                
                similar_image_objects.append(similar_image)
                
                # Set image_name from the first similar image (image_id already set from seed generation)
                if i == 1:
                    state['image_name'] = similar_image['image_name']
                
                print(f"   {i}. {similar_image['image_name']} ({similar_image['category']})")
                print(f"      📝 Description: {similar_image['description'][:100]}{'...' if len(similar_image['description']) > 100 else ''}")
                print(f"      🎯 Objects: {similar_image['objects'][:100]}{'...' if len(similar_image['objects']) > 100 else ''}")
                print(f"      👥 People: {similar_image['people_characters'][:100]}{'...' if len(similar_image['people_characters']) > 100 else ''}")
                print(f"      🏛️  Setting: {similar_image['setting_location'][:100]}{'...' if len(similar_image['setting_location']) > 100 else ''}")
                print(f"      🎨 Composition: {similar_image['composition_colors'][:100]}{'...' if len(similar_image['composition_colors']) > 100 else ''}")
            
            # Save similar image objects to state
            state['similar_image_objects'] = similar_image_objects
            
            print(f"📝 Extracted metadata from {len(similar_image_objects)} similar images")
            
            # Debug: Print raw metadata for first image
            if similar_image_objects:
                first_img = similar_image_objects[0]
                print(f"🔍 DEBUG - First image metadata:")
                print(f"   Description: '{first_img['description']}' (len: {len(first_img['description'])})")
                print(f"   Objects: '{first_img['objects']}' (len: {len(first_img['objects'])})")
                print(f"   People: '{first_img['people_characters']}' (len: {len(first_img['people_characters'])})")
                print(f"   Setting: '{first_img['setting_location']}' (len: {len(first_img['setting_location'])})")
                print(f"   Composition: '{first_img['composition_colors']}' (len: {len(first_img['composition_colors'])})")
            
        else:
            state['similar_images'] = []
            state['similar_image_objects'] = []
            state['image_id'] = 'unknown'
            state['image_name'] = 'unknown'
            state['status'] = "no_similar_images"
            print("⚠️  No similar images found")
        
    except Exception as e:
        state['error'] = f"Failed to find similar images: {str(e)}"
        state['status'] = "error"
        print(f"❌ Error finding similar images: {e}")
    
    return state


def generate_image_and_caption_node(state: WorkflowState) -> WorkflowState:
    """Generate image and caption in parallel using similar images as context.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with generated image and caption
    """
    print("🎨 Generating image and caption in parallel...")
    
    try:
        # Get values from state
        similar_images = state.get('similar_images', [])
        similar_image_objects = state.get('similar_image_objects', [])
        seed_description = state.get('seed_description', '')
        image_id = state.get('image_id', 'unknown')
        image_name = state.get('image_name', 'unknown')
        
        # Initialize variables
        generated_image = None
        generated_caption = None
        
        if not similar_images:
            print("⚠️  No similar images available, using base generation")
            # Generate without context using a fallback seed
            fallback_seed = 42  # Default seed when no image ID available
            print(f"🎲 Using fallback seed: {fallback_seed} (no image ID available)")
            
            image_generator = StableDiffusionGenerator()
            generated_image = image_generator.generate_image(
                prompt=seed_description,
                num_inference_steps=25,
                guidance_scale=7.5,
                seed=fallback_seed
            )
            
            # Generate caption without context
            caption_generator = GPT4OCaptionGenerator()
            generated_caption = caption_generator.generate_caption(
                seed_description, 
                "artwork", 
                []
            )
        else:
            # Extract categories from similar images for dominant category
            categories = []
            for img in similar_images:
                metadata = img.metadata
                category = metadata.get('category', 'artwork')
                categories.append(category)
            
            # Determine dominant category
            dominant_category = max(set(categories), key=categories.count) if categories else "artwork"
            
            # Generate optimized SDXL prompt using GPT-4o
            print("🤖 Generating optimized SDXL prompt with GPT-4o...")
            image_prompt = generate_sdxl_prompt_from_seed(
                user_seed=seed_description,
                dominant_category=dominant_category,
                similar_image_objects=similar_image_objects
            )
            
            # Extract metadata from similar image objects for caption generation only
            settings_list = []
            compositions_list = []
            descriptions_list = []
            
            for img in similar_image_objects:
                # Filter out bad composition_colors values
                composition_colors = img['composition_colors']
                if (composition_colors and composition_colors.strip() and 
                    'composition and colours' not in composition_colors.lower()):
                    compositions_list.append(composition_colors)
                
                # Filter out bad setting_location values
                setting_location = img['setting_location']
                if (setting_location and setting_location.strip() and 
                    'setting or location' not in setting_location.lower()):
                    settings_list.append(setting_location)
                
                # Include descriptions
                description = img['description']
                if description and description.strip():
                    descriptions_list.append(description)
            
            # Create enhanced context prompt for caption generation
            context_prompt = seed_description + f" in {dominant_category} style"
            
            # Add setting/location context for caption
            if settings_list:
                setting_context = ', '.join(settings_list)
                context_prompt += f" in settings: {setting_context}"
            
            # Add composition/colors context for caption
            if compositions_list:
                composition_context = ', '.join(compositions_list)
                context_prompt += f" with composition and colors: {composition_context}"
            
            # Add general descriptions as inspiration for caption
            if descriptions_list:
                context_prompt += f" inspired by: {', '.join(descriptions_list)}"
            
            print(f"🎨 Image generation prompt: {image_prompt}")
            print(f"📝 Caption generation prompt: {context_prompt}")
            print(f"🖼️  Based on image: {image_name} (ID: {image_id})")
            print(f"📝 Using metadata from {len(similar_image_objects)} similar images for caption:")
            print(f"   • Settings: {len(settings_list)}")
            print(f"   • Compositions: {len(compositions_list)}")
            print(f"   • Descriptions: {len(descriptions_list)}")
            
            # Generate image and caption in parallel
            async def generate_parallel():
                # Create tasks for parallel execution
                image_task = asyncio.create_task(generate_image_async(image_prompt, image_id))
                caption_task = asyncio.create_task(generate_caption_async(
                    image_prompt, 
                    dominant_category, 
                    similar_image_objects
                ))
                
                # Wait for both to complete
                generated_image, generated_caption = await asyncio.gather(image_task, caption_task)
                return generated_image, generated_caption
            
            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                generated_image, generated_caption = loop.run_until_complete(generate_parallel())
            finally:
                loop.close()
        
        # Save generated results to state
        state['generated_image'] = generated_image
        state['generated_caption'] = generated_caption
        state['status'] = "generation_complete"
        print(f"✅ Generated image and caption: {generated_caption[:100] if generated_caption else 'No caption'}...")
        
    except Exception as e:
        state['error'] = f"Failed to generate image and caption: {str(e)}"
        state['status'] = "error"
        print(f"❌ Error generating image and caption: {e}")
    
    return state


async def generate_image_async(prompt: str, image_id: str) -> Any:
    """Async wrapper for image generation with deterministic seed based on image ID.
    
    Args:
        prompt: The generation prompt
        image_id: The source image ID to derive the seed from
        
    Returns:
        Generated image
    """
    # Generate a deterministic seed from the image ID
    import hashlib
    seed_hash = hashlib.md5(image_id.encode()).hexdigest()
    # Convert first 8 characters of hash to integer (for 32-bit seed)
    deterministic_seed = int(seed_hash[:8], 16)
    
    print(f"🎲 Using deterministic seed: {deterministic_seed} (derived from image ID: {image_id})")
    
    image_generator = StableDiffusionGenerator()
    return image_generator.generate_image(
        prompt=prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        seed=deterministic_seed
    )


async def generate_caption_async(
    prompt: str, 
    art_category: str, 
    similar_image_objects: List[Dict]
) -> str:
    """Async wrapper for caption generation using GPT4OCaptionGenerator.
    
    Args:
        prompt: The original prompt used for generation
        art_category: The art category (painting, sculpture, etc.)
        similar_image_objects: List of similar image objects with metadata
        
    Returns:
        Generated caption string
    """
    try:
        # Initialize GPT-4o caption generator
        caption_generator = GPT4OCaptionGenerator()
        
        # Combine all metadata from similar image objects for richer context
        all_context = []
        for img_obj in similar_image_objects:
            if img_obj.get('description'):
                all_context.append(img_obj['description'])
            if img_obj.get('objects'):
                all_context.append(img_obj['objects'])
            if img_obj.get('people_characters'):
                all_context.append(img_obj['people_characters'])
            if img_obj.get('setting_location'):
                all_context.append(img_obj['setting_location'])
            if img_obj.get('composition_colors'):
                all_context.append(img_obj['composition_colors'])
        
        # Generate caption using the class with enhanced context
        caption = caption_generator.generate_caption(prompt, art_category, all_context)
        
        return caption
        
    except Exception as e:
        print(f"❌ Error generating caption: {e}")
        # Fallback caption
        # Extract descriptions from similar image objects for fallback
        similar_descriptions = []
        for img_obj in similar_image_objects:
            if img_obj.get('description'):
                similar_descriptions.append(img_obj['description'])
        
        if similar_descriptions:
            similar_context = ', '.join(similar_descriptions[:2])
            return f"A {prompt.lower()}. This {art_category} artwork draws inspiration from similar pieces that feature {similar_context}. The composition combines elements from these reference works to create a unique and cohesive piece."
        else:
            return f"A {prompt.lower()}. This {art_category} artwork showcases artistic creativity and visual appeal."


def save_results_node(state: WorkflowState) -> WorkflowState:
    """Save the generated image and log the caption.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with output path
    """
    print("💾 Saving results...")
    
    try:
        # Get values from state
        generated_image = state.get('generated_image')
        generated_caption = state.get('generated_caption', 'No caption generated')
        workflow_id = state.get('workflow_id', str(uuid.uuid4()))
        image_id = state.get('image_id', 'unknown')
        image_name = state.get('image_name', 'unknown')
        similar_image_objects = state.get('similar_image_objects', [])
        seed_description = state.get('seed_description', 'No seed description')
        

        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if generated_image:
            # Create output directory
            output_dir = Path("generated_images")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            filename = f"generated_{workflow_id}_{timestamp}.png"
            output_path = output_dir / filename
            
            # Save image
            generated_image.save(output_path)
            state['output_path'] = str(output_path)
            
            print(f"✅ Image saved to: {output_path}")
        else:
            state['output_path'] = None
            print("⚠️  No image to save")
        
        # Log the caption
        print(f"📝 Generated caption: {generated_caption}")
        
        # Save caption to file
        caption_dir = Path("generated_captions")
        caption_dir.mkdir(exist_ok=True)
        
        caption_filename = f"caption_{workflow_id}_{timestamp}.txt"
        caption_path = caption_dir / caption_filename
        
        # Extract art category from similar images
        art_category = "artwork"  # Default
        similar_images = state.get('similar_images', [])
        if similar_images:
            # Get category from first similar image
            first_image_metadata = similar_images[0].metadata
            art_category = first_image_metadata.get('category', 'artwork')
        
        with open(caption_path, 'w') as f:
            f.write(f"Generated Image: {state.get('output_path', 'N/A')}\n")
            f.write(f"Seed Description: {seed_description}\n")
            f.write(f"Art Category: {art_category}\n")
            f.write(f"Original Image ID: {image_id}\n")
            f.write(f"Source Image Name: {image_name}\n")
            f.write(f"Similar Images: {len(similar_image_objects)} found\n")
            f.write(f"Similar Images with Objects: {len([img for img in similar_image_objects if img.get('objects')])} found\n")
            f.write(f"Similar Images with People/Characters: {len([img for img in similar_image_objects if img.get('people_characters')])} found\n")
            f.write(f"Similar Images with Settings: {len([img for img in similar_image_objects if img.get('setting_location')])} found\n")
            f.write(f"Similar Images with Compositions: {len([img for img in similar_image_objects if img.get('composition_colors')])} found\n")
            f.write(f"Generated Caption: {generated_caption}\n")
            f.write(f"Timestamp: {state.get('timestamp', 'N/A')}\n")
            f.write(f"Workflow ID: {workflow_id}\n")
        
        print(f"📝 Caption saved to: {caption_path}")
        state['status'] = "complete"
        
    except Exception as e:
        state['error'] = f"Failed to save results: {str(e)}"
        state['status'] = "error"
        print(f"❌ Error saving results: {e}")
    
    return state


def create_initial_state() -> WorkflowState:
    """Create initial state for the workflow.
    
    Returns:
        Initial workflow state
    """
    return {
        'seed_description': None,
        'seed_embedding': None,
        'similar_images': None,
        'image_id': None,
        'image_name': None,
        'similar_image_objects': None,
        'generated_image': None,
        'generated_caption': None,
        'output_path': None,
        'workflow_id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'status': 'initialized',
        'error': None
    }


def run_workflow() -> Dict[str, Any]:
    """Run the complete image generation workflow.
    
    Returns:
        Final workflow state
    """
    print("🚀 Starting Image Generation Workflow")
    print("=" * 50)
    
    # Create workflow
    workflow = create_workflow()
    
    # Compile workflow without checkpointer for simplicity
    app = workflow.compile()
    
    # Create initial state
    initial_state = create_initial_state()
    
    # Run the workflow
    try:
        final_state = app.invoke(initial_state)
        
        print("\n🎉 Workflow completed successfully!")
        print(f"📁 Output: {final_state.get('output_path', 'N/A')}")
        print(f"📝 Caption: {final_state.get('generated_caption', 'N/A')}")
        
        return final_state
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return {'error': str(e), 'status': 'failed'}


def main():
    """Main function to test the workflow."""
    print("🎨 Image Generation Workflow Test")
    print("=" * 40)
    
    # Validate configuration
    if not Config.validate():
        print("❌ Configuration validation failed")
        return
    
    # Run the workflow
    result = run_workflow()
    
    if result.get('status') == 'complete':
        print("\n✅ Workflow completed successfully!")
    else:
        print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main() 