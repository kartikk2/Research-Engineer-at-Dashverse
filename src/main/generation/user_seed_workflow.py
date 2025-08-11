"""LangGraph workflow for image generation using user-provided seed descriptions."""

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
from ..generation.image_generator import StableDiffusionGenerator
from ..generation.gpt4o_caption_generator import GPT4OCaptionGenerator
from ..generation.gpt4o_prompt_generator import generate_sdxl_prompt_from_seed


class UserWorkflowState(TypedDict):
    """State for the user-seeded image generation workflow."""
    user_seed: str
    art_category: Optional[ArtCategory] 
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


def classify_art_category_node(state: UserWorkflowState) -> UserWorkflowState:
    """Classify the user seed into an art category using substring matching.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with classified art category
    """
    print("🎨 Classifying art category for user seed...")
    
    try:
        # Get user seed from state
        user_seed = state.get('user_seed', '')
        if not user_seed:
            raise ValueError("No user seed provided")
        
        # Convert to lowercase for case-insensitive matching
        user_seed_lower = user_seed.lower()
        
        # Define category keywords and their associated art categories
        category_keywords = {
            ArtCategory.DRAWING: [
                "drawing", "sketch", "pencil", "illustration", "doodle"
            ],
            ArtCategory.ENGRAVING: [
                "engraving", "engraved", "etching", "carved"
            ],
            ArtCategory.ICONOGRAPHY: [
                "icon", "iconography"
            ],
            ArtCategory.PAINTING: [
                "painting", "painted", "oil painting", "watercolor", "acrylic"
            ],
            ArtCategory.SCULPTURE: [
                "sculpture", "sculpted", "statue"
            ]
        }
        
        classified_category = ArtCategory.PAINTING  # Default category
        max_matches = 0
        
        for category, keywords in category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in user_seed_lower)
            if matches > max_matches:
                max_matches = matches
                classified_category = category
        
        # If no specific matches found, use additional heuristics
        if max_matches == 0:
            # Check for general art terms that might indicate painting
            general_painting_terms = ["art", "artwork", "beautiful", "colorful", "vibrant", "detailed"]
            if any(term in user_seed_lower for term in general_painting_terms):
                classified_category = ArtCategory.PAINTING
            else:
                # Default to painting for ambiguous cases
                classified_category = ArtCategory.PAINTING
        
        state['art_category'] = classified_category
        state['status'] = "category_classified"
        print(f"✅ Classified as '{classified_category.value}' (matches: {max_matches}): {user_seed}")
        
    except Exception as e:
        state['error'] = f"Failed to classify art category: {str(e)}"
        state['status'] = "error"
        print(f"❌ Error classifying art category: {e}")
        # Default to painting if classification fails
        state['art_category'] = ArtCategory.PAINTING
    
    return state


def create_user_workflow() -> StateGraph:
    """Create the LangGraph workflow for user-seeded image generation.
    
    Returns:
        Configured StateGraph for the workflow
    """
    # Create the workflow graph
    workflow = StateGraph(UserWorkflowState)
    
    # Add nodes (new entry point: category classification)
    workflow.add_node("classify_art_category", classify_art_category_node)
    workflow.add_node("embed_user_seed", embed_user_seed_node)
    workflow.add_node("find_similar_images", find_similar_images_node)
    workflow.add_node("generate_image_and_caption", generate_image_and_caption_node)
    workflow.add_node("save_results", save_results_node)
    
    # Define the workflow edges
    workflow.set_entry_point("classify_art_category")
    workflow.add_edge("classify_art_category", "embed_user_seed")
    workflow.add_edge("embed_user_seed", "find_similar_images")
    workflow.add_edge("find_similar_images", "generate_image_and_caption")
    workflow.add_edge("generate_image_and_caption", "save_results")
    workflow.add_edge("save_results", END)
    
    return workflow


def embed_user_seed_node(state: UserWorkflowState) -> UserWorkflowState:
    """Embed the user-provided seed description using CLIP text encoder.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with seed embedding
    """
    print("🔤 Embedding user seed description...")
    
    try:
        # Get user seed from state
        user_seed = state.get('user_seed', '')
        if not user_seed:
            raise ValueError("No user seed provided")
        
        # Use CLIP text encoder to embed the description
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        model_name = Config.CLIP_MODEL_NAME
        device = torch.device(Config.DEVICE)
        
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        
        # Process text input
        inputs = processor(text=user_seed, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text embedding
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_embedding = text_features.cpu().numpy().flatten().tolist()
        
        state['seed_embedding'] = text_embedding
        state['status'] = "seed_embedded"
        print(f"✅ Embedded user seed: {user_seed}")
        print(f"✅ Embedding dimensions: {len(text_embedding)}")
        
    except Exception as e:
        state['error'] = f"Failed to embed user seed: {str(e)}"
        state['status'] = "error"
        print(f"❌ Error embedding user seed: {e}")
    
    return state


def find_similar_images_node(state: UserWorkflowState) -> UserWorkflowState:
    """Find similar images from Pinecone using the user seed embedding.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with similar images and descriptions
    """
    print("🔍 Finding similar images...")
    
    try:
        # Initialize vector database ingestor
        vector_ingestor = VectorDBIngestor()
        
        # Get seed embedding
        seed_embedding = state.get('seed_embedding')
        if not seed_embedding:
            raise ValueError("No seed embedding available")
        
        # Get art category from state
        art_category = state.get('art_category')
        print(f"🎨 Using art category filter: {art_category.value if art_category else 'None'}")
        
        # Query for similar images with category filter (top 5)
        similar_images = vector_ingestor.query_similar_images(
            embedding=seed_embedding,
            top_k=3,
            category=art_category.value if art_category else ArtCategory.PAINTING.value
        )
        
        if similar_images:
            state['similar_images'] = similar_images
            state['status'] = "similar_images_found"
            print(f"✅ Found {len(similar_images)} similar images")
            
            # Create list of similar image objects with all metadata
            similar_image_objects = []
            
            for i, img in enumerate(similar_images, 1):
                metadata = img.metadata
                
                # Extract all metadata fields
                description = metadata.get('description', '')
                objects = metadata.get('objects', '')
                people_characters = metadata.get('people_characters', '')
                setting_location = metadata.get('setting_location', '')
                composition_colors = metadata.get('composition_colors', '')
                
                # If no description in metadata, create a fallback description
                if not description:
                    category = metadata.get('category', 'artwork')
                    image_name = metadata.get('image_name', 'unknown')
                    description = f"a {category} artwork featuring image name: {image_name}"
                
                # Create similar image object with all metadata fields
                similar_image = {
                    'image_name': metadata.get('image_name', 'unknown'),
                    'category': metadata.get('category', 'artwork'),
                    'description': description,
                    'objects': objects,
                    'people_characters': people_characters,
                    'setting_location': setting_location,
                    'composition_colors': composition_colors
                }
                
                # Debug: Print all available metadata keys
                if i == 1:  # Only for first image
                    print(f"🔍 DEBUG - Available metadata keys: {list(metadata.keys())}")
                    print(f"🔍 DEBUG - Raw metadata values:")
                    for key, value in metadata.items():
                        print(f"   {key}: '{value}'")
                
                similar_image_objects.append(similar_image)
                
                # Set image_name from the first similar image
                if i == 1:
                    state['image_name'] = similar_image['image_name']
                    # Generate a deterministic image_id from user seed
                    import hashlib
                    seed_hash = hashlib.md5(state.get('user_seed', '').encode()).hexdigest()
                    state['image_id'] = f"user_seed_{seed_hash[:8]}"
                
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
            # Generate a deterministic image_id from user seed
            import hashlib
            seed_hash = hashlib.md5(state.get('user_seed', '').encode()).hexdigest()
            state['image_id'] = f"user_seed_{seed_hash[:8]}"
            state['image_name'] = 'user_generated'
            state['status'] = "no_similar_images"
            print("⚠️  No similar images found")
        
    except Exception as e:
        state['error'] = f"Failed to find similar images: {str(e)}"
        state['status'] = "error"
        print(f"❌ Error finding similar images: {e}")
    
    return state


def generate_image_and_caption_node(state: UserWorkflowState) -> UserWorkflowState:
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
        user_seed = state.get('user_seed', '')
        art_category = state.get('art_category', ArtCategory.PAINTING)  # Get classified category
        image_id = state.get('image_id', 'unknown')
        image_name = state.get('image_name', 'unknown')
        
        # Initialize variables
        generated_image = None
        generated_caption = None
        
        if not similar_images:
            print("⚠️  No similar images available, using base generation")
            # Generate without context using user seed
            print(f"🎲 Using user seed: {user_seed}")
            
            image_generator = StableDiffusionGenerator()
            generated_image = image_generator.generate_image(
                prompt=user_seed,
                num_inference_steps=25,
                guidance_scale=7.5,
                seed=42  # Default seed
            )
            
            # Generate caption without context using classified category
            caption_generator = GPT4OCaptionGenerator()
            generated_caption = caption_generator.generate_caption(
                user_seed, 
                art_category.value,  # Use enum value for caption generation
                []
            )
        else:
            # Use the classified art category from substring matching
            dominant_category = art_category.value  # Get string value from enum
            print(f"🎨 Using classified art category: {dominant_category}")
            
            # Generate optimized SDXL prompt using GPT-4o
            print("🤖 Generating optimized SDXL prompt with GPT-4o...")
            image_prompt = generate_sdxl_prompt_from_seed(
                user_seed=user_seed,
                dominant_category=dominant_category,
                similar_image_objects=similar_image_objects
            )
            
            # Extract metadata from similar image objects for caption generation only
            settings_list = []
            compositions_list = []
            descriptions_list = []
            
            for i, img in enumerate(similar_image_objects):
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
            
            # Debug: Show what was filtered
            print(f"🔍 DEBUG - Filtering results:")
            print(f"   Settings: {len(settings_list)}/{len(similar_image_objects)} passed filter")
            print(f"   Compositions: {len(compositions_list)}/{len(similar_image_objects)} passed filter")
            print(f"   Descriptions: {len(descriptions_list)}/{len(similar_image_objects)} passed filter")
            
            # Create enhanced context prompt for caption generation
            context_prompt = user_seed + f" in {dominant_category} style"
            
            # Add setting/location context for caption
            # if settings_list:
            #     setting_context = ', '.join(settings_list)
            #     context_prompt += f" in settings: {setting_context}"
            
            # # Add composition/colors context for caption
            # if compositions_list:
            #     composition_context = ', '.join(compositions_list)
            #     context_prompt += f" with composition and colors: {composition_context}"
            
            # # Add general descriptions as inspiration for caption
            # if descriptions_list:
            #     context_prompt += f" inspired by: {', '.join(descriptions_list)}"
            
            print(f"🎨 Image generation prompt: {image_prompt}")
            print(f"📝 Caption generation prompt: {context_prompt}")
            print(f"🖼️  Based on user seed: {user_seed}")
            print(f"📝 Using metadata from {len(similar_image_objects)} similar images for caption:")
            print(f"   • Settings: {len(settings_list)}")
            print(f"   • Compositions: {len(compositions_list)}")
            print(f"   • Descriptions: {len(descriptions_list)}")
            
            # Generate image and caption in parallel
            async def generate_parallel():
                # Create tasks for parallel execution
                image_task = asyncio.create_task(generate_image_async(image_prompt, image_id))
                caption_task = asyncio.create_task(generate_caption_async(
                    context_prompt, 
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
        caption = caption_generator.generate_caption(prompt, art_category, [])
        
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


def save_results_node(state: UserWorkflowState) -> UserWorkflowState:
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
        user_seed = state.get('user_seed', 'No user seed')
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if generated_image:
            # Create output directory
            output_dir = Path("generated_images")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            filename = f"user_generated_{workflow_id}_{timestamp}.png"
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
        
        caption_filename = f"user_caption_{workflow_id}_{timestamp}.txt"
        caption_path = caption_dir / caption_filename
        
        # Get art category from workflow state (classified by substring matching)
        art_category = state.get('art_category', ArtCategory.PAINTING)
        art_category_str = art_category.value if art_category else 'artwork'
        
        with open(caption_path, 'w') as f:
            f.write(f"Generated Image: {state.get('output_path', 'N/A')}\n")
            f.write(f"User Seed: {user_seed}\n")
            f.write(f"Art Category: {art_category_str}\n")
            f.write(f"Generated Image ID: {image_id}\n")
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


def create_user_initial_state(user_seed: str) -> UserWorkflowState:
    """Create initial state for the user-seeded workflow.
    
    Args:
        user_seed: The user-provided seed description
        
    Returns:
        Initial workflow state
    """
    return {
        'user_seed': user_seed,
        'art_category': None,  # Will be set by classify_art_category_node
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


def run_user_workflow(user_seed: str) -> Dict[str, Any]:
    """Run the user-seeded image generation workflow.
    
    Args:
        user_seed: User-provided seed description
        
    Returns:
        Final workflow state
    """
    print("🚀 Starting User-Seeded Image Generation Workflow")
    print("=" * 60)
    print(f"🎯 User Seed: {user_seed}")
    
    # Create workflow
    workflow = create_user_workflow()
    
    # Compile workflow without checkpointer for simplicity
    app = workflow.compile()
    
    # Create initial state with user seed
    initial_state = create_user_initial_state(user_seed)
    
    # Run the workflow
    try:
        final_state = app.invoke(initial_state)
        
        print("\n🎉 User workflow completed successfully!")
        print(f"📁 Output: {final_state.get('output_path', 'N/A')}")
        print(f"📝 Caption: {final_state.get('generated_caption', 'N/A')}")
        
        return final_state
        
    except Exception as e:
        print(f"❌ User workflow failed: {e}")
        return {'error': str(e), 'status': 'failed'}


if __name__ == "__main__":
    """Test the user workflow with a sample seed."""
    test_seed = "a majestic dragon soaring over a medieval castle at sunset"
    result = run_user_workflow(test_seed)
    
    if result.get('status') == 'complete':
        print("\n✅ User workflow test completed successfully!")
    else:
        print(f"\n❌ User workflow test failed: {result.get('error', 'Unknown error')}") 