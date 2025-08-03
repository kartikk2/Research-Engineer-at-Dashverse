"""GPT-4o prompt generator for creating optimized Stable Diffusion XL prompts."""

import os
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class GPT4OPromptGenerator:
    """Generates optimized Stable Diffusion XL prompts using GPT-4o."""
    
    def __init__(self):
        """Initialize the GPT-4o prompt generator."""
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.4,
            max_tokens=100,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def generate_sdxl_prompt(
        self, 
        user_seed: str, 
        dominant_category: str,
        similar_image_objects: List[Dict]
    ) -> str:
        """Generate an optimized Stable Diffusion XL prompt.
        
        Args:
            user_seed: The user's original seed description
            dominant_category: The dominant art category from similar images
            similar_image_objects: List of similar image objects with metadata
            
        Returns:
            Optimized SDXL prompt string
        """
        # Extract relevant metadata from similar images
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
        
        # Create context from similar images
        context_parts = []
        if settings_list:
            context_parts.append(f"Setting and Location: {', '.join(settings_list[:3])}")
        if compositions_list:
            context_parts.append(f"Composition and Colors: {', '.join(compositions_list[:3])}")
        if descriptions_list:
            context_parts.append(f"Similar descriptions: {', '.join(descriptions_list[:2])}")
        
        similar_context = " | ".join(context_parts) if context_parts else "No similar context available"
        
        # Create the system prompt
        system_prompt = """You are an expert at creating optimized prompts for Stable Diffusion XL (SDXL) image generation. 

Your task is to transform a user's seed description into a crisp, SDXL-optimized prompt that will generate high-quality images.

IMPORTANT GUIDELINES:
1. EMPHASIZE on objects, characters and composition from the user's seed
2. Reference the art category and similar image context for style consistency
3. Use commas to separate different aspects
4. Respond under 50 tokens.

STRUCTURE:
- Key objects and characters (emphasize these from user seed)
- Background and setting

Focus on objects, characters and composition as the most prominent aspects of the generated prompt."""

        # Create the human prompt
        human_prompt = f"""User Seed: "{user_seed}"
Art Category: {dominant_category}
Similar Image Context: {similar_context}

Generate an optimized SDXL prompt that emphasizes the objects, characters and composition mentioned in the user seed. Use the art category and similar image context to inform the style and technique.

Focus on making the user's key elements stand out while maintaining artistic quality and coherence."""

        try:
            # Generate the prompt
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            generated_prompt = response.content.strip()
            
            # Clean up the prompt
            generated_prompt = self._clean_prompt(generated_prompt)
            
            return generated_prompt
            
        except Exception as e:
            print(f"❌ Error generating SDXL prompt: {e}")
            # Fallback to simple prompt
            return f"{user_seed}, highly detailed, professional {dominant_category}, 8k uhd, masterpiece"
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and format the generated prompt.
        
        Args:
            prompt: Raw generated prompt
            
        Returns:
            Cleaned prompt string
        """
        # Remove any markdown formatting
        prompt = prompt.replace("```", "").replace("**", "").replace("*", "")
        
        # Remove quotes if present
        if prompt.startswith('"') and prompt.endswith('"'):
            prompt = prompt[1:-1]
        
        # Clean up extra whitespace
        prompt = " ".join(prompt.split())
        
        return prompt


def generate_sdxl_prompt_from_seed(
    user_seed: str, 
    dominant_category: str,
    similar_image_objects: List[Dict]
) -> str:
    """Convenience function to generate SDXL prompt.
    
    Args:
        user_seed: The user's original seed description
        dominant_category: The dominant art category from similar images
        similar_image_objects: List of similar image objects with metadata
        
    Returns:
        Optimized SDXL prompt string
    """
    generator = GPT4OPromptGenerator()
    return generator.generate_sdxl_prompt(user_seed, dominant_category, similar_image_objects) 