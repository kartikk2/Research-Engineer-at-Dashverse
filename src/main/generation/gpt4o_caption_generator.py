"""GPT-4o caption generator for creating professional art descriptions."""

import os
from typing import List, Optional

from ..config.config import Config

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  LangChain OpenAI package not installed. Using fallback caption generation.")


class GPT4OCaptionGenerator:
    """Generate professional art captions using GPT-4o."""
    
    def __init__(self):
        """Initialize the GPT-4o caption generator."""
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the GPT-4o language model."""
        if not OPENAI_AVAILABLE:
            print("❌ LangChain OpenAI not available")
            return
        
        try:
            if not Config.OPENAI_API_KEY:
                print("❌ OPENAI_API_KEY not found in environment variables")
                return
            
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.4,
                max_tokens=80,
                api_key=Config.OPENAI_API_KEY
            )
            print("✅ GPT-4o caption generator initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing GPT-4o: {e}")
            self.llm = None
    
    def generate_caption(self, prompt: str, art_category: str, similar_descriptions: List[str] = None) -> str:
        """Generate a professional art caption using GPT-4o.
        
        Args:
            prompt: The original generation prompt
            art_category: The art category (painting, sculpture, etc.)
            similar_descriptions: List of similar image descriptions for context
            
        Returns:
            Generated caption string
        """
        if not self.llm:
            print("❌ GPT-4o not available, using fallback caption")
            return self._generate_fallback_caption(prompt, art_category, similar_descriptions)
        
        try:
            # Create context-aware prompt for caption generation
            if similar_descriptions:
                similar_context = ', '.join(similar_descriptions)
                caption_prompt = f"""You are an expert in generating captions for professional art images. Generation a caption for a generated {art_category} artwork.

Image Generation Prompt: {prompt}
Image style: {art_category}
Inspiration Sources: {similar_context}

Write a caption of around 25 words that describes:
- The visual elements and composition
- The objects in the image
- The background of the image
- The image style and technique specific to {art_category}
- The mood and atmosphere

Important note: In your generated caption, do NOT mention any image names that are in the inspiration sources.

Make it sound like a professional art gallery description:"""
            else:
                caption_prompt = f"""You are an expert art curator. Create a detailed, professional caption for a generated {art_category} artwork.

Generation Prompt: {prompt}
Art Category: {art_category}

Write a comprehensive caption (2-3 sentences) that describes:
- The visual elements and composition
- The artistic style and technique specific to {art_category}
- The mood and atmosphere

Make it sound like a professional art gallery description:"""
            
            # Print the prompt being sent to GPT-4o
            print(f"📝 GPT-4o Caption Prompt:")
            print(f"{'='*50}")
            print(caption_prompt)
            print(f"{'='*50}")
            
            # Generate caption using GPT-4o
            response = self.llm.invoke(caption_prompt)
            caption = response.content.strip()
            
            print(f"🤖 GPT-4o generated caption: {caption}")
            return caption
            
        except Exception as e:
            print(f"❌ Error generating caption with GPT-4o: {e}")
            return self._generate_fallback_caption(prompt, art_category, similar_descriptions)
    
    def _generate_fallback_caption(self, prompt: str, art_category: str, similar_descriptions: List[str] = None) -> str:
        """Generate a fallback caption when GPT-4o is not available.
        
        Args:
            prompt: The original prompt used for generation
            art_category: The art category (painting, sculpture, etc.)
            similar_descriptions: List of similar image descriptions for context
            
        Returns:
            Fallback caption string
        """
        if similar_descriptions:
            similar_context = ', '.join(similar_descriptions[:2])
            return f"A {prompt.lower()}. This {art_category} artwork draws inspiration from similar pieces that feature {similar_context}. The composition combines elements from these reference works to create a unique and cohesive piece."
        else:
            return f"A {prompt.lower()}. This {art_category} artwork showcases artistic creativity and visual appeal."
    



def main():
    """Test the GPT-4o caption generator."""
    print("🤖 GPT-4o Caption Generator Test")
    print("=" * 40)
    
    # Initialize caption generator
    caption_gen = GPT4OCaptionGenerator()
    
    # Test basic caption generation
    print("\n🔍 Testing basic caption generation:")
    test_prompt = "Vibrant brushstrokes reveal nature's essence in luminous, swirling colors"
    test_similar = ["a painting of a cityscape", "abstract composition with bold colors"]
    
    caption = caption_gen.generate_caption(test_prompt, "painting", test_similar)
    print(f"📝 Generated caption: {caption}")


if __name__ == "__main__":
    main() 