"""Image generation using Stable Diffusion with textual prompts."""

import sys
from pathlib import Path

# Add the src directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import time
from typing import Optional, List, Dict

from main.config.config import Config


class StableDiffusionGenerator:
    """Generate images using Stable Diffusion with textual prompts."""
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Initialize Stable Diffusion v1.5 model.
        
        Args:
            model_name: Name of the Stable Diffusion model to use (default: v1-5)
        """
        # Use MPS if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 Using MPS (Apple Silicon GPU) for faster generation")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU for generation")
        
        self.model_name = model_name
        
        try:
            print(f"🔄 Loading Stable Diffusion v1.5 model: {self.model_name}")
            model_start_time = time.time()
            
            # Load Stable Diffusion v1.5 pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            # Use DPM++ 2M scheduler for faster inference
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                solver_type="midpoint"
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            
            self.model_load_time = time.time() - model_start_time
            print(f"✅ Stable Diffusion v1.5 model loaded successfully! ({self.model_load_time:.2f}s)")
            print(f"📱 Device: {self.device}")
            print(f"🤖 Model: {self.model_name}")
                
        except Exception as e:
            print(f"❌ Error loading Stable Diffusion v1.5 model {self.model_name}: {e}")
            self.pipeline = None
            self.model_load_time = 0
    
    def get_art_style_prompt(self, base_prompt: str, art_category: str) -> str:
        """Get optimized prompt for specific art category.
        
        Args:
            base_prompt: Base description of the image
            art_category: Art category (drawing, engraving, iconography, painting, sculpture)
            
        Returns:
            Optimized prompt for the art category
        """
        style_prompts = {
            "drawing": f"pencil drawing, sketch style, {base_prompt}, detailed linework, artistic drawing",
            "engraving": f"engraved artwork, etching style, {base_prompt}, fine details, printmaking technique",
            "iconography": f"religious iconography, symbolic art, {base_prompt}, spiritual artwork, traditional icon style",
            "painting": f"oil painting, artistic masterpiece, {base_prompt}, painterly style, canvas texture",
            "sculpture": f"sculptural artwork, 3D form, {base_prompt}, carved details, dimensional art"
        }
        
        return style_prompts.get(art_category.lower(), base_prompt)
    
    def generate_image(
        self, 
        prompt: str, 
        negative_prompt: str = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = None
    ) -> Optional[Image.Image]:
        """Generate an image from a textual prompt.
        
        Args:
            prompt: Textual description of the image to generate
            negative_prompt: Textual description of what to avoid in the image
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            width: Width of the generated image
            height: Height of the generated image
            seed: Random seed for reproducible results
            
        Returns:
            Generated PIL Image or None if generation failed
        """
        if self.pipeline is None:
            print("❌ Stable Diffusion model not loaded")
            return None
        
        try:
            print(f"🎨 Generating image with prompt: '{prompt}'")
            generation_start_time = time.time()
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
            
            # Create generator - use CPU for MPS compatibility
            if seed is not None:
                if self.device.type == "mps":
                    # For MPS, create generator on CPU for compatibility
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                else:
                    generator = torch.Generator(device=self.device).manual_seed(seed)
            else:
                generator = None
            
            # Generate image
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
            
            generation_time = time.time() - generation_start_time
            print(f"✅ Image generated successfully! ({generation_time:.2f}s)")
            
            return result.images[0]
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            return None
    
    def generate_image_with_style(
        self,
        base_prompt: str,
        art_category: str,
        negative_prompt: str = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = None
    ) -> Optional[Image.Image]:
        """Generate an image with specific art style using prompt engineering.
        
        Args:
            base_prompt: Base description of the image
            art_category: Art category (drawing, engraving, iconography, painting, sculpture)
            negative_prompt: Textual description of what to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            width: Width of the generated image
            height: Height of the generated image
            seed: Random seed for reproducible results
            
        Returns:
            Generated PIL Image or None if generation failed
        """
        if self.pipeline is None:
            print("❌ Stable Diffusion model not loaded")
            return None
        
        try:
            # Get style-optimized prompt
            style_prompt = self.get_art_style_prompt(base_prompt, art_category)
            
            print(f"🎨 Generating {art_category} style image with prompt: '{style_prompt}'")
            
            # Generate image
            image = self.generate_image(
                prompt=style_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed
            )
            
            return image
            
        except Exception as e:
            print(f"❌ Error generating {art_category} style image: {e}")
            return None
    
    def save_image(self, image: Image.Image, filepath: str) -> bool:
        """Save generated image to file.
        
        Args:
            image: PIL Image to save
            filepath: Path where to save the image
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            image.save(filepath)
            print(f"💾 Image saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return False
    
    def generate_and_save(
        self, 
        prompt: str, 
        output_path: str,
        negative_prompt: str = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = None
    ) -> bool:
        """Generate an image and save it to file.
        
        Args:
            prompt: Textual description of the image to generate
            output_path: Path where to save the generated image
            negative_prompt: Textual description of what to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for prompt adherence
            width: Width of the generated image
            height: Height of the generated image
            seed: Random seed for reproducible results
            
        Returns:
            True if generation and saving was successful, False otherwise
        """
        image = self.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed
        )
        
        if image is None:
            return False
        
        return self.save_image(image, output_path)


def main():
    """Test function for Stable Diffusion v1.5 image generation."""
    print("🎨 Stable Diffusion v1.5 Image Generation Test")
    print("=" * 50)
    
    # Initialize generator
    generator = StableDiffusionGenerator()
    
    if generator.pipeline is None:
        print("❌ Failed to load Stable Diffusion v1.5 model")
        return
    
    # Test base prompt
    base_prompt = "a serene landscape with mountains and a lake"
    
    # Test different art categories
    art_categories = ["drawing", "engraving", "iconography", "painting", "sculpture"]
    
    # Negative prompts for better quality
    negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, watermark, signature, text"
    
    print("\n🎨 Testing art style generation:")
    print("=" * 40)
    
    # Generate images for each art category
    for i, category in enumerate(art_categories):
        print(f"\n🔄 Testing {category} style...")
        
        # Generate image with style
        image = generator.generate_image_with_style(
            base_prompt=base_prompt,
            art_category=category,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            width=512,
            height=512,
            seed=42 + i
        )
        
        if image:
            # Save image
            output_path = f"sd_{category}_style_image.png"
            if generator.save_image(image, output_path):
                print(f"✅ Successfully generated {category} style: {output_path}")
            else:
                print(f"❌ Failed to save {category} style image")
        else:
            print(f"❌ Failed to generate {category} style image")
    
    print("\n🎉 Art style generation test completed!")
    print("📁 Check the generated images in the current directory")


if __name__ == "__main__":
    main() 