"""BLIP-2 caption generator for image captioning with text prompts."""

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import time

from ..config.config import Config


class BLIP2CaptionGenerator:
    """Generate captions for images using BLIP-2 model with text prompts."""
    
    def __init__(self):
        """Initialize BLIP-2 model and processor."""
        self.device = torch.device(Config.DEVICE)
        self.model_name = Config.BLIP2_MODEL_NAME
        
        try:
            print(f"🔄 Loading BLIP-2 model: {self.model_name}")
            model_start_time = time.time()
            
            # Load BLIP-2 model and processor
            self.processor = Blip2Processor.from_pretrained(self.model_name, use_fast=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
                device_map="auto" if self.device.type == "mps" else None
            )
            
            if self.device.type != "mps":
                self.model = self.model.to(self.device)
            
            self.model_load_time = time.time() - model_start_time
            print(f"✅ BLIP-2 model loaded successfully! ({self.model_load_time:.2f}s)")
                
        except Exception as e:
            print(f"❌ Error loading BLIP-2 model {self.model_name}: {e}")
            print("Falling back to simple caption generation")
            self.processor = None
            self.model = None
            self.model_load_time = 0
    
    def generate_caption(self, image_path: str, prompt: str = "Describe this image") -> str:
        """Generate caption for an image using a text prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to guide caption generation
            
        Returns:
            Generated caption string
        """
        try:
            # Check if BLIP-2 model is available
            if self.model is None or self.processor is None:
                raise Exception("BLIP-2 model not loaded")
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Process image and text prompt with BLIP-2
            inputs = self.processor(
                images=image, 
                text=prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Reduced to prevent long repetitive outputs
                    num_beams=5,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.0,  # Increased significantly to prevent loops
                    length_penalty=1.0,
                    early_stopping=True,  # Re-enabled to stop at natural endpoints
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )

            # out = self.model.generate(**inputs, max_new_tokens=50, num_beams=1, repetition_penalty=1.2)
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            # caption = caption.strip()
            
            # Debug: Print raw generation info
            print(f"🔍 Raw caption length: {len(caption)} characters")
            print(f"🔍 Generated tokens: {len(out[0])}")
            
            return caption
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            raise 