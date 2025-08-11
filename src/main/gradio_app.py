"""Gradio UI for user-seeded image generation workflow."""

import gradio as gr
import sys
from pathlib import Path
from typing import Tuple, Optional
import time

# Add the src directory to the Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from main.config.config import Config
from main.generation.user_seed_workflow import run_user_workflow


def generate_image_from_seed(user_seed: str, progress=gr.Progress()) -> Tuple[Optional[str], str, str]:
    """Generate image from user-provided seed description.
    
    Args:
        user_seed: User's seed description
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (image_path, caption, status_message)
    """
    if not user_seed.strip():
        return None, "", "❌ Please provide a seed description"
    
    try:
        progress(0.1, desc="🚀 Starting workflow...")
        
        # Validate configuration
        if not Config.validate():
            return None, "", "❌ Configuration validation failed. Check your environment variables."
        
        progress(0.2, desc="🔤 Embedding user seed...")
        
        # Run the user workflow
        result = run_user_workflow(user_seed)
        
        progress(0.8, desc="💾 Saving results...")
        
        if result.get('status') == 'complete':
            # Get the generated image path
            image_path = result.get('output_path')
            caption = result.get('generated_caption', 'No caption generated')
            
            if image_path and Path(image_path).exists():
                progress(1.0, desc="✅ Generation complete!")
                status_msg = f"🎉 Successfully generated image from: '{user_seed}'"
                return image_path, caption, status_msg
            else:
                return None, caption, "⚠️ Image generated but file not found"
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            return None, "", f"❌ Generation failed: {error_msg}"
            
    except Exception as e:
        return None, "", f"❌ Unexpected error: {str(e)}"


def create_gradio_interface():
    """Create the Gradio interface for image generation.
    
    Returns:
        Configured Gradio interface
    """
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        color: #2c3e50;
    }
    .description-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .example-seeds {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=css, title="AI Art Generator", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎨 AI Art Generator</h1>
            <h3>Transform your ideas into beautiful artwork</h3>
        </div>
        """)
        
        # Description
        gr.HTML("""
        <div class="description-box">
            <h4>✨ How it works:</h4>
            <ul>
                <li>Enter a detailed description of the artwork you want to create</li>
                <li>Our multimodal AI generates a unique image and it's caption based on your description</li>
            </ul>
        </div>
        """)
        
        # Example seeds
        # gr.HTML("""
        # <div class="example-seeds">
        #     <h4>💡 Example descriptions:</h4>
        #     <ul>
        #         <li>"A painting of a dragon flying over a medieval castle at sunset with golden clouds"</li>
        #         <li>"A serene Japanese garden with cherry blossoms falling on a stone lantern"</li>
        #         <li>"A futuristic cityscape with flying cars and neon lights reflecting in rain puddles"</li>
        #         <li>"A Renaissance-style portrait of a mysterious woman with flowing red hair and emerald eyes"</li>
        #     </ul>
        # </div>
        # """)
        
        # Input section (above everything)
        seed_input = gr.Textbox(
            label="🎯 Describe your artwork",
            placeholder="Enter a detailed description of the artwork you want to create...",
            lines=4,
            max_lines=6,
            info="Be specific about style, colors, mood, and elements you want to see"
        )
        
        generate_btn = gr.Button(
            "🎨 Generate Artwork",
            variant="primary",
            size="lg"
        )
        
        # Status display
        status_output = gr.Textbox(
            label="📊 Status",
            interactive=False,
            lines=2
        )
        
        # Output section (side by side)
        with gr.Row():
            with gr.Column(scale=1):
                # Generated image
                image_output = gr.Image(
                    label="🎨 Generated Artwork",
                    type="filepath",
                    height=400
                )
            
            with gr.Column(scale=1):
                # Generated caption
                caption_output = gr.Textbox(
                    label="📝 Generated Caption",
                    interactive=False,
                    lines=8
                )
        
        # Connect the button to the function
        generate_btn.click(
            fn=generate_image_from_seed,
            inputs=[seed_input],
            outputs=[image_output, caption_output, status_output]
        )
        
        # Allow Enter key to trigger generation
        seed_input.submit(
            fn=generate_image_from_seed,
            inputs=[seed_input],
            outputs=[image_output, caption_output, status_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; color: #6c757d;">
            <p>Powered by Stable Diffusion, CLIP, GPT-4o, and Pinecone</p>
        </div>
        """)
    
    return interface


def main():
    """Launch the Gradio interface."""
    print("🚀 Launching AI Art Generator Gradio Interface")
    print("=" * 50)
    
    # Validate configuration
    if not Config.validate():
        print("❌ Configuration validation failed")
        print("Please check your environment variables:")
        print("- OPENAI_API_KEY")
        print("- PINECONE_API_KEY")
        print("- PINECONE_ENVIRONMENT")
        print("- PINECONE_INDEX_NAME")
        return
    
    print("✅ Configuration validated successfully")
    print("🌐 Starting Gradio server...")
    
    # Create and launch the interface
    interface = create_gradio_interface()
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Don't create public link
        show_error=True,        # Show detailed errors
        quiet=False,            # Show server logs
        inbrowser=True          # Open browser automatically
    )


if __name__ == "__main__":
    main() 