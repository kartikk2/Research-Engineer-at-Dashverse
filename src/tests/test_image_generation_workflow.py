"""Test script for the LangGraph image generation workflow."""

import sys
from pathlib import Path

# Add the src directory to the Python path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from main.generation.image_generation_workflow import run_workflow


def main():
    """Test the image generation workflow."""
    print("🎨 Testing LangGraph Image Generation Workflow")
    print("=" * 60)
    
    try:
        # Run the workflow
        result = run_workflow()
        
        if result.get('status') == 'complete':
            print("\n✅ Workflow test completed successfully!")
            print(f"📁 Generated image: {result.get('output_path', 'N/A')}")
            print(f"📝 Generated caption: {result.get('generated_caption', 'N/A')}")
        else:
            print(f"\n❌ Workflow test failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 