#!/usr/bin/env python3
"""Test script for the Gradio interface."""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from main.gradio_app import generate_image_from_seed, create_gradio_interface
        from main.generation.user_seed_workflow import run_user_workflow
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_workflow():
    """Test the user workflow with a simple seed."""
    try:
        from main.generation.user_seed_workflow import run_user_workflow
        
        test_seed = "a simple test image of a red apple on a white background"
        print(f"🧪 Testing workflow with seed: '{test_seed}'")
        
        result = run_user_workflow(test_seed)
        
        if result.get('status') == 'complete':
            print("✅ Workflow test successful!")
            print(f"📁 Output: {result.get('output_path', 'N/A')}")
            print(f"📝 Caption: {result.get('generated_caption', 'N/A')[:100]}...")
            return True
        else:
            print(f"❌ Workflow test failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Workflow test error: {e}")
        return False

def test_gradio_function():
    """Test the Gradio function interface."""
    try:
        from main.gradio_app import generate_image_from_seed
        
        test_seed = "a blue flower in a garden"
        print(f"🧪 Testing Gradio function with seed: '{test_seed}'")
        
        # Mock progress function
        def mock_progress(progress, desc=""):
            print(f"📊 Progress: {progress*100:.0f}% - {desc}")
        
        image_path, caption, status = generate_image_from_seed(test_seed, mock_progress)
        
        if image_path and "Successfully generated" in status:
            print("✅ Gradio function test successful!")
            print(f"📁 Image: {image_path}")
            print(f"📝 Caption: {caption[:100]}...")
            return True
        else:
            print(f"❌ Gradio function test failed: {status}")
            return False
            
    except Exception as e:
        print(f"❌ Gradio function test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Gradio Interface Components")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Workflow Test", test_workflow),
        ("Gradio Function Test", test_gradio_function)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The Gradio interface is ready to use.")
        print("🌐 You can now run: python run_gradio_app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main() 