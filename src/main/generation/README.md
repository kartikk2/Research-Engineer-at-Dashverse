# Image Generation Workflow

This directory contains the LangGraph workflow for generating images using seed descriptions and similar images from the vector database.

## 🎯 Overview

The workflow performs the following steps:

1. **Generate Random Seed**: Uses `SeedGenerator` to create a random seed description
2. **Embed Seed**: Converts the seed description to a CLIP text embedding
3. **Find Similar Images**: Queries Pinecone for top-K similar images using the embedding
4. **Generate Image & Caption**: Creates an image and caption in parallel using Stable Diffusion and BLIP-2
5. **Save Results**: Saves the generated image and logs the caption

## 📁 Files

- `image_generation_workflow.py` - Main LangGraph workflow implementation
- `seed_generator.py` - Generates random seed descriptions
- `image_generator.py` - Stable Diffusion image generation
- `caption_generator.py` - BLIP-2 caption generation

## 🚀 Usage

### Basic Usage

```python
from main.generation.image_generation_workflow import run_workflow

# Run the complete workflow
result = run_workflow()

if result.get('status') == 'complete':
    print(f"Generated image: {result.get('output_path')}")
    print(f"Generated caption: {result.get('generated_caption')}")
```

### Test Script

```bash
# Run the test script
python test_image_generation_workflow.py
```

### Direct Module Execution

```bash
# Run the workflow module directly
python -m src.main.generation.image_generation_workflow
```

## 🔧 Workflow Steps

### 1. Generate Random Seed
- Uses `SeedGenerator` to select a random image from the dataset
- Extracts or creates a description from the selected image
- Falls back to a generic description if no seed is found

### 2. Embed Seed Description
- Uses CLIP text encoder to convert the description to a vector embedding
- Supports the same CLIP model used for image embeddings
- Creates a 512-dimensional embedding vector

### 3. Find Similar Images
- Queries Pinecone vector database using the seed embedding
- Returns top 5 similar images with their metadata
- Includes image names, categories, and descriptions

### 4. Generate Image & Caption (Parallel)
- **Image Generation**: Uses Stable Diffusion with enhanced prompts based on similar images
- **Caption Generation**: Uses BLIP-2 to generate descriptive captions
- Both processes run in parallel for efficiency

### 5. Save Results
- Saves generated image to `generated_images/` directory
- Saves caption and metadata to `generated_captions/` directory
- Uses timestamped filenames for organization

## 📊 Output Structure

```
generated_images/
├── generated_<workflow_id>_<timestamp>.png
└── ...

generated_captions/
├── caption_<workflow_id>_<timestamp>.txt
└── ...
```

### Caption File Format

```
Generated Image: generated_images/generated_<id>_<timestamp>.png
Seed Description: a painting artwork similar to image_001
Generated Caption: A beautiful oil painting depicting...
Timestamp: 2024-01-15T10:30:45.123456
Workflow ID: <uuid>
```

## ⚙️ Configuration

The workflow uses the following configuration from `Config`:

- `CLIP_MODEL_NAME`: CLIP model for text/image embeddings
- `DEVICE`: Device for model inference (MPS/CPU)
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_INDEX_NAME`: Pinecone index name
- `PINECONE_HOST`: Pinecone host URL

## 🔄 State Management

The workflow uses LangGraph's state management with the following state structure:

```python
class WorkflowState(TypedDict):
    seed_description: Optional[str]
    seed_embedding: Optional[List[float]]
    similar_images: Optional[List[Dict]]
    generated_image: Optional[Any]  # PIL Image
    generated_caption: Optional[str]
    output_path: Optional[str]
    workflow_id: str
    timestamp: str
    status: str
    error: Optional[str]
```

## 🎨 Enhanced Prompts

The workflow creates enhanced prompts by combining:

1. **Base Seed Description**: Original description from the seed
2. **Similar Image Context**: Descriptions from similar images
3. **Style Information**: Dominant art category from similar images

Example enhanced prompt:
```
"a painting artwork similar to image_001 inspired by: detailed landscape with mountains, vibrant sunset colors in painting style"
```

## 🚨 Error Handling

The workflow includes comprehensive error handling:

- **Seed Generation**: Falls back to generic description if no seed found
- **Embedding**: Handles CLIP model loading errors
- **Similar Images**: Continues with base generation if no similar images found
- **Generation**: Handles model loading and generation errors
- **Saving**: Creates directories and handles file I/O errors

## 📈 Performance

- **Parallel Processing**: Image and caption generation run concurrently
- **Model Reuse**: Models are loaded once and reused across the workflow
- **Memory Management**: Proper cleanup of temporary files and resources
- **Checkpointing**: LangGraph memory saver for workflow state persistence

## 🔍 Debugging

Enable debug output by checking the console logs:

```
🎲 Generating random seed...
✅ Generated seed: a painting artwork similar to image_001
🔤 Embedding seed description...
✅ Embedded seed description (dimensions: 512)
🔍 Finding similar images...
✅ Found 5 similar images
🎨 Generating image and caption in parallel...
✅ Generated image and caption: A beautiful oil painting...
💾 Saving results...
✅ Image saved to: generated_images/generated_<id>_<timestamp>.png
📝 Caption saved to: generated_captions/caption_<id>_<timestamp>.txt
```

## 🛠️ Dependencies

- `langgraph`: Workflow orchestration
- `torch`: PyTorch for model inference
- `transformers`: CLIP and BLIP-2 models
- `diffusers`: Stable Diffusion pipeline
- `pinecone-client`: Vector database operations
- `PIL`: Image processing 