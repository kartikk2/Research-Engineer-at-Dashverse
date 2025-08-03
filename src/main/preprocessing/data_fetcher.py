"""Data fetching utilities for Kaggle Hub."""

import kaggle
from pathlib import Path


def download_dataset(dataset_name: str, output_dir: str = "data") -> str:
    """Download a dataset from Kaggle.
    
    Args:
        dataset_name: Dataset name in format 'username/dataset-name'
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    kaggle.api.dataset_download_files(dataset_name, path=output_path, unzip=True)
    
    return str(output_path)


def main() -> None:
    """Main function to download sample dataset."""
    try:
        path = download_dataset("thedownhill/art-images-drawings-painting-sculpture-engraving")
        print("Path to dataset files:", path)
    except Exception as e:
        print(f"Error downloading dataset: {e}")


if __name__ == "__main__":
    main() 