#!/usr/bin/env python3
"""Launcher script for the AI Art Generator Gradio interface."""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from main.gradio_app import main
    main() 