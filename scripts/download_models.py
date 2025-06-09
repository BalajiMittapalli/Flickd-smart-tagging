"""
Script to download pre-trained models for the fashion detection system.
"""

import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_models():
    """Download pre-trained models."""
    try:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Download the valentinafeve/yolos-fashionpedia model
        logger.info("Downloading valentinafeve/yolos-fashionpedia model...")
        model_path = hf_hub_download(
            repo_id="valentinafeve/yolos-fashionpedia",
            filename="pytorch_model.bin",
            local_dir=models_dir
        )
        logger.info(f"Saved model to {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to download models: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        download_models()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1) 