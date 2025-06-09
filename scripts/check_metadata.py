"""
Script to check and decompress video metadata files.
"""

import json
import logging
import lzma
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_metadata(metadata_dir: str = "data/videos") -> None:
    """
    Check and decompress video metadata files.
    
    Args:
        metadata_dir: Directory containing video metadata files
    """
    try:
        metadata_path = Path(metadata_dir)
        if not metadata_path.exists():
            logger.error(f"Metadata directory not found: {metadata_dir}")
            return
        
        # Find all .json.xz files
        metadata_files = list(metadata_path.glob("*.json.xz"))
        logger.info(f"Found {len(metadata_files)} metadata files")
        
        # Process each file
        for file_path in metadata_files:
            try:
                # Decompress and load JSON
                with lzma.open(file_path, 'rt', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Check required fields
                video_id = file_path.stem.split('.')[0]
                logger.info(f"\nChecking metadata for video: {video_id}")
                
                # Print metadata structure
                logger.info("Metadata structure:")
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        logger.info(f"  {key}: {list(value.keys())}")
                    elif isinstance(value, list):
                        logger.info(f"  {key}: [{len(value)} items]")
                    else:
                        logger.info(f"  {key}: {type(value).__name__}")
                
                # Save decompressed JSON
                json_path = file_path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved decompressed metadata to {json_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue
        
        logger.info("Metadata check complete")
        
    except Exception as e:
        logger.error(f"Metadata check failed: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check video metadata files")
    parser.add_argument("--dir", default="data/videos", help="Directory containing metadata files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    check_metadata(args.dir) 