"""
Script to extract relevant video information from metadata files.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_video_info(metadata_dir: str = "data/videos") -> None:
    """
    Extract relevant video information from metadata files.
    
    Args:
        metadata_dir: Directory containing video metadata files
    """
    try:
        metadata_path = Path(metadata_dir)
        if not metadata_path.exists():
            logger.error(f"Metadata directory not found: {metadata_dir}")
            return
        
        # Find all .json files (decompressed metadata)
        metadata_files = list(metadata_path.glob("*.json.json"))
        logger.info(f"Found {len(metadata_files)} metadata files")
        
        # Process each file
        for file_path in metadata_files:
            try:
                # Load metadata
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                
                video_id = file_path.stem.split('.')[0]
                logger.info(f"\nProcessing video: {video_id}")
                
                # Extract relevant information
                node = metadata.get('node', {})
                video_info = {
                    'video_id': video_id,
                    'dimensions': node.get('dimensions', {}),
                    'caption': next(
                        (edge['node']['text'] for edge in node.get('edge_media_to_caption', {}).get('edges', [])),
                        None
                    ),
                    'hashtags': extract_hashtags(node),
                    'brand': node.get('owner', {}).get('username'),
                    'brand_name': node.get('owner', {}).get('full_name'),
                    'video_url': node.get('video_url'),
                    'thumbnail_url': node.get('display_url')
                }
                
                # Save extracted info
                output_path = file_path.with_suffix('.info.json')
                with open(output_path, 'w') as f:
                    json.dump(video_info, f, indent=2)
                logger.info(f"Saved video info to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                continue
        
        logger.info("Video info extraction complete")
        
    except Exception as e:
        logger.error(f"Video info extraction failed: {str(e)}")

def extract_hashtags(node: Dict[str, Any]) -> List[str]:
    """
    Extract hashtags from video caption.
    
    Args:
        node: Video metadata node
        
    Returns:
        List of hashtags
    """
    hashtags = []
    caption_edges = node.get('edge_media_to_caption', {}).get('edges', [])
    
    for edge in caption_edges:
        text = edge.get('node', {}).get('text', '')
        hashtags.extend(word for word in text.split() if word.startswith('#'))
    
    return hashtags

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract video information from metadata")
    parser.add_argument("--dir", default="data/videos", help="Directory containing metadata files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    extract_video_info(args.dir) 