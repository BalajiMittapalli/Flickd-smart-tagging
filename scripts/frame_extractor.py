"""
Frame extraction module for video processing.
Extracts frames from videos at specified intervals and saves them as JPEG images.
"""

import cv2
import os
import logging
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FrameExtractionError(Exception):
    """Custom exception for frame extraction errors."""
    pass

def extract_frames(
    video_path: str,
    output_dir: str,
    interval: int = 5,
    max_frames: Optional[int] = None
) -> List[str]:
    """
    Extract frames from a video file and save them as JPEG images.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        interval (int): Extract every Nth frame (default=5)
        max_frames (Optional[int]): Maximum number of frames to extract (default=None)
    
    Returns:
        List[str]: List of paths to the extracted frame images
        
    Raises:
        FrameExtractionError: If there's an error during frame extraction
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Open the video file
        logger.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FrameExtractionError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        # Create progress bar
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Check if we've reached the maximum frame limit
                if max_frames is not None and saved_count >= max_frames:
                    break
                
                # Save every Nth frame
                if frame_count % interval == 0:
                    frame_path = output_path / f"frame_{frame_count:06d}.jpg"
                    success = cv2.imwrite(str(frame_path), frame)
                    
                    if not success:
                        logger.warning(f"Failed to save frame {frame_count}")
                    else:
                        frame_paths.append(str(frame_path))
                        saved_count += 1
                
                frame_count += 1
                pbar.update(1)
                
        finally:
            cap.release()
            pbar.close()
        
        logger.info(f"Extracted {saved_count} frames to {output_dir}")
        return frame_paths
        
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        raise FrameExtractionError(f"Frame extraction failed: {str(e)}")

def validate_video(video_path: str) -> bool:
    """
    Validate if a video file is valid and can be processed.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        bool: True if video is valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        # Check if we can read at least one frame
        ret, _ = cap.read()
        cap.release()
        return ret
        
    except Exception:
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Directory to save extracted frames")
    parser.add_argument("--interval", type=int, default=5, help="Extract every Nth frame")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to extract")
    
    args = parser.parse_args()
    
    try:
        # Validate video
        if not validate_video(args.video):
            logger.error(f"Invalid video file: {args.video}")
            exit(1)
            
        # Extract frames
        frames = extract_frames(
            args.video,
            args.output,
            interval=args.interval,
            max_frames=args.max_frames
        )
        
        logger.info(f"Successfully extracted {len(frames)} frames")
        
    except FrameExtractionError as e:
        logger.error(str(e))
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)
