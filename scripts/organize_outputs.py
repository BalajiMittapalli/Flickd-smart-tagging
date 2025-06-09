import os
import shutil
from pathlib import Path
import json

def organize_existing_outputs(video_id: str):
    """Organize existing outputs into a structured format."""
    # Create video-specific output directory
    output_dir = Path("outputs") / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create frames directory
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Move existing files to the new structure
    # 1. Move detections
    detections_src = Path("outputs") / f"detections_{video_id}.json"
    if detections_src.exists():
        shutil.copy2(detections_src, output_dir / "detections.json")
    
    # 2. Move matched products
    matches_src = Path("outputs") / f"matched_products_{video_id}.json"
    if matches_src.exists():
        shutil.copy2(matches_src, output_dir / "matched_products.json")
    
    # 3. Move vibes
    vibes_src = Path("outputs") / f"vibes_{video_id}.json"
    if vibes_src.exists():
        shutil.copy2(vibes_src, output_dir / "vibes.json")
    
    # 4. Move frames
    frames_src = Path("frames")
    if frames_src.exists():
        for frame in frames_src.glob("*.jpg"):
            shutil.copy2(frame, frames_dir / frame.name)

if __name__ == "__main__":
    # Organize outputs for the video
    video_id = "2025-06-02_11-31-19_UTC"
    organize_existing_outputs(video_id)
    print(f"Organized outputs for video {video_id}") 