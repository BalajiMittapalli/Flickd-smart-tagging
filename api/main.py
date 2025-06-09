"""
FastAPI application for the Flickd Smart Tagging & Vibe Classification Engine.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import shutil
from datetime import datetime

from scripts.frame_extractor import extract_frames
from scripts.detect_objects import process_video_frames
from scripts.match_products import process_detections
from scripts.classify_vibes import process_video

app = FastAPI(
    title="Flickd Smart Tagging API",
    description="API for automatic fashion item detection and vibe classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
for dir_name in ["frames", "outputs", "temp"]:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_video(
    video: Union[UploadFile, Any] = File(...),
    caption: str = "",
    catalog_path: str = "data/catalog.json"
) -> Dict[str, Any]:
    """
    Analyze a video and return structured data about detected items and vibes.
    
    Args:
        video: Uploaded video file (FastAPI UploadFile or file-like object)
        caption: Video caption or transcript
        catalog_path: Path to product catalog JSON
        
    Returns:
        Structured data with detected items and vibes
    """
    try:
        # Generate unique ID for this analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_id = f"reel_{timestamp}"
        
        # Save uploaded video
        video_path = f"temp/{video_id}.mp4"
        with open(video_path, "wb") as buffer:
            if isinstance(video, UploadFile):
                shutil.copyfileobj(video.file, buffer)
            else:
                # Handle regular file object
                buffer.write(video.read())
        
        # Extract frames
        frames_dir = f"frames/{video_id}"
        frame_paths = extract_frames(video_path, frames_dir)
        
        # Detect objects
        detections_file = f"outputs/{video_id}_detections.json"
        process_video_frames(frames_dir, detections_file)
        
        # Match products
        matches_file = f"outputs/{video_id}_matches.json"
        process_detections(detections_file, catalog_path, matches_file)
        
        # Classify vibes
        vibes_file = f"outputs/{video_id}_vibes.json"
        process_video(video_id, caption, vibes_file)
        
        # Combine results
        with open(matches_file, 'r') as f:
            matches_data = json.load(f)
        with open(vibes_file, 'r') as f:
            vibes_data = json.load(f)
        
        result = {
            "video_id": video_id,
            "vibes": vibes_data["vibes"],
            "products": matches_data["products"]
        }
        
        # Clean up temporary files
        os.remove(video_path)
        shutil.rmtree(frames_dir)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 