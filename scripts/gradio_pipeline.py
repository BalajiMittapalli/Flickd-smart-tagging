import os
import sys
import gradio as gr
import shutil
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from api.main import analyze_video
from scripts.frame_extractor import extract_frames
from scripts.detect_objects import FashionDetector
from scripts.match_products import ProductMatcher
from scripts.classify_vibes import process_video

def get_video_id_from_filename(file_obj):
    """Extract video_id from uploaded file's name (without extension)."""
    filename = os.path.basename(getattr(file_obj, 'name', 'uploaded_video.mp4'))
    return os.path.splitext(filename)[0]

def get_video_output_dir(video_id: str) -> Path:
    return Path("outputs") / video_id

def check_existing_outputs(video_id: str) -> dict:
    output_dir = get_video_output_dir(video_id)
    return {
        "frames": output_dir.exists() and (output_dir / "frames").exists(),
        "detections": output_dir.exists() and (output_dir / "detections.json").exists(),
        "matches": output_dir.exists() and (output_dir / "matched_products.json").exists(),
        "vibes": output_dir.exists() and (output_dir / "vibes.json").exists()
    }

def save_outputs(video_id: str, results: dict):
    output_dir = get_video_output_dir(video_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save detections
    if "detections" in results:
        with open(output_dir / "detections.json", "w") as f:
            json.dump(results["detections"], f, indent=2)
    # Save matches
    if "matches" in results:
        with open(output_dir / "matched_products.json", "w") as f:
            json.dump(results["matches"], f, indent=2)
    # Save vibes
    if "vibes" in results:
        with open(output_dir / "vibes.json", "w") as f:
            json.dump(results["vibes"], f, indent=2)

async def process_video_pipeline(video_file, caption, hashtags=None, output_file=None):
    try:
        # Use filename (without extension) as video_id
        video_id = get_video_id_from_filename(video_file)
        print(f"Using video ID: {video_id}")
        existing_outputs = check_existing_outputs(video_id)
        print(f"Existing outputs: {existing_outputs}")
        output_dir = get_video_output_dir(video_id)

        # If all outputs exist, just load and return them
        if all(existing_outputs.values()):
            print("Using existing outputs")
            final_output = {
                "video_id": video_id,
                "vibes": [],
                "products": []
            }
            
            # Load vibes data
            vibes_file = output_dir / "vibes.json"
            if vibes_file.exists():
                try:
                    with open(vibes_file, "r") as f:
                        vibes_data = json.load(f)
                        final_output["vibes"] = vibes_data.get("vibes", [])
                except Exception as e:
                    print(f"Warning: Could not load vibes data: {str(e)}")
            
            # Load matches data
            matches_file = output_dir / "matched_products.json"
            if matches_file.exists():
                try:
                    with open(matches_file, "r") as f:
                        matches_data = json.load(f)
                        print(f"Loaded matches data type: {type(matches_data)}")
                        
                        # Handle both list and dictionary formats
                        products_list = []
                        if isinstance(matches_data, list):
                            products_list = matches_data
                        elif isinstance(matches_data, dict) and "products" in matches_data:
                            products_list = matches_data["products"]
                        
                        print(f"Found {len(products_list)} products in matches")
                        for match in products_list:
                            if isinstance(match, dict):
                                confidence = match.get("confidence", 0.0)
                                # Only include products with confidence > 0.75
                                if confidence > 0.75:
                                    match_label = match.get("match_label", "").lower()
                                    # Skip products with "No Match"
                                    if "no match" in match_label:
                                        continue
                                    product = {
                                        "type": match.get("type", ""),
                                        "color": match.get("color", ""),
                                        "match_type": match_label.replace(" match", ""),
                                        "matched_product_id": match.get("matched_product_id", ""),
                                        "confidence": confidence
                                    }
                                    final_output["products"].append(product)
                        print(f"Added {len(final_output['products'])} products to final output")
                except Exception as e:
                    print(f"Warning: Could not load matches data: {str(e)}")
                    print(f"Matches file content: {matches_file.read_text() if matches_file.exists() else 'File not found'}")
            
            return final_output

        # If we get here, we need to process the video
        results = {}
        # Save the uploaded video to a temp file for processing if needed
        temp_video_path = os.path.join("temp", f"{video_id}.mp4")
        os.makedirs("temp", exist_ok=True)
        
        # Handle Gradio file input
        if hasattr(video_file, 'name'):
            # If it's a Gradio file input
            print(f"Copying video from {video_file.name} to {temp_video_path}")
            shutil.copy(video_file.name, temp_video_path)
        else:
            # If it's a regular file object
            print(f"Writing video to {temp_video_path}")
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())

        # Verify the file exists and has content
        if not os.path.exists(temp_video_path):
            raise Exception(f"Failed to save video file to {temp_video_path}")
        if os.path.getsize(temp_video_path) == 0:
            raise Exception(f"Video file is empty: {temp_video_path}")

        # Extract frames first
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting frames to {frames_dir}")
        frame_paths = extract_frames(temp_video_path, str(frames_dir))
        if not frame_paths:
            raise Exception("Failed to extract frames from video")

        # Only run missing steps
        # 1. Detections
        if not existing_outputs["detections"]:
            print("Running object detection...")
            detector = FashionDetector()
            detections = detector.process_frames(
                str(frames_dir),  # Use frames directory
                output_dir / "detections.json"
            )
            results["detections"] = detections
        # 2. Matches - Force this to run
        print("Matching products...")
        matcher = ProductMatcher()
        # Load catalog before matching
        catalog_path = "data/catalog.json"
        if not os.path.exists(catalog_path):
            raise Exception(f"Catalog file not found at {catalog_path}")
        matcher.load_catalog(catalog_path)
        # Force a fresh run by deleting the existing matches file
        matches_file = output_dir / "matched_products.json"
        if matches_file.exists():
            matches_file.unlink()
        matches = matcher.match_products(
            output_dir / "detections.json",
            output_dir / "matched_products.json"
        )
        print(f"Matches result type: {type(matches)}")
        results["matches"] = matches
        # 3. Vibes
        print("Classifying vibes...")
        vibes = process_video(
            video_id=video_id,
            caption=caption,
            hashtags=hashtags or [],
            output_file=str(output_dir / "vibes.json")
        )
        results["vibes"] = vibes
        # Save any new outputs
        save_outputs(video_id, results)
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        # Compose final output
        final_output = {
            "video_id": video_id,
            "vibes": [],
            "products": []
        }
        
        # Load vibes data if available
        vibes_file = output_dir / "vibes.json"
        if vibes_file.exists():
            try:
                with open(vibes_file, "r") as f:
                    vibes_data = json.load(f)
                    final_output["vibes"] = vibes_data.get("vibes", [])
            except Exception as e:
                print(f"Warning: Could not load vibes data: {str(e)}")
        
        # Load matches data if available
        matches_file = output_dir / "matched_products.json"
        if matches_file.exists():
            try:
                with open(matches_file, "r") as f:
                    matches_data = json.load(f)
                    print(f"Loaded matches data type: {type(matches_data)}")
                    
                    # Handle both list and dictionary formats
                    products_list = []
                    if isinstance(matches_data, list):
                        products_list = matches_data
                    elif isinstance(matches_data, dict) and "products" in matches_data:
                        products_list = matches_data["products"]
                    
                    print(f"Found {len(products_list)} products in matches")
                    for match in products_list:
                        if isinstance(match, dict):
                            confidence = match.get("confidence", 0.0)
                            # Only include products with confidence > 0.75
                            if confidence > 0.75:
                                match_label = match.get("match_label", "").lower()
                                # Skip products with "No Match"
                                if "no match" in match_label:
                                    continue
                                product = {
                                    "type": match.get("type", ""),
                                    "color": match.get("color", ""),
                                    "match_type": match_label.replace(" match", ""),
                                    "matched_product_id": match.get("matched_product_id", ""),
                                    "confidence": confidence
                                }
                                final_output["products"].append(product)
                    print(f"Added {len(final_output['products'])} products to final output")
            except Exception as e:
                print(f"Warning: Could not load matches data: {str(e)}")
                print(f"Matches file content: {matches_file.read_text() if matches_file.exists() else 'File not found'}")
        
        return final_output
    except Exception as e:
        print(f"Error in process_video_pipeline: {str(e)}")
        # Clean up temp file in case of error
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return {"error": str(e)}

demo = gr.Interface(
    fn=process_video_pipeline,
    inputs=[
        gr.File(label="Upload Video"),
        gr.Textbox(label="Caption", placeholder="Enter video caption"),
        gr.Textbox(label="Hashtags (optional)", placeholder="Enter hashtags separated by spaces")
    ],
    outputs=gr.JSON(label="Results"),
    title="Video Analysis Pipeline",
    description="Upload a video to analyze its content and detect products. The pipeline will reuse existing outputs if available."
)

if __name__ == "__main__":
    demo.launch()