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

async def process_video(video_file, caption, hashtags=None, output_file=None):
    try:
        # Use filename (without extension) as video_id
        video_id = get_video_id_from_filename(video_file)
        print(f"Using video ID: {video_id}")
        existing_outputs = check_existing_outputs(video_id)
        print(f"Existing outputs: {existing_outputs}")
        output_dir = get_video_output_dir(video_id)
        results = {}

        # If all outputs exist, just load and return
        if all(existing_outputs.values()):
            print("Using existing outputs")
            with open(output_dir / "vibes.json", "r") as f:
                vibes_data = json.load(f)
            with open(output_dir / "matched_products.json", "r") as f:
                matches_data = json.load(f)
            final_output = {
                "video_id": video_id,
                "vibes": vibes_data.get("vibes", []),
                "products": []
            }
            for match in matches_data.get("products", []):
                match_type = match.get("match_label", "").lower().replace(" match", "")
                if match_type == "similar":
                    product = {
                        "type": match.get("type", ""),
                        "color": match.get("color", ""),
                        "match_type": match_type,
                        "matched_product_id": match.get("matched_product_id", ""),
                        "confidence": match.get("confidence", 0.0)
                    }
                    final_output["products"].append(product)
            return final_output

        # Save the uploaded video to a temp file for processing if needed
        temp_video_path = os.path.join("temp", f"{video_id}.mp4")
        os.makedirs("temp", exist_ok=True)
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        # Only run missing steps
        # 1. Frames (optional, not used in output, but could be added)
        # 2. Detections
        if not existing_outputs["detections"]:
            detector = FashionDetector()
            detections = detector.process_frames(
                temp_video_path,  # or frames dir if needed
                output_dir / "detections.json"
            )
            results["detections"] = detections
        # 3. Matches
        if not existing_outputs["matches"]:
            matcher = ProductMatcher()
            matches = matcher.match_products(
                output_dir / "detections.json",
                output_dir / "matched_products.json"
            )
            results["matches"] = matches
        # 4. Vibes
        if not existing_outputs["vibes"]:
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
        os.remove(temp_video_path)
        # Compose final output
        # Always load the latest outputs for consistency
        with open(output_dir / "vibes.json", "r") as f:
            vibes_data = json.load(f)
        with open(output_dir / "matched_products.json", "r") as f:
            matches_data = json.load(f)
        final_output = {
            "video_id": video_id,
            "vibes": vibes_data.get("vibes", []),
            "products": []
        }
        for match in matches_data.get("products", []):
            match_type = match.get("match_label", "").lower().replace(" match", "")
            if match_type == "similar":
                product = {
                    "type": match.get("type", ""),
                    "color": match.get("color", ""),
                    "match_type": match_type,
                    "matched_product_id": match.get("matched_product_id", ""),
                    "confidence": match.get("confidence", 0.0)
                }
                final_output["products"].append(product)
        return final_output
    except Exception as e:
        return {"error": str(e)}

demo = gr.Interface(
    fn=process_video,
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