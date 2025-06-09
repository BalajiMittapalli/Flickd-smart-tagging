import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from gradio_pipeline import process_video_pipeline

def process_all_videos(input_dir="outputs", output_dir="output1"):
    """
    Process all videos in the input directory and save results to output1 directory
    
    Args:
        input_dir (str): Directory containing video outputs
        output_dir (str): Directory to save processed results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video directories
    video_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Found {len(video_dirs)} video directories to process")
    
    for video_dir in video_dirs:
        try:
            print(f"\nProcessing video: {video_dir}")
            
            # Find result.json
            result_file = os.path.join(input_dir, video_dir, "result.json")
            
            if not os.path.exists(result_file):
                print(f"Missing result.json for {video_dir}, skipping...")
                continue
            
            # Load result data
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            # Create final output
            final_output = {
                "video_id": video_dir,
                "vibes": result_data.get("vibes", []),
                "products": []
            }
            
            # Process matches
            products_list = result_data.get("products", [])
            
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
            
            # Save the result directly in output folder with video name
            output_file = os.path.join(output_dir, f"{video_dir}.json")
            with open(output_file, 'w') as f:
                json.dump(final_output, f, indent=2)
            
            print(f"Saved result to {output_file}")
            
        except Exception as e:
            print(f"Error processing {video_dir}: {str(e)}")
            continue

if __name__ == "__main__":
    process_all_videos() 