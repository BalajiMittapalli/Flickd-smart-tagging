"""
Object detection script using DETR model with ResNet-50 backbone for fashion items.
Detects clothing items and accessories in video frames.
"""

import os
import json
import logging
from tqdm import tqdm
from colorthief import ColorThief
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fashionpedia classes from the model card
FASHION_CLASSES = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket',
    'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape',
    'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove',
    'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe',
    'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette',
    'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead',
    'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel'
]

class FashionDetector:
    def __init__(self):
        logger.info("Loading valentinafeve/yolos-fashionpedia model from Hugging Face...")
        self.processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
        self.model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded successfully. Using device: {self.device}")

    def extract_dominant_colors(self, image_path, num_colors=3):
        try:
            color_thief = ColorThief(image_path)
            palette = color_thief.get_palette(color_count=num_colors)
            return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette]
        except Exception as e:
            logger.warning(f"Failed to extract colors from {image_path}: {str(e)}")
            return []

    def detect_frame(self, frame_path):
        try:
            image = Image.open(frame_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.1
            )[0]
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.1:
                    class_name = FASHION_CLASSES[label]
                    detections.append({
                        "bbox": [float(x) for x in box.tolist()],
                        "confidence": float(score),
                        "class": class_name
                    })
            return detections
        except Exception as e:
            logger.error(f"Error processing frame {frame_path}: {str(e)}")
            return []

    def process_frames(self, frames_path, output_path):
        logger.info(f"Processing frames from {frames_path}")
        all_detections = []
        if os.path.isdir(frames_path):
            frame_files = [f for f in os.listdir(frames_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for frame_file in tqdm(frame_files, desc="Processing frames"):
                frame_path = os.path.join(frames_path, frame_file)
                frame_detections = self.detect_frame(frame_path)
                if frame_detections:
                    frame_data = {
                        "frame": frame_file,
                        "detections": frame_detections,
                        "colors": self.extract_dominant_colors(frame_path)
                    }
                    all_detections.append(frame_data)
        else:
            frame_detections = self.detect_frame(frames_path)
            if frame_detections:
                frame_data = {
                    "frame": os.path.basename(frames_path),
                    "detections": frame_detections,
                    "colors": self.extract_dominant_colors(frames_path)
                }
                all_detections.append(frame_data)
        output = {
            "total_frames": len(all_detections),
            "total_detections": sum(len(d["detections"]) for d in all_detections),
            "model": "valentinafeve/yolos-fashionpedia",
            "classes": FASHION_CLASSES,
            "frames": all_detections
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved {len(all_detections)} frames with detections to {output_path}")

def process_video_frames(frames_dir: str, output_path: str) -> None:
    """
    Process all frames in a directory and save detections to a JSON file.
    
    Args:
        frames_dir: Directory containing video frames
        output_path: Path to save the detections JSON file
    """
    detector = FashionDetector()
    detector.process_frames(frames_dir, output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Detect fashion items in video frames')
    parser.add_argument('--frames', required=True, help='Directory containing video frames or a single image file')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    process_video_frames(args.frames, args.output)

if __name__ == "__main__":
    main() 