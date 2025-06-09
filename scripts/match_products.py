"""
Product matching script using CLIP and FAISS for fashion items.
Matches detected items against a product catalog, labeling:
  ● Exact Match (similarity >= 0.90)
  ● Similar Match (0.75 ≤ similarity < 0.90)
  ● No Match (similarity < 0.75)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from PIL import Image
import requests
from io import BytesIO
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set up logging
type_strings = {
    'exact': "Exact Match",
    'similar': "Similar Match",
    'none': "No Match"
}

SIMILARITY_THRESHOLDS = {
    'exact': 0.80,
    'similar': 0.65
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductMatchingError(Exception):
    """Custom exception for product matching errors."""
    pass

class ProductMatcher:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        try:
            logger.info(f"Loading CLIP model: {model_name}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.index = None
            self.catalog_items = []
            self.catalog_embeddings = None
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ProductMatchingError(f"Model loading failed: {e}")

    def _compute_catalog_hash(self, catalog: Dict) -> str:
        import hashlib
        catalog_str = json.dumps(catalog, sort_keys=True)
        return hashlib.md5(catalog_str.encode()).hexdigest()

    def _save_index(self, index_path: str, hash_path: str, catalog_hash: str):
        try:
            faiss.write_index(self.index, index_path)
            with open(hash_path, 'w', encoding='utf-8') as f:
                f.write(catalog_hash)
            logger.info(f"Saved index to {index_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise ProductMatchingError(f"Index saving failed: {e}")

    def _load_index(self, index_path: str, hash_path: str, catalog_hash: str) -> bool:
        try:
            if not os.path.exists(index_path) or not os.path.exists(hash_path):
                return False
            with open(hash_path, 'r', encoding='utf-8') as f:
                saved_hash = f.read().strip()
            if saved_hash != catalog_hash:
                logger.info("Catalog has changed, rebuilding index")
                return False
            self.index = faiss.read_index(index_path)
            with open(self.catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
            # extract catalog_items same as in load_catalog
            items = []
            if isinstance(catalog, dict):
                if 'first_product' in catalog:
                    items = [catalog['first_product']]
                elif 'items' in catalog:
                    items = catalog['items']
                elif 'products' in catalog:
                    items = catalog['products']
                elif 'catalog' in catalog:
                    items = catalog['catalog']
                else:
                    items = [v for v in catalog.values() if isinstance(v, dict) and 'id' in v and 'images' in v]
            if not items:
                logger.warning("No valid catalog items found in catalog file")
                return False
            self.catalog_items = items
            logger.info(f"Loaded {len(items)} catalog items from catalog file")
            logger.info(f"Loaded index from {index_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def load_catalog(self, catalog_path: str) -> None:
        try:
            logger.info(f"Loading catalog from {catalog_path}")
            self.catalog_path = catalog_path
            with open(catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
            catalog_hash = self._compute_catalog_hash(catalog)
            index_path = "data/faiss_index.bin"
            hash_path = "data/catalog_hash.txt"
            if self._load_index(index_path, hash_path, catalog_hash):
                logger.info("Using cached index")
                logger.info(f"Indexed {len(self.catalog_items)} items; FAISS size: {self.index.ntotal}")
                return
            # build index from scratch
            logger.info("Processing catalog items...")
            items = []
            if isinstance(catalog, dict):
                if 'first_product' in catalog:
                    items = [catalog['first_product']]
                elif 'items' in catalog:
                    items = catalog['items']
                elif 'products' in catalog:
                    items = catalog['products']
                elif 'catalog' in catalog:
                    items = catalog['catalog']
                else:
                    items = [v for v in catalog.values() if isinstance(v, dict) and 'id' in v and 'images' in v]
            if not items:
                raise ProductMatchingError("No valid catalog items found")
            embeddings = []
            for item in tqdm(items, desc="Catalog→features"):
                image_url = item.get('images', [None])[0]
                if not image_url:
                    continue
                if not image_url.startswith(('http://', 'https://')):
                    image_url = 'https://' + image_url
                try:
                    resp = requests.get(image_url, timeout=10)
                    resp.raise_for_status()
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                except Exception:
                    continue
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=1, keepdim=True)
                self.catalog_items.append(item)
                embeddings.append(feats.cpu().numpy())
            if not self.catalog_items:
                raise ProductMatchingError("No valid catalog items found")
            self.catalog_embeddings = np.vstack(embeddings)
            dim = self.catalog_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.catalog_embeddings)
            self._save_index(index_path, hash_path, catalog_hash)
            logger.info(f"Indexed {len(self.catalog_items)} items; FAISS size: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            raise ProductMatchingError(f"Catalog loading failed: {e}")

    def match_products(self, detections_path: str, output_path: str) -> None:
        try:
            logger.info(f"Loading detections from {detections_path}")
            with open(detections_path, 'r') as f:
                detections = json.load(f)
            if 'frames' not in detections:
                raise ProductMatchingError("Invalid detections format")
            if self.index is None or self.index.ntotal == 0:
                raise ProductMatchingError("Empty FAISS index—did you load the catalog?")
            
            # Dictionary to store best matches for each product
            best_matches = {}
            
            for frame in tqdm(detections['frames'], desc="Matching→results"):
                frame_file = os.path.join('frames', frame['frame'])
                if not os.path.exists(frame_file):
                    # Search all subdirectories of 'frames/' for the frame
                    found = False
                    for root, dirs, files in os.walk('frames'):
                        if frame['frame'] in files:
                            frame_file = os.path.join(root, frame['frame'])
                            found = True
                            break
                    if not found:
                        logger.warning(f"Frame image not found: {frame['frame']}")
                        continue
                img = Image.open(frame_file).convert('RGB')
                w, h = img.size
                for det in frame['detections']:
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = img.crop((x1, y1, x2, y2))
                    if crop.size[0] == 0 or crop.size[1] == 0:
                        continue
                    inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        feat = self.model.get_image_features(**inputs)
                    feat = feat / feat.norm(dim=1, keepdim=True)
                    sims, idxs = self.index.search(feat.cpu().numpy(), k=1)
                    score = float(sims[0][0])
                    if score >= SIMILARITY_THRESHOLDS['exact']:
                        key = 'exact'
                    elif score >= SIMILARITY_THRESHOLDS['similar']:
                        key = 'similar'
                    else:
                        key = 'none'
                    
                    product_id = self.catalog_items[idxs[0][0]]['id']
                    match = {
                        'type': det['class'],
                        'color': (frame.get('colors') or ['unknown'])[0],
                        'match_label': type_strings[key],
                        'matched_product_id': product_id,
                        'confidence': score
                    }
                    
                    # Update best match if this is better than existing one
                    if product_id not in best_matches or score > best_matches[product_id]['confidence']:
                        best_matches[product_id] = match
            
            # Convert dictionary to list of matches
            matches = list(best_matches.values())
            
            out = {
                'video_id': Path(detections_path).stem,
                'products': matches,
                'vibes': []
            }
            with open(output_path, 'w') as f:
                json.dump(out, f, indent=2)
            logger.info(f"Saved {len(matches)} unique matches to {output_path}")
        except Exception as e:
            logger.error(f"Failed to match products: {e}")
            raise ProductMatchingError(f"Product matching failed: {e}")

def process_detections(detections_path: str, catalog_path: str, output_path: str) -> None:
    """
    Process detections and match them against the catalog.
    
    Args:
        detections_path: Path to the detections JSON file
        catalog_path: Path to the catalog JSON file
        output_path: Path to save the matched products JSON file
    """
    matcher = ProductMatcher()
    matcher.load_catalog(catalog_path)
    matcher.match_products(detections_path, output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Match detected fashion items against a product catalog')
    parser.add_argument('--detections', required=True, help='Path to detections JSON file')
    parser.add_argument('--catalog', required=True, help='Path to catalog JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    process_detections(args.detections, args.catalog, args.output)

if __name__ == "__main__":
    main()
