# Flickd-smart-tagging

## Overview
This project is the "Smart Tagging & Vibe Classification Engine" for Flickd, designed to automate the tagging of products and classification of fashion "vibes" from short-form videos. The engine extracts frames, detects fashion items using an object detection model, matches these items against a product catalog using image embeddings, and analyzes video captions/hashtags to classify vibes using NLP. The results are exposed via a FastAPI backend.

This project was developed as part of the Flickd AI Hackathon.

## Features
- **Video Frame Extraction**: Extracts frames from input videos at specified intervals using OpenCV.
- **Fashion Item Detection**: Utilizes the `valentinafeve/yolos-fashionpedia` model (a YOLOS-based object detection model from Hugging Face Transformers) to identify fashion items (tops, bottoms, dresses, accessories, etc.) in video frames.
    - Outputs class name, bounding box, and confidence score for each detected item.
- **Dominant Color Extraction**: Identifies dominant colors (as hex codes) from video frames where items are detected, using the `colorthief` library.
- **Product Matching**:
    - Crops detected objects from frames.
    - Generates CLIP (OpenAI ViT-B/32) image embeddings for cropped items.
    - Matches embeddings against a pre-built FAISS index of a product catalog using cosine similarity.
    - Classifies matches as "Exact Match", "Similar Match", or "No Match" based on similarity scores (thresholds are configurable).
- **Vibe Classification**:
    - Analyzes video captions and hashtags using an NLP model (DistilBERT from Hugging Face Transformers).
    - Classifies videos into 1-3 predefined fashion vibes (e.g., Coquette, Clean Girl, Y2K, Boho, Party Glam, etc.).
    - Incorporates weighted scoring for captions, hashtags, and keyword matching for improved accuracy.
- **API**:
    - FastAPI backend to expose the full analysis pipeline.
    - `POST /analyze` endpoint to process an uploaded video and optional caption, returning a structured JSON output.
    - `GET /health` endpoint for health checks.
    - CORS enabled for all origins.
- **Gradio Demo**: An interactive web interface (`scripts/gradio_pipeline.py`) to test the video analysis pipeline locally.
- **Data Preparation**: Includes scripts to prepare the product catalog from Excel/CSV inputs into a JSON format suitable for the matching module.

## Tech Stack
- **Programming Language**: Python 3.x
- **Object Detection**: `valentinafeve/yolos-fashionpedia` (via Hugging Face `transformers`)
- **Image Embedding & Matching**: OpenAI CLIP (`clip-vit-base-patch32` via Hugging Face `transformers`), FAISS (`faiss-cpu`)
- **NLP / Vibe Classification**: DistilBERT (`distilbert-base-uncased` via Hugging Face `transformers`)
- **Video Processing**: OpenCV (`opencv-python`)
- **API Framework**: FastAPI, Uvicorn
- **Data Handling**: Pandas, NumPy
- **Image Processing**: Pillow, Colorthief
- **Core Libraries**: `torch`, `torchvision`, `python-multipart`, `tqdm`, `spacy`

## Directory Structure
Use code with caution.
Markdown
.
├── api/ # FastAPI application
│ ├── main.py # FastAPI main application logic
│ └── README.md # API specific documentation
├── data/ # Data files (input catalogs, generated indexes, sample video metadata)
│ ├── Catalog.xlsx # (Input) Product catalog in Excel format
│ ├── images.csv # (Input) Product images mapping in CSV format
│ ├── catalog.json # (Generated) Combined product catalog in JSON format
│ ├── catalog_sample.json # (Generated) A sample of the catalog.json for quick inspection
│ ├── faiss_index.bin # (Generated) FAISS index for product matching
│ ├── catalog_hash.txt # (Generated) Hash of catalog.json to track changes for index rebuilding
│ └── videos/ # (Input) Optional directory for sample video metadata files
├── frames/ # Directory for storing extracted video frames (created dynamically)
├── models/ # Directory for storing downloaded pre-trained models (via script)
├── outputs/ # Directory for storing analysis results (JSON files, created dynamically)
├── scripts/ # Utility and processing scripts
│ ├── check_metadata.py # Script to check and decompress video metadata
│ ├── classify_vibes.py # Core logic for vibe classification
│ ├── detect_objects.py # Core logic for fashion object detection
│ ├── download_models.py # Script to download some pre-trained models
│ ├── extract_video_info.py # Script to extract info from video metadata
│ ├── frame_extractor.py # Core logic for extracting frames from videos
│ ├── gradio_pipeline.py # Gradio demo application
│ ├── match_products.py # Core logic for matching detected products to catalog
│ ├── organize_outputs.py # Helper script to organize output files
│ ├── prepare_catalog.py # Script to convert Excel/CSV catalog to JSON
│ ├── run_classify.py # Example script to run vibe classification
│ └── show_catalog_sample.py # Script to generate catalog_sample.json
├── temp/ # Temporary file storage (created dynamically)
├── README.md # This file
└── requirements.txt # Python dependencies
## Setup and Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Flickd-smart-tagging
Use code with caution.
2. Create and Activate a Virtual Environment (Recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Use code with caution.
Bash
3. Install Dependencies
pip install -r requirements.txt
Use code with caution.
Bash
4. Create Necessary Directories
The API and various scripts will attempt to create these if they don't exist. However, you can create them upfront:
mkdir -p models data frames outputs temp
mkdir -p data/videos # If you plan to use scripts working with video metadata
Use code with caution.
Bash
5. Download Pre-trained Models
Run the script to download some of the pre-trained model files (specifically for valentinafeve/yolos-fashionpedia). These will be saved in the models/ directory.
python scripts/download_models.py
Use code with caution.
Bash
Note: This script downloads specific model files. Other models (like DistilBERT for vibe classification and CLIP) will be downloaded automatically by the Hugging Face transformers library to its cache (usually ~/.cache/huggingface/hub) on their first use if not found or if the local setup from models/ is incomplete for them. The download process might take some time and requires a stable internet connection.
6. Prepare the Product Catalog
a. Place your product catalog source files (Catalog.xlsx and images.csv, as provided for the hackathon) into the data/ directory.
data/Catalog.xlsx
data/images.csv
b. Run the script to process these files and generate data/catalog.json, which is used by the product matching module:
python scripts/prepare_catalog.py
Use code with caution.
Bash
This script will also create data/catalog_sample.json for a quick look at the catalog structure.
The FAISS index (data/faiss_index.bin) and catalog hash (data/catalog_hash.txt) will be automatically generated in the data/ directory by the product matching module (e.g., when the /analyze API endpoint is called for the first time, or scripts/match_products.py is run) if they don't exist or if data/catalog.json has changed.
7. (Optional) Prepare Video Metadata
If you have video metadata files (e.g., .json.xz files from a dataset) and want to use scripts like check_metadata.py or extract_video_info.py:
a. Place these metadata files into the data/videos/ directory.
b. Run the relevant scripts, for example:
python scripts/check_metadata.py --dir data/videos
python scripts/extract_video_info.py --dir data/videos
Use code with caution.
Bash
8. Running the Application
a. Start the FastAPI Server
To run the main analysis API:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Use code with caution.
Bash
The API will be accessible at http://localhost:8000. Refer to api/README.md or the "API Endpoints" section below for details.
b. Run the Gradio Demo
For an interactive way to test the full pipeline with a video upload:
python scripts/gradio_pipeline.py
Use code with caution.
Bash
This will launch a Gradio interface, typically accessible in your browser at http://127.0.0.1:7860.
c. Run Individual Scripts (for testing/debugging specific modules)
You can execute individual scripts for more granular testing. Ensure that any required input files (e.g., video frames, detection JSONs, catalog) are available in the expected locations.
Frame Extraction:
# python scripts/frame_extractor.py --video path/to/your/video.mp4 --output frames/your_video_id
Use code with caution.
Bash
Object Detection (requires frames):
# python scripts/detect_objects.py --frames frames/your_video_id/ --output outputs/your_video_id_detections.json
Use code with caution.
Bash
Product Matching (requires detections JSON and catalog.json):
# python scripts/match_products.py --detections outputs/your_video_id_detections.json --catalog data/catalog.json --output outputs/your_video_id_matches.json
Use code with caution.
Bash
Vibe Classification (example from run_classify.py):
python scripts/run_classify.py
# Or directly:
# python scripts/classify_vibes.py --video-id "test_video" --caption "Amazing outfit for a sunny day!" --hashtags "#fashion #summer" --output "outputs/test_video_vibes.json"
Use code with caution.
Bash
Managing Large Files with Git
Directories like models/ (partially), frames/, outputs/, temp/, and some generated files in data/ (like faiss_index.bin, catalog_hash.txt), along with raw video files, can be very large. These are generally not suitable for version control with Git.
It is highly recommended to add them to a .gitignore file. Here’s a sample .gitignore:
# .gitignore

# Python virtual environment and cache
venv/
__pycache__/
*.pyc
.env

# Downloaded Models (can be large, script handles download)
models/*
!models/.gitkeep # Use .gitkeep if you want to commit the empty directory

# Data (large generated files or input data not for VCS if too big)
data/faiss_index.bin
data/catalog_hash.txt
data/videos/          # If storing large video/metadata files here
# data/catalog.json     # Consider ignoring if very large and easily reproducible
# data/catalog_sample.json # Or keep if small

# Frames (dynamically generated during processing)
frames/
frames/*/

# Outputs (dynamically generated analysis results)
outputs/
outputs/*/
!outputs/.gitkeep # Use .gitkeep if you want to commit the empty directory

# Temporary files
temp/

# IDE specific files
.idea/
.vscode/

# Media files (if you are adding samples directly)
*.mp4
*.json.xz
Use code with caution.
Ensure essential small input files like data/Catalog.xlsx and data/images.csv are committed if they are part of the baseline project and not excessively large. If they are large, provide them separately.
scripts/download_models.py handles the acquisition of some model files.
frames/, outputs/, and temp/ are primarily for runtime-generated data.
API Endpoints
Base URL: http://localhost:8000
Health Check
Endpoint: GET /health
Description: Checks if the API is running and healthy.
Response:
{
    "status": "healthy"
}
Use code with caution.
Json
Analyze Video
Endpoint: POST /analyze
Description: Analyzes an uploaded video to detect fashion items, match them against the product catalog, and classify fashion vibes based on the video and optional caption.
Request: multipart/form-data
video: Video file (e.g., MP4 format). (Required)
caption: Video caption or transcript. (Optional, string, default: "")
catalog_path: Path to the product catalog JSON file. (Optional, string, default: "data/catalog.json")
Response: (See "Output Format" section below for structure)
Output Format
The /analyze endpoint (and the overall pipeline) aims to produce a JSON object structured as follows for each video:
{
    "video_id": "reel_20240321_123456", // Unique ID generated for this analysis instance
    "vibes": [
        "Coquette",
        "Clean Girl",
        "Party Glam"
        // List of 1-3 classified vibes
    ],
    "products": [
        {
            "type": "dress",            // Detected item class/type (e.g., from FASHION_CLASSES)
            "color": "black",           // Dominant color of the item (e.g., "black", "gold" as per api/README.md example; actual output might be hex code from colorthief)
            "matched_product_id": "prod_002", // ID of the matched product from your catalog
            "match_type": "exact",      // Match category: "exact", "similar". (Note: internal scripts might use "Exact Match", "Similar Match")
            "confidence": 0.93          // Similarity score (0.0 to 1.0) for the product match
        },
        {
            "type": "earrings",
            "color": "gold",
            "matched_product_id": "prod_003",
            "match_type": "similar",
            "confidence": 0.85
        }
        // ... more products
    ]
}
Use code with caution.
Json
Note: The color field in the example output (e.g., "black") is illustrative as per api/README.md. The actual match_products.py script assigns a hex color code from colorthief (e.g., "#000000") or "unknown". The match_type field is also illustrative; the current api/main.py directly uses product data from match_products.py which contains a match_label field (e.g., "Exact Match", "Similar Match"). For the API to strictly conform to the example match_type and color names, a transformation step in api/main.py would be needed.
Project Structure Rationale (as per Hackathon Guidelines)
/api: Contains the FastAPI backend, making a clear separation for the serving layer.
/data: Centralizes all input datasets (like catalogs) and large generated data files (like FAISS indexes), keeping them distinct from code.
/frames: Logically stores transient data (video frames) generated during processing.
/models: Intended for storing downloaded pre-trained model files, promoting organized model management.
/outputs: Collects all final JSON results from video analyses, useful for evaluation and review.
/scripts: Houses all reusable Python scripts for various operations – from data preparation and model downloads to running individual pipeline stages or demos. This modularizes functionality.
README.md: This comprehensive document serves as the primary guide for understanding, setting up, and using the project.
requirements.txt: Ensures a reproducible Python environment by listing all dependencies.
This structure aims for clarity, modularity, and adherence to common practices in ML project organization, making it easier to navigate, maintain, and extend the codebase.
