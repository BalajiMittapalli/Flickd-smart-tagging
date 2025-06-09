# Flickd-smart-tagging

Flickd-smart-tagging is a Python-based system designed to automatically analyze short-form videos, particularly focusing on fashion content. It detects fashion items, classifies the "vibe" or aesthetic of the video, and attempts to match detected items with a product catalog. The system exposes its functionality via a FastAPI backend and includes various scripts for processing, model management, and data preparation.

## Core Features

*   **Fashion Item Detection:** Identifies various clothing items and accessories in video frames.
*   **Vibe Classification:** Analyzes video captions and hashtags to classify the video into predefined fashion vibes (e.g., Coquette, Clean Girl, Y2K).
*   **Product Matching:** Matches detected fashion items against a product catalog to find exact or similar products.
*   **FastAPI Backend:** Provides API endpoints for video analysis.
*   **Modular Scripts:** Includes standalone scripts for individual tasks like frame extraction, object detection, vibe classification, and catalog preparation.
*   **Gradio Demo:** An interactive Gradio interface to test the full pipeline.

## Models & Technologies

The system leverages several pre-trained models and libraries:

*   **Object Detection:**
    *   **Model:** `valentinafeve/yolos-fashionpedia` (YOLOS-based model fine-tuned on Fashionpedia dataset).
    *   **Library:** Hugging Face Transformers, PyTorch.
*   **Image Embedding & Product Matching:**
    *   **Model:** `openai/clip-vit-base-patch32` (CLIP for image-text similarity).
    *   **Technology:** FAISS for efficient similarity search.
    *   **Library:** Hugging Face Transformers, PyTorch, Faiss-cpu.
*   **Vibe Classification:**
    *   **Model:** `distilbert-base-uncased` (DistilBERT for text classification, fine-tuned for multi-label vibe prediction).
    *   **Technology:** Keyword matching and weighted scoring of caption/hashtags.
    *   **Library:** Hugging Face Transformers, PyTorch, Spacy (implicitly via dependencies, text processing mainly uses regex and HF tokenizer).
*   **Backend:**
    *   FastAPI, Uvicorn.
*   **Video & Image Processing:**
    *   OpenCV-Python, Pillow.
*   **Data Handling & Utilities:**
    *   NumPy, Pandas, TQDM, Matplotlib, Seaborn, Plotly.

## Directory Structure

A brief overview of key directories:

*   `api/`: Contains the FastAPI application (`main.py`) and its documentation (`README.md`).
*   `data/`: Intended for input data like product catalogs (`catalog.json`, `Catalog.xlsx`, `images.csv`) and video metadata.
    *   `data/videos/`: Example location for video metadata files (e.g., `.json.xz`).
*   `frames/`: Default directory for storing extracted video frames during API processing.
*   `models/`: Directory where downloaded model files (like YOLOS) are stored.
*   `outputs/`: Default directory for storing JSON outputs from processing (detections, matches, vibes). The Gradio pipeline and API also structure outputs here, often in subdirectories named by `video_id`.
*   `scripts/`: Contains various Python scripts for individual tasks:
    *   `check_metadata.py`: Decompresses and inspects video metadata.
    *   `classify_vibes.py`: Performs vibe classification.
    *   `detect_objects.py`: Detects fashion items from frames.
    *   `download_models.py`: Downloads the YOLOS object detection model.
    *   `extract_video_info.py`: Extracts structured info from video metadata.
    *   `frame_extractor.py`: Extracts frames from videos.
    *   `gradio_pipeline.py`: Runs a Gradio web interface for the full pipeline.
    *   `match_products.py`: Matches detected objects to a product catalog.
    *   `organize_outputs.py`: Helper script to structure output files.
    *   `prepare_catalog.py`: Prepares `catalog.json` from Excel and CSV files.
    *   `run_classify.py`: Example script to run vibe classification.
    *   `show_catalog_sample.py`: Generates a sample of the product catalog.
*   `temp/`: Temporary storage for uploaded videos during API processing.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd Flickd-smart-tagging
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Pre-trained Models:**
    The YOLOS object detection model needs to be downloaded explicitly. Other models (CLIP, DistilBERT) will be downloaded automatically by Hugging Face Transformers on first use.
    ```bash
    python scripts/download_models.py
    ```
    This will download `pytorch_model.bin` for `valentinafeve/yolos-fashionpedia` into the `models/` directory.

5.  **Prepare Product Catalog:**
    The system expects a product catalog in JSON format at `data/catalog.json`.
    *   If you have `data/Catalog.xlsx` and `data/images.csv`, you can generate `catalog.json` using:
        ```bash
        python scripts/prepare_catalog.py
        ```
    *   Otherwise, ensure `data/catalog.json` exists and is correctly formatted. A sample structure can be seen by running `python scripts/show_catalog_sample.py` (which creates `data/catalog_sample.json`).
    *   The API `/analyze` endpoint defaults to using `data/catalog.json`.

6.  **(Optional) Prepare Video Data:**
    If you plan to use the `scripts/check_metadata.py` or `scripts/extract_video_info.py` scripts, place your compressed metadata files (e.g., `*.json.xz`) in `data/videos/`.

## Usage

### 1. FastAPI Backend

The API provides endpoints for video analysis.

*   **Start the Server:**
    ```bash
    uvicorn api.main:app --reload
    ```
    The API will be accessible at `http://localhost:8000`.

*   **Key Endpoints:**
    *   `GET /health`: Checks if the API is running.
    *   `POST /analyze`: Analyzes a video file.
        *   **Request:** `multipart/form-data` with `video` (file), `caption` (text, optional), `catalog_path` (text, optional, default: "data/catalog.json").
        *   **Response:** JSON with `video_id`, `vibes`, and detected `products`.

*   **Example API Usage (Python):**
    ```python
    import requests

    url = "http://localhost:8000/analyze"
    files = {
        "video": ("my_video.mp4", open("path/to/your/video.mp4", "rb"), "video/mp4")
    }
    data = {
        "caption": "Summer fashion look with floral dress and gold earrings"
        # "catalog_path": "data/custom_catalog.json" # Optional
    }

    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    ```
    For detailed API documentation, refer to `api/README.md` or access `http://localhost:8000/docs` in your browser when the server is running.

### 2. Gradio Web Interface

A Gradio interface provides an easy way to test the video analysis pipeline.

*   **Run the Gradio App:**
    ```bash
    python scripts/gradio_pipeline.py
    ```
    Open the URL provided in your terminal (usually `http://127.0.0.1:7860`) in a web browser. You can upload a video, provide a caption and hashtags, and see the JSON results. The Gradio app will also attempt to reuse previously generated outputs if available.

### 3. Individual Scripts

The scripts in the `scripts/` directory can be run for specific tasks. Most of them support command-line arguments. Use the `--help` flag for more information, e.g.:

*   **Frame Extraction:**
    ```bash
    python scripts/frame_extractor.py --video path/to/video.mp4 --output extracted_frames_dir --interval 10
    ```
*   **Object Detection:**
    ```bash
    python scripts/detect_objects.py --frames path/to/frames_dir --output detections.json
    ```
*   **Product Matching:**
    ```bash
    python scripts/match_products.py --detections path/to/detections.json --catalog data/catalog.json --output matches.json
    ```
*   **Vibe Classification:**
    ```bash
    python scripts/classify_vibes.py --video-id my_video --caption "My cool video caption" --hashtags "#fashion #summer" --output vibes_output.json
    ```

## Workflow Overview (API `/analyze` endpoint)

1.  **Video Upload:** A video file and optional caption are sent to the `/analyze` endpoint.
2.  **Temporary Storage:** The video is saved temporarily.
3.  **Frame Extraction (`frame_extractor.py`):** Frames are extracted from the video.
4.  **Object Detection (`detect_objects.py`):** Fashion items are detected in the extracted frames using the YOLOS model. Dominant colors may also be extracted.
5.  **Product Matching (`match_products.py`):**
    *   The product catalog is loaded, and image embeddings are generated using CLIP and indexed with FAISS (cached for efficiency).
    *   Detected objects (crops from frames) are embedded using CLIP.
    *   Similarity search is performed against the catalog index to find potential matches (exact, similar, or no match based on thresholds).
6.  **Vibe Classification (`classify_vibes.py`):**
    *   The video caption and hashtags are processed using a DistilBERT model and keyword-based scoring.
    *   Top vibes are identified based on model predictions and keyword relevance.
7.  **Results Aggregation:** Detected products and classified vibes are combined into a structured JSON response.
8.  **Cleanup:** Temporary video and frame files are removed.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

(Specify your project's license here, e.g., MIT, Apache 2.0. If not specified, it's typically proprietary.)
