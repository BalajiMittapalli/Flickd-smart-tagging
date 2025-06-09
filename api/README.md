# Flickd Smart Tagging API Documentation

## Overview
The Flickd Smart Tagging API provides endpoints for automatic fashion item detection and vibe classification from short-form videos.

## Base URL
```
http://localhost:8000
```

## Endpoints

### Health Check
```http
GET /health
```
Check if the API is running properly.

**Response**
```json
{
    "status": "healthy"
}
```

### Analyze Video
```http
POST /analyze
```
Analyze a video and return structured data about detected items and vibes.

**Request**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `video`: Video file (MP4)
  - `caption`: Video caption or transcript (optional)
  - `catalog_path`: Path to product catalog CSV (default: "data/catalog.csv")

**Response**
```json
{
    "video_id": "reel_20240321_123456",
    "vibes": [
        "Coquette",
        "Clean Girl",
        "Party Glam"
    ],
    "products": [
        {
            "type": "dress",
            "color": "black",
            "matched_product_id": "prod_002",
            "match_type": "exact",
            "confidence": 0.93
        },
        {
            "type": "earrings",
            "color": "gold",
            "matched_product_id": "prod_003",
            "match_type": "similar",
            "confidence": 0.85
        }
    ]
}
```

## Model Versions
- Object Detection: YOLOv8n
- Image Embedding: CLIP ViT-B/32
- Vibe Classification: DistilBERT

## Error Handling
The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad Request
- 500: Internal Server Error

Error responses include a detail message:
```json
{
    "detail": "Error message"
}
```

## Rate Limiting
Currently, no rate limiting is implemented.

## CORS
CORS is enabled for all origins.

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download models:
```bash
python scripts/download_models.py
```

3. Start the server:
```bash
uvicorn api.main:app --reload
```

## Example Usage
```python
import requests

url = "http://localhost:8000/analyze"
files = {
    "video": ("video.mp4", open("video.mp4", "rb"), "video/mp4")
}
data = {
    "caption": "Summer fashion look with floral dress"
}

response = requests.post(url, files=files, data=data)
print(response.json())
``` 