"""
Vibe classification script using NLP to analyze video captions and hashtags.
Classifies videos into fashion vibes using transformer models with multi-label support.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
import numpy as np
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available vibes
VIBES = [
    'Coquette',
    'Clean Girl',
    'Cottagecore',
    'Streetcore',
    'Y2K',
    'Boho',
    'Party Glam'
]

# Define vibe categories with their associated keywords
VIBE_KEYWORDS = {
    "Boho": ["traditional", "ethnic", "cultural", "artistic", "handcrafted", "natural", "organic", "silver", "jewelry"],
    "Clean Girl": ["professional", "sophisticated", "elegant", "minimal", "high-end", "luxury", "vogue", "model"],
    "Party Glam": ["glamorous", "statement", "bold", "luxury", "high-fashion", "vogue", "model", "professional"],
    "Coquette": ["romantic", "feminine", "delicate", "vintage", "soft"],
    "Cottagecore": ["rural", "pastoral", "natural", "vintage", "rustic"],
    "Streetcore": ["urban", "street", "casual", "modern", "trendy"],
    "Y2K": ["retro", "2000s", "nostalgic", "trendy", "vintage"]
}

class VibeClassificationError(Exception):
    """Custom exception for vibe classification errors."""
    pass

class VibeClassifier:
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        threshold: float = 0.4,  # Lowered threshold
        top_k: int = 3,
        max_length: int = 512,
        caption_weight: float = 0.8,
        hashtag_weight: float = 0.2,
        keyword_weight: float = 0.4,  # Increased keyword weight
        device: Optional[str] = None
    ):
        """
        Initialize the vibe classifier with optimized parameters
        
        Args:
            model_name: Name of the transformer model to use
            threshold: Confidence threshold for multi-label classification
            top_k: Maximum number of vibes to return
            max_length: Maximum sequence length for tokenization
            caption_weight: Weight for caption
            hashtag_weight: Weight for hashtags
            keyword_weight: Weight for keyword matching
            device: Device to use for computation (auto-detected if None)
            
        Raises:
            VibeClassificationError: If model loading fails
        """
        try:
            logger.info(f"Loading model: {model_name}")
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(VIBES),
                problem_type="multi_label_classification"
            ).to(self.device)
            
            self.threshold = threshold
            self.top_k = top_k
            self.max_length = max_length
            self.caption_weight = caption_weight
            self.hashtag_weight = hashtag_weight
            self.keyword_weight = keyword_weight
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise VibeClassificationError(f"Model loading failed: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for classification while preserving important features.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for keyword matching
        return text.lower()
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text.
        
        Args:
            text: Input text containing hashtags
            
        Returns:
            List of hashtags
        """
        return re.findall(r'#\w+', text)
    
    def _tokenize_inputs(
        self,
        caption: str,
        hashtags: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize inputs separately for caption and hashtags.
        
        Args:
            caption: Video caption
            hashtags: List of hashtags
            
        Returns:
            Dictionary of tokenized inputs
        """
        # Preprocess caption
        clean_caption = self._preprocess_text(caption)
        
        # Tokenize caption
        caption_tokens = self.tokenizer(
            clean_caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Tokenize hashtags if present
        if hashtags:
            hashtag_text = " ".join(hashtags)
            hashtag_tokens = self.tokenizer(
                hashtag_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
        else:
            hashtag_tokens = None
        
        return {
            "caption": caption_tokens,
            "hashtags": hashtag_tokens
        }
    
    def get_keyword_scores(self, text: str) -> np.ndarray:
        """Calculate keyword-based scores for each vibe"""
        scores = np.zeros(len(VIBES))
        text = text.lower()
        
        for i, vibe in enumerate(VIBES):
            keywords = VIBE_KEYWORDS[vibe]
            score = sum(1 for keyword in keywords if keyword in text)
            scores[i] = score / len(keywords) if keywords else 0
            
        return scores
    
    def classify_vibes(
        self,
        caption: str,
        hashtags: List[str]
    ) -> Dict[str, Union[List[str], List[float]]]:
        """
        Classify the vibe of a video based on its caption and hashtags.
        
        Args:
            caption: Video caption
            hashtags: List of hashtags
            
        Returns:
            Dictionary with classified vibes and confidence scores
            
        Raises:
            VibeClassificationError: If classification fails
        """
        try:
            # Get keyword-based scores first
            keyword_scores = self.get_keyword_scores(caption)
            keyword_scores = torch.tensor(keyword_scores, device=self.device).float()
            
            # Tokenize inputs
            tokens = self._tokenize_inputs(caption, hashtags)
            
            # Get predictions for caption
            with torch.no_grad():
                caption_outputs = self.model(**tokens["caption"])
                caption_logits = caption_outputs.logits
                caption_probs = torch.sigmoid(caption_logits)
            
            # Get predictions for hashtags if present
            if tokens["hashtags"] is not None:
                with torch.no_grad():
                    hashtag_outputs = self.model(**tokens["hashtags"])
                    hashtag_logits = hashtag_outputs.logits
                    hashtag_probs = torch.sigmoid(hashtag_logits)
                
                # Combine predictions with weights
                combined_probs = (
                    self.caption_weight * caption_probs +
                    self.hashtag_weight * hashtag_probs
                )
            else:
                combined_probs = caption_probs
            
            # Combine model predictions with keyword scores
            final_probs = (
                (1 - self.keyword_weight) * combined_probs +
                self.keyword_weight * keyword_scores
            )
            
            # Get top-k predictions
            probs = final_probs[0].cpu().numpy()
            top_indices = np.argsort(probs)[-self.top_k:][::-1]
            
            # Get vibes and scores
            selected_vibes = []
            confidence_scores = []
            
            for idx in top_indices:
                selected_vibes.append(VIBES[idx])
                confidence_scores.append(float(probs[idx]))
            
            # If no vibes meet threshold, return top vibes anyway
            if not selected_vibes:
                selected_vibes = [VIBES[idx] for idx in top_indices]
                confidence_scores = [float(probs[idx]) for idx in top_indices]
            
            logger.info(f"Classified vibes: {', '.join(selected_vibes)}")
            return {
                "vibes": selected_vibes,
                "confidence_scores": confidence_scores
            }
            
        except Exception as e:
            logger.error(f"Error in vibe classification: {str(e)}")
            raise VibeClassificationError(f"Failed to classify vibes: {str(e)}")
    
    def evaluate(
        self,
        validation_file: str,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate the model on a validation set.
        
        Args:
            validation_file: Path to validation set (JSON)
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Load validation data
            with open(validation_file, 'r') as f:
                validation_data = json.load(f)
            
            # Initialize metrics
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'mAP': 0.0
            }
            
            # Process in batches
            for i in range(0, len(validation_data), batch_size):
                batch = validation_data[i:i + batch_size]
                
                # Get predictions
                predictions = []
                for item in batch:
                    vibes = self.classify_vibes(
                        item['caption'],
                        item.get('hashtags') or []
                    )
                    predictions.append(vibes)
                
                # Calculate metrics
                # TODO: Implement metric calculation
                
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise VibeClassificationError(f"Evaluation failed: {str(e)}")

def process_video(
    video_id: str,
    caption: str,
    hashtags: List[str],
    output_file: str,
    threshold: float = 0.4,
    top_k: int = 3,
    caption_weight: float = 0.8,
    hashtag_weight: float = 0.2,
    keyword_weight: float = 0.4,
    debug: bool = False
) -> Dict:
    """
    Process a video and classify its vibe with optimized parameters.
    
    Args:
        video_id: Video identifier
        caption: Video caption
        hashtags: List of hashtags
        output_file: Path to save results
        threshold: Confidence threshold for classification
        top_k: Maximum number of vibes to return
        caption_weight: Weight for caption
        hashtag_weight: Weight for hashtags
        keyword_weight: Weight for keyword matching
        debug: Enable debug logging
        
    Returns:
        Dictionary with classification results
        
    Raises:
        VibeClassificationError: If processing fails
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    
    logger.debug(f"Processing video: {video_id}")
    logger.debug(f"Output file: {output_file}")
    
    # Initialize classifier
    classifier = VibeClassifier(
        threshold=threshold,
        top_k=top_k,
        caption_weight=caption_weight,
        hashtag_weight=hashtag_weight,
        keyword_weight=keyword_weight
    )
    
    # Classify vibes
    result = classifier.classify_vibes(caption, hashtags)
    
    # Prepare output
    output = {
        "video_id": video_id,
        "vibes": result["vibes"],
        "confidence_scores": result["confidence_scores"],
        "caption": caption,
        "hashtags": hashtags
    }
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Classified vibes: {', '.join(result['vibes'])}")
    logger.info(f"Saved results to {output_file}")
    logger.debug(f"Result: {output}")
    
    return output

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify video vibes")
    parser.add_argument("--video-id", required=True, help="Video ID")
    parser.add_argument("--caption", required=True, help="Video caption")
    parser.add_argument("--hashtags", nargs="*", help="List of hashtags")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--threshold", type=float, default=0.4, help="Classification threshold")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top vibes to return")
    parser.add_argument("--caption-weight", type=float, default=0.8, help="Weight for caption")
    parser.add_argument("--hashtag-weight", type=float, default=0.2, help="Weight for hashtags")
    parser.add_argument("--keyword-weight", type=float, default=0.4, help="Weight for keyword matching")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    process_video(
        video_id=args.video_id,
        caption=args.caption,
        hashtags=args.hashtags,
        output_file=args.output,
        threshold=args.threshold,
        top_k=args.top_k,
        caption_weight=args.caption_weight,
        hashtag_weight=args.hashtag_weight,
        keyword_weight=args.keyword_weight,
        debug=args.debug
    ) 