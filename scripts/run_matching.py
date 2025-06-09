import os
import json
from pathlib import Path
from match_products import ProductMatcher

def run_matching(detections_path, output_path):
    """
    Run product matching on an existing detections.json file
    
    Args:
        detections_path (str): Path to the detections.json file
        output_path (str): Path where to save the matched_products.json file
    """
    try:
        # Initialize matcher
        matcher = ProductMatcher()
        
        # Load catalog
        catalog_path = "data/catalog.json"
        if not os.path.exists(catalog_path):
            raise Exception(f"Catalog file not found at {catalog_path}")
        print(f"Loading catalog from {catalog_path}")
        matcher.load_catalog(catalog_path)
        
        # Run matching
        print(f"Running product matching on {detections_path}")
        matches = matcher.match_products(detections_path, output_path)
        
        # Print results
        if matches and isinstance(matches, dict) and "products" in matches:
            print(f"\nFound {len(matches['products'])} matches:")
            for match in matches["products"]:
                if match.get("confidence", 0) >= 0.75:  # Only show high confidence matches
                    print(f"- {match.get('type', 'Unknown')} ({match.get('color', 'Unknown')})")
                    print(f"  Match: {match.get('match_label', 'Unknown')}")
                    print(f"  Confidence: {match.get('confidence', 0):.2f}")
                    print(f"  Product ID: {match.get('matched_product_id', 'Unknown')}")
                    print()
        else:
            print("No matches found or invalid matches format")
            
    except Exception as e:
        print(f"Error during matching: {str(e)}")

if __name__ == "__main__":
    # Example usage
    detections_path = r"D:\Projects\flickd-smart-tagging\outputs\2025-05-28_13-40-09_UTC\detections.json"
    output_path = r"D:\Projects\flickd-smart-tagging\outputs\2025-05-28_13-40-09_UTC\matched_products.json"
    
    run_matching(detections_path, output_path) 