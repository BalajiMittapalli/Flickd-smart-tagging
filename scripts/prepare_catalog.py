"""
Script to prepare the product catalog by combining Excel and CSV data into a single JSON file.
"""

import json
import logging
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def prepare_catalog(excel_path='data/Catalog.xlsx', 
                   csv_path='data/images.csv',
                   output_path='data/catalog.json',
                   image_dir='data/catalog_images'):
    """
    Prepare the product catalog by combining Excel and CSV data.
    
    Args:
        excel_path (str): Path to the Excel catalog file
        csv_path (str): Path to the CSV images file
        output_path (str): Path to save the output JSON file
        image_dir (str): Directory to save catalog images
    """
    try:
        # Create image directory if it doesn't exist
        os.makedirs(image_dir, exist_ok=True)
        
        # Read Excel catalog
        logging.info(f"Reading Excel catalog from {excel_path}")
        catalog_df = pd.read_excel(excel_path)
        
        # Read images CSV
        logging.info(f"Reading images CSV from {csv_path}")
        images_df = pd.read_csv(csv_path)
        
        # Group images by product ID
        product_images = {}
        for _, row in tqdm(images_df.iterrows(), desc="Processing images"):
            product_id = str(row['id'])
            if product_id not in product_images:
                product_images[product_id] = []
            product_images[product_id].append(row['image_url'])
        
        # Create catalog items
        catalog_items = []
        for _, row in tqdm(catalog_df.iterrows(), desc="Processing catalog items"):
            product_id = str(row['id'])
            item = {
                'id': product_id,
                'title': row['title'],
                'description': row.get('description', ''),
                'price': float(row.get('mrp', 0)),
                'discount_percentage': float(row.get('discount_percentage', 0)),
                'product_type': row.get('product_type', ''),
                'alias': row.get('alias', ''),
                'tags': row.get('product_tags', '').split(',') if pd.notna(row.get('product_tags')) else [],
                'collections': row.get('product_collections', '').split(',') if pd.notna(row.get('product_collections')) else [],
                'images': product_images.get(product_id, [])
            }
            catalog_items.append(item)
        
        # Save catalog to JSON
        logging.info(f"Saving catalog to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'version': '1.0',
                'total_items': len(catalog_items),
                'items': catalog_items
            }, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Catalog preparation complete. Total items: {len(catalog_items)}")
        
    except Exception as e:
        logging.error(f"Error preparing catalog: {str(e)}")
        raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare product catalog from Excel and CSV data')
    parser.add_argument('--excel', default='data/Catalog.xlsx', help='Path to Excel catalog file')
    parser.add_argument('--csv', default='data/images.csv', help='Path to CSV images file')
    parser.add_argument('--output', default='data/catalog.json', help='Path to save output JSON file')
    parser.add_argument('--image-dir', default='data/catalog_images', help='Directory to save catalog images')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    prepare_catalog(
        excel_path=args.excel,
        csv_path=args.csv,
        output_path=args.output,
        image_dir=args.image_dir
    ) 