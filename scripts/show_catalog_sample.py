import json

with open('data/catalog.json', 'r', encoding='utf-8') as f:
    catalog = json.load(f)
    sample = {
        'catalog_version': catalog.get('version'),
        'total_items': catalog.get('total_items'),
        'first_product': catalog['items'][0]
    }

with open('data/catalog_sample.json', 'w', encoding='utf-8') as out:
    json.dump(sample, out, indent=2, ensure_ascii=False)

print('Sample written to data/catalog_sample.json') 