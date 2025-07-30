import json

def load_lookup_dict(path='lookup_dictionaries.json'):
    with open(path, 'r') as f:
        lookup = json.load(f)
    return lookup

def reverse_translation_dict(lookup):
    return {v: k for k, v in lookup.get("translation", {}).items()}
