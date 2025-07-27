import re
import json

def extract_list_from_response(response):
    # Try to extract a JSON-style list
    pattern = r'\[(?:[^\[\]]*(?:"[^"]*"[^\[\]]*)*)*\]'
    match = re.search(pattern, response)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: extract quoted codes individually
    codes = re.findall(r'"([A-Za-z]+)"', response)
    return codes if codes else []
