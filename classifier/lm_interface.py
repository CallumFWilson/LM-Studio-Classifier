import requests
from classifier.utils import extract_list_from_response

def call_lm_studio(prompt, model_name, server_url="http://localhost:1234/v1/completions"):
    try:
        response = requests.post(
            server_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": model_name,
                "prompt": prompt,
                "temperature": 0,
                "stop": ["\n", "</s>"]
            },
            timeout=1000
        )
        response.raise_for_status()
        result = response.json()
        return extract_list_from_response(result.get("choices", [{}])[0].get("text", "").strip())
    except requests.exceptions.RequestException as e:
        print("❌ Request error:", e)
        return "ERROR"
        