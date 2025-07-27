# LM-Studio-Classifier
A lightweight local classifier built using language models running in LM Studio. This project demonstrates how to leverage local LLMs for text classification tasks without relying on cloud APIs. Includes prompt templates, model configurations, and examples for building custom classification workflows using LM Studio.

This project processes and prepares categorized data for classification tasks. It supports merging labeled and unlabeled data, formatting guidance information, and constructing prompts for downstream models.

---

## 📁 Project Structure

```text
LM-Studio-Classifier/
├── categories/                  # Contains keyword-specific data folders
│   └── keyword/                # Each folder corresponds to a classification category
│       ├── keyword.csv
│       ├── keyword_classified.csv
│       ├── keyword_unclassified.csv
│       └── keyword_guidance.csv
├── processing/                  # Data processing logic
│   ├── merge.py                # Merge classified and unclassified CSVs
│   ├── guidance.py             # Load & rename guidance files
│   ├── segment.py              # Load keyword data CSV
├── utils/                       # Utility functions
│   ├── paths.py                # get_base_dir() utility
│   ├── prompt.py               # Load prompt text, format guidance, build full prompt
├── classifier/                  # Model communication and classification pipeline
│   ├── lm_interface.py         # Call LM Studio API with prompt and model name
│   ├── utils.py                # Extract list of codes from model response
│   ├── run.py                  # classification() and find_unclassified_keywords()
├── prompt.txt                   # Reusable prompt instruction template


```

---

## ⚙️ Key Functions

### 🔄 Merging
- `merge_classified_and_unclassified(base_dir)`
  - Merges `_classified.csv` and `_unclassified.csv` into a single `keyword.csv` per category.
  
### 📚 Guidance
- `rename_guidance_files(base_dir)`
  - Renames `guidance.csv` to `keyword_guidance.csv` if needed.
- `load_guidance_csv(base_dir, keyword)`
  - Loads the structured guidance table for the keyword.

### 🔍 Data Loading
- `load_segment_csv(base_dir, keyword)`
  - Loads the main `keyword.csv` for analysis or classification.

### 🧠 Prompt Construction
- `load_prompt(file_path)`
  - Loads the base instruction or template for the model prompt.
- `get_guidance_table(guidance_df)`
  - Converts the guidance DataFrame into a markdown-formatted table.
- `build_prompt(segment_id, segment_text, guidance_string, instruction_text)`
  - Constructs the final prompt to send to the language model.
  - Combines the base instruction, guidance table, and the current text segment.
  - Produces a structured prompt with strict response formatting rules.

### 🤖 Model Inference
- `classification(base_dir, keyword, model_name)`  
  Runs the full classification pipeline:
  - Loads guidance, prompt text, and segments
  - Builds a prompt for each segment
  - Sends each prompt to LM Studio via API
  - Parses and saves predictions to a CSV file  
  **Output saved to:**  
  `categories/<keyword>/<keyword>_classified_segments_<model_name>.csv`

- `call_lm_studio(prompt, model_name, server_url="http://localhost:1234/v1/completions")`  
  Sends a prompt to a local LM Studio server and returns a list of topic codes extracted from the model's response.  
  Accepts a model name and optional server URL.

- `extract_list_from_response(response)`  
  Extracts topic codes from the model’s raw response.  
  - Attempts to parse a JSON list first (e.g. `["GChRhet", "GChSubs"]`)  
  - Falls back to extracting quoted strings if the response is unstructured
