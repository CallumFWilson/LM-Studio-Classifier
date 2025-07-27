# LM-Studio-Classifier
A lightweight local classifier built using language models running in LM Studio. This project demonstrates how to leverage local LLMs for text classification tasks without relying on cloud APIs. Includes prompt templates, model configurations, and examples for building custom classification workflows using LM Studio.

This project processes and prepares categorized data for classification tasks. It supports merging labeled and unlabeled data, formatting guidance information, and constructing prompts for downstream models.

---

## 📁 Project Structure

project/
├── categories/ # Contains keyword-specific data folders
│ └── keyword/
│ ├── keyword.csv
│ ├── keyword_classified.csv
│ ├── keyword_unclassified.csv
│ └── keyword_guidance.csv
├── processing/
│ ├── merge.py # Merging classified and unclassified CSVs
│ ├── guidance.py # Load & rename guidance files
│ ├── segment.py # Load keyword data CSV
├── utils/
│ ├── paths.py # get_base_dir() utility
│ ├── prompt.py # Load prompt text and format guidance tables
├── prompt.txt # Reusable prompt template
├── submit.py # Main entry point to run the pipeline

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
