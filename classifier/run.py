import os
import time
import pandas as pd
from processing.guidance import load_guidance_csv
from processing.segment import load_segment_csv
from utils.prompt import load_prompt, get_guidance_table, build_prompt
from classifier.lm_interface import call_lm_studio

def classification(base_dir, keyword, model_name):
    folder = os.path.join(base_dir, keyword)

    # Load guidance and prompt
    guidance_df = load_guidance_csv(base_dir, keyword)
    guidance_string = get_guidance_table(guidance_df)
    prompt_instruction = load_prompt("prompt.txt")

    # Load segments to classify
    label_df = load_segment_csv(base_dir, keyword)
    results = []
    total = len(label_df)

    for i, (idx, row) in enumerate(label_df.iterrows(), start=1):
        prompt = build_prompt(
            segment_id=row['segment_id'],
            segment_text=row['segment_text'],
            guidance_string=guidance_string,
            instruction_text=prompt_instruction
        )

        try:
            result_text = call_lm_studio(prompt, model_name)
            results.append({
                "Segment ID": row['segment_id'],
                "Segment Text": row['segment_text'],
                "Response": result_text
            })
            print(f"✅ [{i}/{total}] [{keyword}] Processed Segment ID {row['segment_id']}")
        except Exception as e:
            print(f"❌ [{i}/{total}] [{keyword}] Error on Segment ID {row['segment_id']}: {e}")
            results.append({
                "Segment ID": row['segment_id'],
                "Segment Text": row['segment_text'],
                "Response": "ERROR"
            })

        time.sleep(1.0)  # Rate limit requests

    # Save results
    output_file = os.path.join(folder, f"{keyword}_classified_segments_{model_name}.csv")
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"🎉 Done! Results saved to '{output_file}'")

def find_unclassified_keywords(base_dir, model_name):
    """
    Scans category folders and returns a list of keywords that do not have
    the expected classification output file.

    Args:
        base_dir (str): Root directory containing keyword folders.
        model_name (str): Name of the model used for classification.

    Returns:
        List[str]: List of keywords missing the classified output file.
    """
    missing = []

    for keyword in os.listdir(base_dir):
        keyword_path = os.path.join(base_dir, keyword)

        if not os.path.isdir(keyword_path):
            continue  # skip files

        expected_file = os.path.join(
            keyword_path, f"{keyword}_classified_segments_{model_name}.csv"
        )

        if not os.path.exists(expected_file):
            missing.append(keyword)

    return missing
