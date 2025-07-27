import os
import pandas as pd

def merge_classified_and_unclassified(base_dir):
    """
    Merges classified and unclassified CSV files in subdirectories under the given base directory.

    Args:
        base_dir (str): Path to the base directory containing category subfolders.
    """
    if not os.path.isdir(base_dir):
        print(f"❌ Base directory does not exist: {base_dir}")
        return

    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)

        if not os.path.isdir(category_path):
            continue  # Skip files, only process folders

        for file in os.listdir(category_path):
            if file.endswith("_classified.csv"):
                keyword = file.replace("_classified.csv", "")
                classified_path = os.path.join(category_path, f"{keyword}_classified.csv")
                unclassified_path = os.path.join(category_path, f"{keyword}_unclassified.csv")
                merged_path = os.path.join(category_path, f"{keyword}.csv")

                if not os.path.exists(unclassified_path):
                    print(f"⚠️  Missing unclassified file for '{keyword}' in '{category}'")
                    continue

                try:
                    classified_df = pd.read_csv(classified_path)
                    unclassified_df = pd.read_csv(unclassified_path)

                    # Rename 'issues' in unclassified to 'auto_issues'
                    unclassified_df_renamed = unclassified_df.rename(columns={"issues": "auto_issues"})

                    # Add missing columns
                    for col in classified_df.columns:
                        if col not in unclassified_df_renamed.columns:
                            unclassified_df_renamed[col] = pd.NA
                    unclassified_df_renamed = unclassified_df_renamed[classified_df.columns]

                    # Add issue_type
                    classified_df['issue_type'] = 'auto_issues'
                    unclassified_df_renamed['issue_type'] = 'issues'

                    # Merge and save
                    merged_df = pd.concat([classified_df, unclassified_df_renamed], ignore_index=True)
                    merged_df.to_csv(merged_path, index=False)

                    print(f"📁 Merged file saved to: {merged_path}")

                except Exception as e:
                    print(f"❌ Error processing '{keyword}' in '{category}': {e}")
                    