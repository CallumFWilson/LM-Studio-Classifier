import os
import pandas as pd

def rename_guidance_files(base_dir):
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            original_file = os.path.join(subfolder_path, "guidance.csv")
            renamed_file = os.path.join(subfolder_path, f"{subfolder}_guidance.csv")

            if os.path.exists(original_file):
                if not os.path.exists(renamed_file):
                    os.rename(original_file, renamed_file)
                    print(f"Renamed: {original_file} → {renamed_file}")
                else:
                    print(f"Skipped: {renamed_file} already exists")

def load_guidance_csv(base_dir, keyword):
    guidance_path = os.path.join(base_dir, keyword, f"{keyword}_guidance.csv")
    return pd.read_csv(guidance_path)
    