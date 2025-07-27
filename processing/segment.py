import os
import pandas as pd

def load_segment_csv(base_dir, keyword):
    csv_path = os.path.join(base_dir, keyword, f"{keyword}.csv")
    return pd.read_csv(csv_path)