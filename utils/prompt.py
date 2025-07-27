import pandas as pd

def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_guidance_table(guidance_df):
    # Format the guidance data into a table format for the prompt
    guidance_table = "# Classification Guidance Table\n\n"
    guidance_table += "| Code | Descriptor | Include | Exclude |\n"
    guidance_table += "|------|-----------|---------|----------|\n"
    
    for _, row in guidance_df.iterrows():
        code = row['Code']
        descriptor = row['Descriptor']
        include = row['Include'] if not pd.isna(row['Include']) else ""
        exclude = row['Exclude'] if not pd.isna(row['Exclude']) else ""
        
        guidance_table += f"| {code} | {descriptor} | {include} | {exclude} |\n"
    return guidance_table
        