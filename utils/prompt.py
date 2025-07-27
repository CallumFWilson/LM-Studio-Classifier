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

def build_prompt(segment_id, segment_text, guidance_string, instruction_text):
    return f"""{instruction_text}\n\n{guidance_string}

Please classify the following text segment using the provided guidance:

Segment ID: {segment_id}
Segment Text: "{segment_text}"

For each text segment, return the list of topic codes that best match the semantic meaning of the segment.
If the segment doesn't match any topic codes, return an empty list.
Example responses:
- For a segment matching multiple codes: ["GChRhet","GChSubs"]
- For a segment matching one code: ["GChRhet"]
- For a segment matching no codes: []

Return only the list of code or empty list with no additional text or explanation.
"""
    