import os
import pandas as pd
from itertools import chain
from utils.formatting import parse_label_column
from utils.lookup import load_lookup_dict, reverse_translation_dict
from processing.guidance import load_guidance_csv

def load_classification_outputs(folder, keyword, model_name):
    llm_path = os.path.join(folder, f"{keyword}_classified_segments_{model_name}.csv")
    manual_path = os.path.join(folder, f"{keyword}.csv")

    llm_df = pd.read_csv(llm_path)
    manual_df = pd.read_csv(manual_path)
    return llm_df, manual_df

def merge_classifications(llm_df, manual_df, guidance_df, lookup_path='lookup_dictionaries.json'):
    # Rename for consistency
    llm_df = llm_df.rename(columns={"Segment ID": "segment_id", "Response": "llm_response"})
    manual_df = manual_df.rename(columns={"auto_issues": "manual_response"})

    # Merge
    df = pd.merge(llm_df, manual_df, on="segment_id", how="outer")
    df = df[["segment_id", "llm_response", "manual_response", "issue_type"]].copy()

    # ✅ Clear manual labels for unclassified rows (baseline fix)
    df.loc[df["issue_type"] == "issues", "manual_response"] = [[""]]

    # Parse responses into lists
    df["llm_response"] = parse_label_column(df["llm_response"])
    df["manual_response"] = parse_label_column(df["manual_response"])

    # Map manual labels using reverse lookup
    guidance_df.columns = guidance_df.columns.str.strip()
    valid_codes = set(guidance_df['Code'].dropna().unique())

    lookup = load_lookup_dict(lookup_path)
    rev_dict = reverse_translation_dict(lookup)

    df["manual_response"] = df["manual_response"].apply(
        lambda labels: [rev_dict.get(label, label) for label in labels]
    )

    return df

def summarise_labels(df):
    manual_labels = set(chain.from_iterable(df['manual_response']))
    llm_labels = set(chain.from_iterable(df['llm_response']))
    print(f"🧾 Manual Labels: {manual_labels}")
    print(f"🤖 LLM Labels: {llm_labels}")

def compute_classification_comparison(df, valid_codes, folder, keyword, model_name):
    """
    Computes TP, TN, FP, FN for each row in the classification DataFrame,
    saves the result to a comparison CSV.

    Args:
        df (pd.DataFrame): The merged classification DataFrame
        valid_codes (set): Set of all valid topic codes from guidance
        folder (str): Path to the keyword folder
        keyword (str): Keyword name
        model_name (str): Name of the model used

    Returns:
        pd.DataFrame: Updated DataFrame with TP, TN, FP, FN columns
    """

    def count_classification_outcomes(row):
        llm_set = set(row['llm_response'])
        manual_set = set(row['manual_response'])
        all_codes = valid_codes

        tp = len(llm_set & manual_set)
        fp = len(llm_set - manual_set)
        fn = len(manual_set - llm_set)
        tn = len(all_codes - (llm_set | manual_set))

        return pd.Series([tp, tn, fp, fn], index=["TP", "TN", "FP", "FN"])

    # Apply row-wise comparison
    df[["TP", "TN", "FP", "FN"]] = df.apply(count_classification_outcomes, axis=1)

    # Save result
    output_path = os.path.join(folder, f"{keyword}_comparison_{model_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"📄 Comparison saved to: {output_path}")

    return df

def compare_all_keywords_for_models(base_dir, model_names):
    """
    Loop through each keyword and model to compute comparison results (TP/TN/FP/FN).
    Saves one CSV per keyword-model combination.

    Args:
        base_dir (str): Path to the categories folder
        model_names (List[str]): List of model names to evaluate
    """
    keywords = [
        folder for folder in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, folder))
    ]

    for keyword in keywords:
        guidance_df = load_guidance_csv(base_dir, keyword)
        valid_codes = set(guidance_df['Code'].dropna().unique())
        folder = os.path.join(base_dir, keyword)

        for model_name in model_names:
            try:
                llm_df, manual_df = load_classification_outputs(folder, keyword, model_name)
                final_df = merge_classifications(llm_df, manual_df, guidance_df)
                compute_classification_comparison(final_df, valid_codes, folder, keyword, model_name)
                print(f"✅ Compared {keyword} using model '{model_name}'")
            except FileNotFoundError as e:
                print(f"⚠️  Skipped {keyword} ({model_name}) — {e}")
                