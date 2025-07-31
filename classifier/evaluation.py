import os
import pandas as pd
import ast
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def generate_model_comparisons(model_names, base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    total_sums = {}  # Accumulator for totals across all keywords

    for keyword in os.listdir(base_dir):
        keyword_path = os.path.join(base_dir, keyword)
        if not os.path.isdir(keyword_path):
            continue

        print(f"\n📁 Keyword: {keyword}")
        
        comparison_data = {
            "auto_issues_TP": {},
            "auto_issues_FP": {},
            "auto_issues_FN": {},
            "issues_TP": {},
            "issues_FP": {},
        }

        baseline_auto_TP = None
        baseline_issues_TP = None

        for model in model_names:
            file_path = os.path.join(keyword_path, f"{keyword}_comparison_{model}.csv")
            if not os.path.exists(file_path):
                print(f"⚠️ Missing file: {file_path}")
                continue

            df = pd.read_csv(file_path)

            auto_issues_df = df[df["issue_type"] == "auto_issues"]
            auto_TP, auto_FP, auto_FN = auto_issues_df[["TP", "FP", "FN"]].sum()
            auto_BL = auto_issues_df["manual_response"].apply(ast.literal_eval).apply(len).sum()

            issues_df = df[df["issue_type"] == "issues"]
            issues_TP = issues_df["llm_response"].apply(lambda x: ast.literal_eval(x) == []).sum()
            issues_FP, _ = issues_df[["FP", "FN"]].sum()
            issues_BL = issues_df["manual_response"].apply(lambda x: ast.literal_eval(x) == []).sum()

            if model == model_names[0]:
                baseline_auto_TP = auto_BL
                baseline_issues_TP = issues_BL

                # Track total baseline
                total_sums.setdefault("auto_issues_TP", {}).setdefault("baseline", 0)
                total_sums["auto_issues_TP"]["baseline"] += auto_BL
                total_sums.setdefault("issues_TP", {}).setdefault("baseline", 0)
                total_sums["issues_TP"]["baseline"] += issues_BL

            # Model totals
            for metric, val in [
                ("auto_issues_TP", auto_TP),
                ("auto_issues_FP", auto_FP),
                ("auto_issues_FN", auto_FN),
                ("issues_TP", issues_TP),
                ("issues_FP", issues_FP)
            ]:
                total_sums.setdefault(metric, {}).setdefault(model, 0)
                total_sums[metric][model] += val

            # Populate per-keyword output
            comparison_data["auto_issues_TP"][model] = auto_TP
            comparison_data["auto_issues_FP"][model] = auto_FP
            comparison_data["auto_issues_FN"][model] = auto_FN
            comparison_data["issues_TP"][model] = issues_TP
            comparison_data["issues_FP"][model] = issues_FP

        # Add baseline to per-keyword comparison
        comparison_data["auto_issues_TP"]["baseline"] = baseline_auto_TP
        comparison_data["auto_issues_FP"]["baseline"] = "-"
        comparison_data["auto_issues_FN"]["baseline"] = "-"
        comparison_data["issues_TP"]["baseline"] = baseline_issues_TP
        comparison_data["issues_FP"]["baseline"] = "-"

        # Save individual keyword CSV
        df_comparison = pd.DataFrame(comparison_data).T
        df_comparison = df_comparison[["baseline"] + model_names]
        df_comparison.to_csv(os.path.join(output_dir, f"{keyword}_model_comparison.csv"))

        display(df_comparison)
        print("—" * 50)

    # Create and save TOTAL summary
    if total_sums:
        total_df = pd.DataFrame(total_sums).T
        total_df = total_df[["baseline"] + model_names]  # Ensure column order
        total_df.to_csv(os.path.join(output_dir, "TOTAL_model_comparison.csv"))
        print(f"\n✅ TOTAL summary saved to {output_dir}/TOTAL_model_comparison.csv")
        display(total_df)

def generate_model_venn_diagrams(model_names, base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def parsed(row, col):
        try:
            return ast.literal_eval(row[col])
        except Exception:
            return []

    for keyword in os.listdir(base_dir):
        keyword_path = os.path.join(base_dir, keyword)
        if not os.path.isdir(keyword_path):
            continue

        print(f"\n📁 Keyword: {keyword}")

        # Prepare all model data for this keyword
        venn_data = {
            "classified": [],
            "unclassified": []
        }

        for model in model_names:
            file_path = os.path.join(keyword_path, f"{keyword}_comparison_{model}.csv")
            if not os.path.exists(file_path):
                print(f"⚠️ Missing file: {file_path}")
                continue

            df = pd.read_csv(file_path)

            auto_issues_df = df[df["issue_type"] == "auto_issues"]
            issues_df = df[df["issue_type"] == "issues"]

            # Baseline
            BL_classified = set(auto_issues_df['segment_id'])
            BL_unclassified = set(issues_df['segment_id'])

            # Model outcomes
            TP = {row['segment_id'] for _, row in auto_issues_df.iterrows()
                  if len(parsed(row, 'llm_response')) > 0 and len(parsed(row, 'manual_response')) > 0}
            TN = {row['segment_id'] for _, row in issues_df.iterrows()
                  if len(parsed(row, 'llm_response')) == 0 and len(parsed(row, 'manual_response')) == 0}
            FP = {row['segment_id'] for _, row in df.iterrows()
                  if len(parsed(row, 'manual_response')) == 0 and len(parsed(row, 'llm_response')) > 0}
            FN = {row['segment_id'] for _, row in df.iterrows()
                  if len(parsed(row, 'manual_response')) > 0 and len(parsed(row, 'llm_response')) == 0}

            venn_data["classified"].append((BL_classified, TP.union(FP), model))
            venn_data["unclassified"].append((BL_unclassified, TN.union(FN), model))

        # Create grid of subplots (2 rows: classified/unclassified, N columns for N models)
        fig, axes = plt.subplots(2, len(model_names), figsize=(6 * len(model_names), 10))

        for col, (bl_set, model_set, model_name) in enumerate(venn_data["classified"]):
            ax = axes[0, col] if len(model_names) > 1 else axes[0]
            plt.sca(ax)
            venn2([bl_set, model_set], set_labels=("BL_classified", "TP ∪ FP"))
            ax.set_title(f"{model_name} — Classified")

        for col, (bl_set, model_set, model_name) in enumerate(venn_data["unclassified"]):
            ax = axes[1, col] if len(model_names) > 1 else axes[1]
            plt.sca(ax)
            venn2([bl_set, model_set], set_labels=("BL_unclassified", "TN ∪ FN"))
            ax.set_title(f"{model_name} — Unclassified")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{keyword}_venn_comparison_grid.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Saved grid Venn diagram: {plot_path}")