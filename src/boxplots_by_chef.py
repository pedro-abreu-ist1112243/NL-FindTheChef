#!/usr/bin/env python3

import sys
import os
import ast
import json
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt


def safe_parse_list_cell(cell: Any):
    """Try to parse a stringified Python/JSON list (or return [] on failure)."""
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, tuple)):
        return list(cell)
    s = str(cell).strip()
    if s == "" or s == "nan":
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            if "," in s and not (s.startswith("{") and s.endswith("}")):
                return [part.strip() for part in s.split(",") if part.strip()]
            return []


def ensure_numeric_col(df: pd.DataFrame, col: str, default=0):
    """Ensure column exists and is numeric (fill missing with default)."""
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    return df


def main(train_csv: str, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)

    df = pd.read_csv(train_csv, sep=";", engine="python", dtype=str, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    for col in ["tags", "steps", "ingredients"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list_cell)
        else:
            df[col] = [[] for _ in range(len(df))]

    if "n_ingredients" not in df.columns:
        df["n_ingredients"] = df["ingredients"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0)
    else:
        df["n_ingredients"] = pd.to_numeric(df["n_ingredients"], errors="coerce").fillna(0).astype(int)

    df["n_tags"] = df["tags"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0)
    df["n_steps"] = df["steps"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 0)
    df["len_recipe_name"] = df.get("recipe_name", "").fillna("").astype(str).apply(len)
    df["len_description"] = df.get("description", "").fillna("").astype(str).apply(len)

    if "chef_id" not in df.columns:
        raise RuntimeError("chef_id column not found in the CSV.")
    df["chef_id"] = df["chef_id"].astype(str).str.strip()

    feature_to_col = {
        "n_ingredients": "Number of ingredients",
        "n_tags": "Number of tags",
        "n_steps": "Number of steps",
        "len_recipe_name": "Recipe name length (chars)",
        "len_description": "Description length (chars)",
    }

    for feat in feature_to_col.keys():
        df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0)

    chef_order = sorted(df["chef_id"].unique(), key=lambda x: int(x) if x.isdigit() else x)

    len_desc_clip = int(df["len_description"].quantile(0.95))

    for feat, title in feature_to_col.items():
        grouped = []
        labels = []
        for chef in chef_order:
            series = df.loc[df["chef_id"] == chef, feat].dropna().astype(float)
            if series.empty:
                grouped.append([0.0])
            else:
                grouped.append(series.values)
            labels.append(str(chef))

        plt.figure(figsize=(10, 6))
        box = plt.boxplot(grouped, labels=labels, patch_artist=True, showfliers=True)
        plt.title(f"{title} by chef_id (training set)")
        plt.xlabel("chef_id")
        plt.ylabel(title)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

        if feat == "len_description":
            plt.ylim(0, len_desc_clip)

        # Color boxes
        colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"]
        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.9)

        out_path = os.path.join(out_folder, f"box_{feat}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/boxplots_by_chef.py <train_csv> <output_folder>")
        sys.exit(1)
    train_csv = sys.argv[1]
    out_folder = sys.argv[2]
    main(train_csv, out_folder)

# use with 'python src/boxplots_by_chef.py data/train.csv figures/boxplots_by_chef'
