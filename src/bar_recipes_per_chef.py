#!/usr/bin/env python3


import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def main(train_csv: str, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df = pd.read_csv(train_csv, sep=";", engine="python", on_bad_lines="skip")

    # Check required column
    if "chef_id" not in df.columns:
        raise RuntimeError("chef_id column not found in the dataset.")

    counts = df["chef_id"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index.astype(str), counts.values, color="#86bf91", edgecolor="black")

    # Labels and title
    plt.title("Number of Recipes per Chef (Training Set)")
    plt.xlabel("chef_id")
    plt.ylabel("Number of Recipes")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{int(height)}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved bar chart to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/bar_recipes_per_chef.py <train_csv> <output_path>")
        sys.exit(1)

    train_csv = sys.argv[1]
    output_path = sys.argv[2]
    main(train_csv, output_path)

#run with 'python src/bar_recipes_per_chef.py data/train.csv figures/bar_recipes_per_chef.png'