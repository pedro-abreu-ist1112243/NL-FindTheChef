import pandas as pd
import numpy as np
import joblib
import preprocessing
import bert_train
from logging import info


def main(
    artifacts_path: str,
    bert_dir: str,
    recipe_path: str,
    output_file: str | None = None,
) -> np.ndarray:
    info("Loading bert and artifacts...")
    bert = bert_train.get_cached_bert_model(bert_dir)
    artifacts = load_artifacts(artifacts_path)

    info("Loading recipe...")
    recipe = preprocessing.process_csv(recipe_path)

    predictions = predict_chefs(recipe, artifacts, bert)

    # Save predictions to a text file, one prediction per line
    if output_file is not None:
        with open(output_file, "w") as f:
            info(f"Saving predictions to {output_file}")
            for pred in predictions:
                _ = f.write(f"{pred}\n")
    return predictions


def load_artifacts(artifacts_path: str):
    info(f"Loading artifacts from {artifacts_path}")
    artifacts = joblib.load(artifacts_path)
    return artifacts


def predict_chefs(
    recipes: pd.DataFrame, artifacts, bert
) -> np.ndarray:
    model = artifacts["model"]
    bounds = artifacts["bounds"]
    feature_cols = artifacts["feature_cols"]

    recipes = preprocessing.add_text_column(
        recipes,
        text_columns=["recipe_name", "description"],
        list_columns=["ingredients", "steps", "tags"],
        col_name="text",
    )

    recipes = bert_train.predict_with_bert(bert, recipes, "text")

    preprocessing.add_count_columns(recipes, ["tags", "steps", "ingredients"])
    preprocessing.add_length_columns(recipes, ["recipe_name", "description"])
    preprocessing.add_date_features(recipes, "data")
    preprocessing.normalize_features(
        recipes,
        [
            "n_ingredients",
            "n_tags",
            "n_steps",
            "len_recipe_name",
            "len_description",
            "year",
        ],
        bounds,
    )

    final_features = recipes[feature_cols]  # Ensure correct column order

    info("Predicting chef...")
    predictions = model.predict(final_features)
    info(f"Predicted chef_ids:\n{predictions}")

    return predictions


if __name__ == "__main__":
    import os
    import logging

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    _ = main("classifier_artifacts.joblib", "bert_model", "data/test-no-labels.csv", "results.txt")
