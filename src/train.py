import preprocessing
import joblib
import bert_train
import pandas as pd
from logging import info
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import classification_report, confusion_matrix


def main(
    dataset_path: str, artifacts_path: str, grid_search: bool = False, seed: int = 42
):
    ##### Load and parse data
    df = preprocessing.process_csv(dataset_path)

    df = preprocessing.add_text_column(
        df,
        text_columns=["recipe_name", "description"],
        list_columns=["ingredients", "steps", "tags"],
        col_name="text",
    )

    ##### Train validation test split
    X_train, X_val, X_test, y_train, y_val, y_test = split(df, "chef_id", seed=seed)

    ##### Fine tune BERT model
    train_df = pd.concat([X_train, y_train], axis=1)
    bert = bert_train.train_bert(
        train_df,
        text_col="text",
    )

    ##### Get BERT predictions
    cols = set(X_train.columns.tolist())
    X_train = bert_train.predict_with_bert(bert, X_train, "text")
    X_val = bert_train.predict_with_bert(bert, X_val, "text")
    X_test = bert_train.predict_with_bert(bert, X_test, "text")
    bert_cols = list(set(X_train.columns.tolist()) - cols)

    ##### Feature extraction and normalization
    X_train, X_val, X_test, bounds = feature_extraction(X_train, X_val, X_test)

    ##### Train model
    feature_cols = [
        "norm_n_ingredients",
        "norm_n_tags",
        "norm_n_steps",
        "norm_len_recipe_name",
        "norm_len_description",
        "norm_year",
        "day_sin",
        "day_cos",
    ]
    feature_cols += bert_cols

    mlp = train(X_train, y_train, X_val, y_val, feature_cols, grid_search, seed=seed)

    ##### Evaluate model
    info(f"\n--- Final model evaluation on hold-out test set ---")
    test_predictions = mlp.predict(X_test[feature_cols])

    info(f"Test accuracy: {mlp.score(X_test[feature_cols], y_test)}")
    class_labels = mlp.classes_

    info("Classification Report:")
    print(
        classification_report(
            y_test, test_predictions, target_names=[str(c) for c in class_labels]
        )
    )

    info("Confusion Matrix:")
    cm = confusion_matrix(y_test, test_predictions, labels=class_labels)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"
    print(cm_df)
    save_artifacts(
        mlp,
        bounds,
        feature_cols,
        artifacts_path,
    )

def split(
    df: pd.DataFrame,
    y_col: str,
    test_size=0.20,
    val_size=0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[y_col])
    y = df[y_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_val,
    )

    # Create explicit copies to avoid SettingWithCopyWarning
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

    info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def feature_extraction(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[int, int]]]:
    # Feature extraction and normalization
    preprocessing.add_count_columns(X_train, ["tags", "steps"])
    preprocessing.add_length_columns(X_train, ["recipe_name", "description"])
    preprocessing.add_date_features(X_train, "data")
    bounds = preprocessing.normalize_features(
        X_train,
        [
            "n_ingredients",
            "n_tags",
            "n_steps",
            "len_recipe_name",
            "len_description",
            "year",
        ],
    )

    preprocessing.add_count_columns(X_val, ["tags", "steps"])
    preprocessing.add_length_columns(X_val, ["recipe_name", "description"])
    preprocessing.add_date_features(X_val, "data")
    _ = preprocessing.normalize_features(
        X_val,
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

    preprocessing.add_count_columns(X_test, ["tags", "steps"])
    preprocessing.add_length_columns(X_test, ["recipe_name", "description"])
    preprocessing.add_date_features(X_test, "data")
    _ = preprocessing.normalize_features(
        X_test,
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

    return X_train, X_val, X_test, bounds


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_cols: list[str],
    do_grid_search: bool = False,
    k: int = 5,
    seed: int = 42,
):
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    if not do_grid_search:
        # Model discovered by previous grid search
        mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-3,
            learning_rate_init=0.001,
            activation="relu",
            max_iter=1000,
            random_state=seed,
            solver="adam",
        )

        pipeline = ImbPipeline(
            [
                ("smote", SMOTE(random_state=seed)),
                ("mlp", mlp),
            ]
        )

        info(f"Fitting model on training data...")
        pipeline.fit(X_train[feature_cols], y_train)

        val_score = pipeline.score(X_val[feature_cols], y_val)
        info(f"Validation accuracy: {val_score}")

        # Fit the final model on all training data
        pipeline.fit(X_train_val[feature_cols], y_train_val)
        return pipeline

    # Grid search branch
    param_grid = {
        "mlp__hidden_layer_sizes": [
            (64,),
            (128,),
            (256,),
            (64, 32),
            (128, 64),
            (256, 128),
        ],
        "mlp__alpha": [1e-3, 1e-4, 1e-5],
        "mlp__learning_rate_init": [0.01, 0.001],
        "mlp__activation": ["relu", "tanh"],
    }

    pipeline = ImbPipeline(
        [
            ("smote", SMOTE(random_state=seed)),
            ("mlp", MLPClassifier(max_iter=1000, random_state=seed, solver="adam")),
        ]
    )

    split_index = [-1] * len(X_train) + [0] * len(X_val)
    pds = PredefinedSplit(test_fold=split_index)

    info(f"Starting grid search over hyperparameters...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=pds,
        n_jobs=-1,
        verbose=2,  # print progress
        # refit=True # default is True
    )

    start_time = time.time()

    grid_search.fit(X_train_val[feature_cols], y_train_val)

    end_time = time.time()
    duration = end_time - start_time
    info(f"Grid search completed in {int(duration // 60):02}:{int(duration % 60):02}")

    info(f"Best parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    info(f"Best cross-validation accuracy: {grid_search.best_score_}")

    return best_model


def save_artifacts(
    model: ImbPipeline,
    bounds: dict[str, tuple[int, int]],
    feature_cols: list[str],
    file_path: str,
):
    artifacts = {
        "model": model,
        "bounds": bounds,
        "feature_cols": feature_cols,
    }
    joblib.dump(artifacts, file_path)
    info(f"Saved model artifacts to {file_path}")


if __name__ == "__main__":
    import os
    import logging
    import sys

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    args = sys.argv[1:]
    grid_search = False
    if len(args) > 0 and args[0] == "grid":
        grid_search = True
    main("data/train.csv", "classifier_artifacts.joblib", grid_search)
