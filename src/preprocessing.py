import pandas as pd
import numpy as np
import ast  # For evaluating string lists
from typing import cast
from logging import debug


def process_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a recipe CSV, sets the correct data types, and processes list-like columns.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The processed DataFrame.
    """
    debug(f"Loading data from {file_path}")
    # Can only handle list columns only after loading
    data_types = {
        "chef_id": np.int64,
        "recipe_name": pd.StringDtype(),
        "description": pd.StringDtype(),
        "n_ingredients": np.int64,
    }

    df = pd.read_csv(
        file_path,
        sep=";",
        dtype=data_types,  # due to pandas' stubs and python's type checker, this line will always error, sadly
        parse_dates=["data"],
        dayfirst=True,
    )

    list_columns = ["tags", "steps", "ingredients"]
    for col in list_columns:
        df[col] = df[col].apply(ast.literal_eval)

    return df


def add_text_column(
    df: pd.DataFrame,
    text_columns: list[str],
    list_columns: list[str],
    col_name: str = "text",
) -> pd.DataFrame:
    """
    Combine multiple text columns and list columns into a single text column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        text_columns (list[str]): List of column names that contain text.
        list_columns (list[str]): List of column names that contain lists of text.
    Returns:
        df (pandas.DataFrame): The DataFrame with an added 'combined_text' column.
    """

    combined_text = df[text_columns].apply(" ".join, axis=1)
    for col in list_columns:
        combined_text += " " + df[col].str.join(" ")
    df[col_name] = combined_text
    return df


def add_count_columns(df: pd.DataFrame, list_columns: list[str]) -> None:
    """
    Adds count columns for specified list-like columns in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        list_columns (list[str]): List of column names that contain lists.
    """
    for col in list_columns:
        count_col = f"n_{col}"
        df[count_col] = df[col].apply(len)


# Technically this could be merged with add_count_columns, but I'm keeping them separate for clarity and future flexibility
def add_length_columns(df: pd.DataFrame, text_columns: list[str]) -> None:
    """
    Adds length columns for specified text columns in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        text_columns (list[str]): List of column names that contain text.
    """
    for col in text_columns:
        length_col = f"len_{col}"
        df[length_col] = df[col].apply(len)


def add_date_features(df: pd.DataFrame, date_column: str) -> None:
    """
    Adds date-related features to the DataFrame.
    Uses sine and cosine to capture the cyclical nature of days in a year.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        date_column (str): The name of the column containing datetime objects.
    """
    df["year"] = df[date_column].dt.year

    df["day_sin"] = np.sin(2 * np.pi * df[date_column].dt.dayofyear / 365.25)
    df["day_cos"] = np.cos(2 * np.pi * df[date_column].dt.dayofyear / 365.25)


def normalize_features(
    df: pd.DataFrame,
    columns: list[str],
    bounds: dict[str, tuple[int, int]] | None = None,
) -> dict[str, tuple[int, int]]:
    """
    Normalizes specified numerical columns in the DataFrame to a 0-1 range.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns (list[str]): List of column names to normalize.
        bounds (dict[str, tuple[int, int]]): A dictionary with min and max values for normalization.
            If not provided, min and max will be computed from the DataFrame.

    Returns:
        dict[str, tuple[int, int]]: The bounds used for normalization.
    """
    if bounds is None:
        bounds = {}
        for col in columns:
            min_val = cast(int, df[col].min())
            max_val = cast(int, df[col].max())
            bounds[col] = (min_val, max_val)
        debug(f"Computed normalization bounds: {bounds}")

    for col in columns:
        min_val, max_val = bounds[col]
        norm_col = f"norm_{col}"
        df[norm_col] = ((df[col] - min_val) / (max_val - min_val)).clip(0, 1)

    return bounds
