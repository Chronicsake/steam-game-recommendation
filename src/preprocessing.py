"""Data cleaning and derived columns for the Steam dataset.

clean_data() takes the raw DataFrame and returns (clean_df, report), where
`report` counts what was fixed so the cleaning is transparent and testable.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.config import RATING_SMOOTHING

TEXT_COLUMNS = ["name", "developer", "publisher", "platforms",
                "categories", "genres", "steamspy_tags"]


def split_multi_value(value) -> List[str]:
    """Split a ';'-separated field into clean, de-duplicated items.

    'Action; RPG;;Action' -> ['Action', 'RPG']. Non-strings give [].
    Multi-word tags such as 'Open World' stay as ONE item.
    """
    if not isinstance(value, str):
        return []
    items: List[str] = []
    for token in value.split(";"):
        token = token.strip()
        if token and token not in items:
            items.append(token)
    return items


def make_display_names(df: pd.DataFrame) -> pd.Series:
    """Build a unique, human-readable label for every game.

    The dataset contains different games that share a title (e.g. two unrelated
    games called 'Alone'). Instead of deleting them we disambiguate step by step:
    name -> name (year) -> name (year) - developer -> name (year) - developer #appid.
    """
    label = df["name"].copy()

    clash = label.str.lower().duplicated(keep=False)
    label = label.where(~clash, df["name"] + " (" + df["release_year"].astype(str) + ")")

    clash = label.str.lower().duplicated(keep=False)
    label = label.where(~clash, label + " - " + df["developer"])

    clash = label.str.lower().duplicated(keep=False)
    label = label.where(~clash, label + " #" + df["appid"].astype(str))
    return label


def clean_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean the raw data and add derived columns used by the recommender."""
    df = raw.copy()
    report: Dict[str, int] = {"rows_raw": len(df)}

    # 1. Text columns: missing -> "", trim stray whitespace (e.g. 'Exodus from the Earth ').
    original_names = df["name"].fillna("").astype(str)
    report["names_with_extra_whitespace"] = int((original_names != original_names.str.strip()).sum())
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # 2. Rows without a name cannot be searched or shown.
    blank_name = df["name"] == ""
    report["rows_without_name_removed"] = int(blank_name.sum())
    df = df[~blank_name].copy()

    # 3. Duplicates: identical rows, then repeated appids (appid is the unique game id).
    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_removed"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset="appid", keep="first").copy()
    report["duplicate_appids_removed"] = before - len(df)

    # 4. Missing developer / publisher / platforms -> "Unknown".
    for col in ("developer", "publisher", "platforms"):
        missing = df[col] == ""
        report[f"missing_{col}_filled"] = int(missing.sum())
        df[col] = df[col].where(~missing, "Unknown")

    # 5. Numeric columns: coerce to numbers, repair invalid values.
    for col in ("positive_ratings", "negative_ratings"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype(int)

    price = pd.to_numeric(df["price"], errors="coerce")
    report["missing_price_filled"] = int(price.isna().sum())
    fill_value = price.median() if price.notna().any() else 0.0
    df["price"] = price.fillna(fill_value).clip(lower=0).astype(float)

    # 6. Release date -> release year (invalid dates get the median year).
    dates = pd.to_datetime(df["release_date"], errors="coerce")
    report["invalid_release_dates"] = int(dates.isna().sum())
    year = dates.dt.year
    year = year.fillna(year.median() if year.notna().any() else 0)
    df["release_year"] = year.astype(int)

    # 7. 'owners' is a range string like '10000000-20000000' -> numeric low / high.
    parts = df["owners"].astype(str).str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    df["owners_low"] = pd.to_numeric(parts[0], errors="coerce").fillna(0)
    df["owners_high"] = pd.to_numeric(parts[1], errors="coerce").fillna(0)

    # 8. Rating features.
    #    total_ratings : number of reviews (popularity)
    #    positive_pct  : plain % of positive reviews, for display
    #    positive_ratio: smoothed ratio in [0,1] used as a model feature. A game with
    #                    2 reviews (both positive) should not look "perfect", so we
    #                    blend in the global average with a small pseudo-count.
    df["total_ratings"] = df["positive_ratings"] + df["negative_ratings"]
    total = df["total_ratings"].sum()
    global_ratio = df["positive_ratings"].sum() / total if total > 0 else 0.5
    df["positive_ratio"] = (
        (df["positive_ratings"] + RATING_SMOOTHING * global_ratio)
        / (df["total_ratings"] + RATING_SMOOTHING)
    )
    df["positive_pct"] = (100 * df["positive_ratings"] / df["total_ratings"]).where(df["total_ratings"] > 0)

    # 9. List versions of the multi-value columns (used for charts and explanations).
    df["genres_list"] = df["genres"].map(split_multi_value)
    df["tags_list"] = df["steamspy_tags"].map(split_multi_value)
    df["categories_list"] = df["categories"].map(split_multi_value)

    # 10. Unique display label per game (handles repeated titles).
    df = df.reset_index(drop=True)
    df["display_name"] = make_display_names(df)

    report["rows_clean"] = len(df)
    return df, report
