"""Feature engineering: turn every game into one numeric vector.

Each game vector is the concatenation of four weighted blocks:

    [ tags (TF-IDF) | genres (TF-IDF) | categories (TF-IDF) | numeric (scaled) ]

Because the blocks sit side by side in one vector, a single cosine similarity
compares games on all of them at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from src.config import FEATURE_WEIGHTS
from src.preprocessing import split_multi_value

# block name -> dataset column it is built from
TEXT_BLOCKS = {"tags": "steamspy_tags", "genres": "genres", "categories": "categories"}

# Numeric features (all rescaled to 0-1 before weighting).
NUMERIC_FEATURES = ("log_price", "positive_ratio", "log_reviews", "release_year")


@dataclass
class FeatureSpace:
    """The final feature matrix plus how many columns each block contributed."""
    matrix: csr_matrix
    block_sizes: Dict[str, int] = field(default_factory=dict)


def _tfidf_block(series: pd.Series, weight: float):
    """TF-IDF over ';'-separated items. Each tag/genre is one 'word'.

    TF-IDF gives rare items (e.g. 'Roguelike') more weight than items almost
    every game has (e.g. 'Indie'), so rare shared tags say more about similarity.
    """
    vectorizer = TfidfVectorizer(analyzer=split_multi_value)
    try:
        block = vectorizer.fit_transform(series)
    except ValueError:  # "empty vocabulary": this column has no items at all
        return csr_matrix((len(series), 0)), 0
    return block * weight, block.shape[1]


def _numeric_block(df: pd.DataFrame, weight: float):
    """Scaled numeric features.

    * price and review count are heavily right-skewed, so we take log1p first;
    * everything is MinMax-scaled to 0-1 so no feature dominates by its units.
    """
    numeric = pd.DataFrame({
        "log_price": np.log1p(df["price"]),
        "positive_ratio": df["positive_ratio"],
        "log_reviews": np.log1p(df["total_ratings"]),
        "release_year": df["release_year"],
    })[list(NUMERIC_FEATURES)]
    scaled = MinMaxScaler().fit_transform(numeric)
    return csr_matrix(scaled * weight), scaled.shape[1]


def build_feature_matrix(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> FeatureSpace:
    """Build the (n_games x n_features) sparse matrix used for similarity search."""
    weights = weights or FEATURE_WEIGHTS
    blocks, sizes = [], {}

    for block_name, column in TEXT_BLOCKS.items():
        block, size = _tfidf_block(df[column], weights[block_name])
        blocks.append(block)
        sizes[block_name] = size

    numeric, size = _numeric_block(df, weights["numeric"])
    blocks.append(numeric)
    sizes["numeric"] = size

    matrix = hstack(blocks, format="csr")
    if not np.isfinite(matrix.data).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return FeatureSpace(matrix=matrix, block_sizes=sizes)
