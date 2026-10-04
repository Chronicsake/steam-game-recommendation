"""Central configuration: paths, required columns and model weights.

All paths are relative to the project root (found from this file's location),
so the project runs the same on any machine after cloning.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "steam.csv"

# Columns the pipeline needs. If any is missing, loading fails with a clear message.
REQUIRED_COLUMNS = [
    "appid", "name", "release_date", "developer", "publisher", "platforms",
    "categories", "genres", "steamspy_tags",
    "positive_ratings", "negative_ratings", "owners", "price",
]

# How much each feature group contributes to the final game vector.
# Tags are the most descriptive signal, so they get the highest weight;
# categories (e.g. "Steam Cloud", "Steam Trading Cards") describe store
# features rather than gameplay, so they get the lowest weight among text groups.
FEATURE_WEIGHTS = {
    "tags": 1.0,
    "genres": 0.7,
    "categories": 0.4,
    "numeric": 0.25,   # applied to each scaled numeric feature (price, rating, ...)
}

# Pseudo-count used to smooth the positive-review ratio of games with few reviews.
RATING_SMOOTHING = 10

DEFAULT_TOP_N = 5

# The dataset does not state a currency, so none is shown by default.
# Set e.g. "£" or "$" here once you have confirmed the currency of your data source.
CURRENCY_SYMBOL = ""
