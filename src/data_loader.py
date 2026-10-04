"""Loading the raw CSV and validating that it has the columns we need."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, REQUIRED_COLUMNS


class DataLoadError(Exception):
    """Raised when the dataset is missing, unreadable or has the wrong columns."""


def load_raw_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the Steam CSV and check it contains every required column."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(
            f"Dataset not found at '{path}'. Place the Steam CSV there "
            "(see data/README.md for details)."
        )
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Could not read '{path.name}' as a CSV file: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise DataLoadError(f"'{path.name}' contains no rows.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(
            f"'{path.name}' is missing required column(s): {', '.join(missing)}. "
            f"Columns found: {', '.join(df.columns)}"
        )
    return df
