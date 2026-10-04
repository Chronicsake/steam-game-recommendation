"""Exploratory data analysis: summary numbers and matplotlib charts.

Every plot function takes the CLEANED DataFrame (see preprocessing.clean_data)
and returns a matplotlib Figure, so the same code feeds the Streamlit app, the
README images and the unit tests.
"""
from __future__ import annotations

from typing import Dict

import matplotlib

matplotlib.use("Agg")  # no display needed (works on servers and in tests)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PRIMARY = "#1b6ac9"
ACCENT = "#66c0f4"
DARK = "#1b2838"
MIN_REVIEWS_FOR_RATING_CHART = 50


def _new_axes(title: str, xlabel: str = "", ylabel: str = "", size=(7, 4.2)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, fontsize=12, fontweight="bold", color=DARK, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def _top_items(list_column: pd.Series, n: int) -> pd.Series:
    """Count how many games have each item of a list column (genres / tags)."""
    return list_column.explode().dropna().value_counts().head(n)


# --------------------------------------------------------------------- numbers
def summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Headline numbers shown at the top of the analysis page."""
    paid = df[df["price"] > 0]
    rated = df[df["total_ratings"] >= MIN_REVIEWS_FOR_RATING_CHART]
    return {
        "n_games": len(df),
        "n_genres": int(df["genres_list"].explode().nunique()),
        "n_tags": int(df["tags_list"].explode().nunique()),
        "pct_free": float((df["price"] == 0).mean() * 100),
        "median_paid_price": float(paid["price"].median()) if len(paid) else 0.0,
        "avg_positive_pct": float(rated["positive_pct"].mean()) if len(rated) else float("nan"),
        "total_reviews": int(df["total_ratings"].sum()),
        "year_min": int(df["release_year"].min()),
        "year_max": int(df["release_year"].max()),
    }


# ---------------------------------------------------------------------- charts
def plot_top_genres(df: pd.DataFrame, n: int = 15):
    counts = _top_items(df["genres_list"], n).sort_values()
    fig, ax = _new_axes(f"Top {n} genres", "Number of games")
    ax.barh(counts.index, counts.values, color=PRIMARY)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(counts.values):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_top_tags(df: pd.DataFrame, n: int = 15):
    counts = _top_items(df["tags_list"], n).sort_values()
    fig, ax = _new_axes(f"Top {n} SteamSpy tags", "Number of games")
    ax.barh(counts.index, counts.values, color=ACCENT)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(counts.values):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_price_distribution(df: pd.DataFrame, cap: float = 60.0):
    paid = df[df["price"] > 0]["price"]
    n_free = int((df["price"] == 0).sum())
    shown = paid[paid <= cap]
    fig, ax = _new_axes(
        f"Price distribution of paid games (up to {cap:.0f})",
        "Price", "Number of games",
    )
    ax.hist(shown, bins=60, color=PRIMARY, edgecolor="white")
    if len(paid):
        ax.axvline(paid.median(), color="crimson", linestyle="--", linewidth=1.2)
        ax.text(paid.median(), ax.get_ylim()[1] * 0.92, f"  median {paid.median():.2f}",
                color="crimson", fontsize=9)
    ax.text(0.98, 0.78, f"{n_free:,} free games\n({n_free / len(df):.1%}) not shown\n"
            f"{len(paid) - len(shown):,} paid games above {cap:.0f} not shown",
            transform=ax.transAxes, ha="right", fontsize=8, color=DARK)
    fig.tight_layout()
    return fig


def plot_rating_distribution(df: pd.DataFrame, min_reviews: int = MIN_REVIEWS_FOR_RATING_CHART):
    rated = df[df["total_ratings"] >= min_reviews]
    fig, ax = _new_axes(
        f"Positive-review percentage (games with at least {min_reviews} reviews)",
        "% positive reviews", "Number of games",
    )
    ax.hist(rated["positive_pct"], bins=25, color=PRIMARY, edgecolor="white")
    ax.text(0.02, 0.92, f"{len(rated):,} games shown", transform=ax.transAxes, fontsize=9, color=DARK)
    fig.tight_layout()
    return fig


def plot_review_counts(df: pd.DataFrame):
    values = np.log10(df["total_ratings"] + 1)
    fig, ax = _new_axes("How many reviews do games get? (log scale)",
                        "log10(number of reviews + 1)", "Number of games")
    ax.hist(values, bins=40, color=ACCENT, edgecolor="white")
    median = df["total_ratings"].median()
    ax.axvline(np.log10(median + 1), color="crimson", linestyle="--", linewidth=1.2)
    ax.text(np.log10(median + 1), ax.get_ylim()[1] * 0.92, f"  median {median:,.0f} reviews",
            color="crimson", fontsize=9)
    fig.tight_layout()
    return fig


def plot_games_per_year(df: pd.DataFrame):
    per_year = df["release_year"].value_counts().sort_index()
    fig, ax = _new_axes("Games released per year", "Release year", "Number of games")
    ax.bar(per_year.index.astype(int), per_year.values, color=PRIMARY)
    fig.tight_layout()
    return fig


def plot_owners(df: pd.DataFrame):
    """Estimated owners come as ranges (e.g. 20000-50000); show games per range."""
    grouped = (df.groupby(["owners_low", "owners_high"]).size().reset_index(name="games")
               .sort_values("owners_low"))
    labels = [f"{int(lo):,}-{int(hi):,}" for lo, hi in zip(grouped["owners_low"], grouped["owners_high"])]
    fig, ax = _new_axes("Estimated owners per game (range)", "Number of games (log scale)",
                        size=(7, 4.6))
    ax.barh(labels, grouped["games"], color=ACCENT)
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(grouped["games"]):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_price_by_genre(df: pd.DataFrame, n: int = 12):
    """Median price of PAID games for the n most common genres."""
    top = _top_items(df["genres_list"], n).index
    exploded = df[["genres_list", "price"]].explode("genres_list")
    paid = exploded[(exploded["price"] > 0) & (exploded["genres_list"].isin(top))]
    medians = paid.groupby("genres_list")["price"].median().sort_values()
    fig, ax = _new_axes(f"Median price of paid games, top {n} genres", "Median price")
    ax.barh(medians.index, medians.values, color=PRIMARY)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(medians.values):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


ALL_CHARTS = {
    "Top genres": plot_top_genres,
    "Top tags": plot_top_tags,
    "Price distribution": plot_price_distribution,
    "Positive-review percentage": plot_rating_distribution,
    "Review counts": plot_review_counts,
    "Games released per year": plot_games_per_year,
    "Estimated owners": plot_owners,
    "Median price by genre": plot_price_by_genre,
}
