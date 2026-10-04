"""The recommendation engine: cosine similarity between game vectors.

Quick demo from the command line (run from the project root):

    python -m src.recommender "Portal 2"
    python -m src.recommender "Portal 2" --min-reviews 500
"""
from __future__ import annotations

import argparse
import difflib
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DATA_PATH, DEFAULT_TOP_N
from src.data_loader import DataLoadError, load_raw_data
from src.features import FeatureSpace, build_feature_matrix
from src.preprocessing import clean_data


class GameNotFoundError(ValueError):
    """Raised when a requested game title does not exist in the dataset."""

    def __init__(self, query: str, suggestions: List[str]):
        self.query = query
        self.suggestions = suggestions
        super().__init__(f"Game '{query}' was not found in the dataset.")


class GameRecommender:
    """Recommends games whose feature vectors are closest (by cosine) to a chosen game."""

    def __init__(self, games: pd.DataFrame, space: FeatureSpace):
        if len(games) != space.matrix.shape[0]:
            raise ValueError("Games table and feature matrix have different numbers of rows.")
        self.games = games.reset_index(drop=True)
        self.space = space
        self.matrix = space.matrix
        # lower-cased title -> row number, for fast case-insensitive lookup
        self._lower_names = self.games["display_name"].str.lower()
        self._lookup: Dict[str, int] = {name: i for i, name in enumerate(self._lower_names)}
        self._reviews = self.games["total_ratings"].to_numpy()

    # ------------------------------------------------------------------ search
    def search(self, query: str, limit: int = 50) -> List[str]:
        """Titles containing `query` (case-insensitive).

        Titles that START with the query come first, then more-reviewed games first,
        so well-known games appear at the top. An empty query returns the most
        popular games.
        """
        q = (query or "").strip().lower()
        contains = self._lower_names.str.contains(q, regex=False)
        starts = self._lower_names.str.startswith(q)
        hits = self.games.loc[contains, ["display_name", "total_ratings"]].copy()
        hits["starts"] = starts[contains]
        hits = hits.sort_values(["starts", "total_ratings"], ascending=[False, False])
        return hits["display_name"].head(limit).tolist()

    def suggest(self, query: str, n: int = 5) -> List[str]:
        """Closest titles to a mistyped query (used for 'Did you mean ...?')."""
        q = (query or "").strip().lower()
        if not q:
            return []
        substring_matches = self.search(q, limit=n)
        if substring_matches:
            return substring_matches
        close = difflib.get_close_matches(q, self._lower_names.tolist(), n=n, cutoff=0.6)
        return [self.games["display_name"].iloc[self._lookup[c]] for c in close]

    def find_index(self, title: str) -> int:
        """Row number of an exact title (case-insensitive) or raise GameNotFoundError."""
        key = (title or "").strip().lower()
        if key in self._lookup:
            return self._lookup[key]
        raise GameNotFoundError(title, self.suggest(title))

    # ------------------------------------------------------------- game details
    def _record(self, i: int) -> dict:
        row = self.games.iloc[i]
        return {
            "appid": int(row["appid"]),
            "name": row["display_name"],
            "genres": row["genres_list"],
            "tags": row["tags_list"],
            "price": float(row["price"]),
            "positive_pct": None if pd.isna(row["positive_pct"]) else float(row["positive_pct"]),
            "total_ratings": int(row["total_ratings"]),
            "release_year": int(row["release_year"]),
            "developer": row["developer"],
            "publisher": row["publisher"],
            "platforms": row["platforms"],
            "owners_low": int(row["owners_low"]),
            "owners_high": int(row["owners_high"]),
        }

    def describe_game(self, title: str) -> dict:
        """Details of a single game, as a dict."""
        return self._record(self.find_index(title))

    # ---------------------------------------------------------- recommendations
    def recommend(self, title: str, top_n: int = DEFAULT_TOP_N, min_reviews: int = 0) -> pd.DataFrame:
        """Return the `top_n` games most similar to `title`.

        Steps:
          1. find the row of the chosen game;
          2. cosine similarity between its vector and every game vector;
          3. drop the game itself and games with fewer than `min_reviews` reviews;
          4. sort by similarity (ties broken by number of reviews) and keep the top N.
        """
        if not isinstance(top_n, (int, np.integer)) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")

        idx = self.find_index(title)

        # cosine_similarity(1 x F, N x F) -> (1 x N) array of similarities.
        sims = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        sims = np.clip(sims, 0.0, 1.0)  # remove tiny floating-point overshoot

        eligible = self._reviews >= min_reviews
        eligible[idx] = False  # never recommend the game itself

        # Round before sorting so floating-point noise doesn't break exact ties;
        # tied games are then ordered by popularity (more reviews first).
        scores = np.where(eligible, np.round(sims, 6), -np.inf)
        order = np.lexsort((-self._reviews, -scores))  # primary key: -scores
        order = order[: min(top_n, int(eligible.sum()))]

        query_traits = set(self.games["genres_list"].iloc[idx]) | set(self.games["tags_list"].iloc[idx])
        records = []
        for rank, i in enumerate(order, start=1):
            rec = self._record(int(i))
            traits = set(rec["genres"]) | set(rec["tags"])
            rec["rank"] = rank
            rec["similarity"] = float(sims[i])
            rec["similarity_pct"] = float(sims[i]) * 100
            rec["shared_features"] = sorted(query_traits & traits)
            records.append(rec)

        columns = ["rank", "appid", "name", "similarity", "similarity_pct", "genres", "tags",
                   "price", "positive_pct", "total_ratings", "release_year", "developer",
                   "publisher", "platforms", "owners_low", "owners_high", "shared_features"]
        return pd.DataFrame.from_records(records, columns=columns)


def build_recommender(path=DATA_PATH) -> Tuple[GameRecommender, Dict[str, int]]:
    """Full pipeline: load -> clean -> engineer features -> recommender."""
    raw = load_raw_data(path)
    games, report = clean_data(raw)
    space = build_feature_matrix(games)
    return GameRecommender(games, space), report


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Find games similar to a given Steam game.")
    parser.add_argument("title", help="game title, e.g. 'Portal 2'")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="how many games to return")
    parser.add_argument("--min-reviews", type=int, default=0, help="ignore games with fewer reviews")
    args = parser.parse_args(argv)

    try:
        recommender, _ = build_recommender()
        results = recommender.recommend(args.title, args.top, args.min_reviews)
    except DataLoadError as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1
    except GameNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if exc.suggestions:
            print("Did you mean: " + "; ".join(exc.suggestions), file=sys.stderr)
        return 1

    print(f"\nGames similar to '{args.title}':\n")
    for _, r in results.iterrows():
        print(f"{r['rank']}. {r['name']}  -  {r['similarity_pct']:.1f}% similar")
        print(f"   genres: {', '.join(r['genres'])} | tags: {', '.join(r['tags'])}")
        print(f"   shared: {', '.join(r['shared_features']) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
