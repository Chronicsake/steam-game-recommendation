"""Tests for the full pipeline. They run on the REAL dataset in data/steam.csv.

Run from the project root:
    python -m unittest discover -s tests -t . -v
"""
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import eda
from src.config import DATA_PATH, REQUIRED_COLUMNS
from src.data_loader import DataLoadError, load_raw_data
from src.preprocessing import clean_data, split_multi_value
from src.recommender import GameNotFoundError, build_recommender

_CACHE = {}


def get_system():
    """Build the recommender once and share it across all tests (it takes ~1 s)."""
    if "system" not in _CACHE:
        _CACHE["system"] = build_recommender()
    return _CACHE["system"]


class TestDataLoader(unittest.TestCase):
    def test_real_dataset_loads_with_required_columns(self):
        df = load_raw_data(DATA_PATH)
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, df.columns)
        self.assertGreater(len(df), 1000)

    def test_missing_file_gives_clear_error(self):
        with self.assertRaises(DataLoadError) as ctx:
            load_raw_data("this/file/does/not/exist.csv")
        self.assertIn("not found", str(ctx.exception))

    def test_missing_column_is_reported_by_name(self):
        raw = load_raw_data(DATA_PATH).drop(columns=["genres"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.csv"
            raw.to_csv(path, index=False)
            with self.assertRaises(DataLoadError) as ctx:
                load_raw_data(path)
        self.assertIn("genres", str(ctx.exception))

    def test_empty_file_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("")
            with self.assertRaises(DataLoadError):
                load_raw_data(path)


class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw_data(DATA_PATH)
        cls.clean, cls.report = clean_data(cls.raw)

    def test_split_multi_value(self):
        self.assertEqual(split_multi_value("Action; RPG;;Action"), ["Action", "RPG"])
        self.assertEqual(split_multi_value("Open World;Co-op"), ["Open World", "Co-op"])
        self.assertEqual(split_multi_value(None), [])
        self.assertEqual(split_multi_value(float("nan")), [])

    def test_no_duplicate_appids(self):
        self.assertFalse(self.clean["appid"].duplicated().any())

    def test_display_names_unique_case_insensitive(self):
        self.assertTrue(self.clean["display_name"].str.lower().is_unique)

    def test_names_are_trimmed(self):
        self.assertTrue((self.clean["name"] == self.clean["name"].str.strip()).all())

    def test_no_missing_values_in_model_columns(self):
        for col in ["price", "positive_ratio", "total_ratings", "release_year",
                    "developer", "publisher", "platforms"]:
            self.assertFalse(self.clean[col].isna().any(), col)

    def test_value_ranges(self):
        self.assertTrue((self.clean["price"] >= 0).all())
        self.assertTrue(self.clean["positive_ratio"].between(0, 1).all())
        self.assertTrue((self.clean["total_ratings"] >= 0).all())

    def test_report_is_consistent(self):
        self.assertEqual(self.report["rows_raw"], len(self.raw))
        self.assertEqual(self.report["rows_clean"], len(self.clean))
        removed = (self.report["rows_without_name_removed"]
                   + self.report["duplicate_rows_removed"]
                   + self.report["duplicate_appids_removed"])
        self.assertEqual(self.report["rows_raw"] - removed, self.report["rows_clean"])

    def test_cleaning_does_not_modify_input(self):
        raw_copy = self.raw.copy()
        clean_data(self.raw)
        pd.testing.assert_frame_equal(self.raw, raw_copy)


class TestRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec, _ = get_system()
        # Most-reviewed game: a real title picked from the data, not hard-coded.
        cls.popular = cls.rec.games.sort_values("total_ratings").iloc[-1]["display_name"]

    def test_feature_matrix_shape_and_values(self):
        m = self.rec.matrix
        self.assertEqual(m.shape[0], len(self.rec.games))
        self.assertTrue(np.isfinite(m.data).all())
        self.assertGreater(m.shape[1], 0)

    def test_returns_exactly_five(self):
        self.assertEqual(len(self.rec.recommend(self.popular)), 5)

    def test_never_recommends_itself(self):
        self.assertNotIn(self.popular, self.rec.recommend(self.popular, 20)["name"].tolist())

    def test_scores_sorted_and_in_range(self):
        sims = self.rec.recommend(self.popular, 20)["similarity"].to_numpy()
        self.assertTrue((np.diff(sims) <= 1e-6).all())  # descending
        self.assertTrue(((sims >= 0) & (sims <= 1)).all())

    def test_similarity_matches_manual_cosine(self):
        """Independent check: recompute cosine for the top result with plain numpy."""
        top = self.rec.recommend(self.popular, 1).iloc[0]
        a = self.rec.matrix[self.rec.find_index(self.popular)].toarray().ravel()
        b = self.rec.matrix[self.rec.find_index(top["name"])].toarray().ravel()
        manual = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        self.assertAlmostEqual(top["similarity"], manual, places=6)

    def test_min_reviews_filter(self):
        recs = self.rec.recommend(self.popular, 5, min_reviews=1000)
        self.assertTrue((recs["total_ratings"] >= 1000).all())

    def test_filter_that_excludes_everything_returns_empty(self):
        recs = self.rec.recommend(self.popular, 5, min_reviews=10 ** 12)
        self.assertTrue(recs.empty)

    def test_lookup_is_case_insensitive(self):
        a = self.rec.recommend(self.popular.upper(), 5)["name"].tolist()
        b = self.rec.recommend(self.popular.lower(), 5)["name"].tolist()
        self.assertEqual(a, b)

    def test_deterministic(self):
        a = self.rec.recommend(self.popular, 5)["name"].tolist()
        b = self.rec.recommend(self.popular, 5)["name"].tolist()
        self.assertEqual(a, b)

    def test_unknown_game_raises_with_message(self):
        with self.assertRaises(GameNotFoundError) as ctx:
            self.rec.recommend("zzzz definitely not a steam game 98765")
        self.assertIn("not found", str(ctx.exception))

    def test_empty_title_raises(self):
        with self.assertRaises(GameNotFoundError):
            self.rec.recommend("   ")

    def test_invalid_top_n_raises(self):
        for bad in (0, -3, 2.5, "5"):
            with self.assertRaises(ValueError):
                self.rec.recommend(self.popular, bad)

    def test_search_results_contain_query_and_popular_first(self):
        hits = self.rec.search("the")
        self.assertTrue(all("the" in h.lower() for h in hits))
        self.assertLessEqual(len(hits), 50)
        self.assertEqual(self.rec.search("")[0], self.popular)

    def test_search_no_match_returns_empty_list(self):
        self.assertEqual(self.rec.search("zzzz definitely not a steam game 98765"), [])

    def test_recommendations_are_content_related(self):
        """On a sample of real games, recommendations should share genres/tags with the query."""
        sample = self.rec.games[self.rec.games["total_ratings"] >= 100].sample(40, random_state=1)
        overlaps = [len(self.rec.recommend(t, 5)["shared_features"].iloc[0])
                    for t in sample["display_name"]]
        self.assertGreaterEqual(np.mean([o >= 1 for o in overlaps]), 0.9)


class TestEDA(unittest.TestCase):
    def test_summary_stats(self):
        rec, _ = get_system()
        stats = eda.summary_stats(rec.games)
        self.assertEqual(stats["n_games"], len(rec.games))
        self.assertGreater(stats["n_genres"], 0)
        self.assertTrue(0 <= stats["pct_free"] <= 100)

    def test_every_chart_renders(self):
        rec, _ = get_system()
        for name, plot in eda.ALL_CHARTS.items():
            fig = plot(rec.games)
            self.assertTrue(hasattr(fig, "savefig"), name)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
