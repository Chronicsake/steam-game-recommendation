"""A simple sanity check for the recommender (not a full accuracy benchmark).

The dataset has no user ratings, so we cannot compute classic recommender
accuracy. Instead we test a signal the model was NOT given: the developer.
If the model captures real game similarity, its top-5 should contain games by
the same developer more often than 5 randomly chosen games would.

Run from the project root:   python -m src.evaluation
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.recommender import GameRecommender, build_recommender


def same_developer_rate(recommender: GameRecommender, sample_size: int = 300,
                        top_n: int = 5, seed: int = 42) -> Dict[str, float]:
    """Share of recommended games made by the query game's developer, vs random picks."""
    games = recommender.games
    counts = games["developer"].value_counts()
    multi_game_devs = counts[counts >= 2].index
    # Only games whose developer has at least one other game (otherwise a match is impossible).
    pool = games[games["developer"].isin(multi_game_devs) & (games["developer"] != "Unknown")]
    sample = pool.sample(min(sample_size, len(pool)), random_state=seed)

    rng = np.random.default_rng(seed)
    model_hits = random_hits = total = 0
    for title, developer in zip(sample["display_name"], sample["developer"]):
        recs = recommender.recommend(title, top_n)
        model_hits += int((recs["developer"] == developer).sum())
        random_idx = rng.choice(len(games), size=top_n, replace=False)
        random_hits += int((games["developer"].iloc[random_idx] == developer).sum())
        total += top_n

    return {
        "games_tested": len(sample),
        "model_rate": model_hits / total,
        "random_rate": random_hits / total,
    }


if __name__ == "__main__":
    rec, _ = build_recommender()
    result = same_developer_rate(rec)
    lift = result["model_rate"] / result["random_rate"] if result["random_rate"] else float("inf")
    print(f"Games tested                : {result['games_tested']}")
    print(f"Same-developer rate (model) : {result['model_rate']:.2%}")
    print(f"Same-developer rate (random): {result['random_rate']:.2%}")
    print(f"Model is {lift:.0f}x better than random")
