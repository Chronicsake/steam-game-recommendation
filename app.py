"""Streamlit GUI for the Steam Game Recommendation System.

Run locally:   streamlit run app.py
"""
import math

import matplotlib.pyplot as plt
import streamlit as st

from src import eda
from src.config import CURRENCY_SYMBOL, DEFAULT_TOP_N, FEATURE_WEIGHTS
from src.data_loader import DataLoadError
from src.features import NUMERIC_FEATURES
from src.recommender import GameNotFoundError, build_recommender

st.set_page_config(page_title="Steam Game Recommender", page_icon="🎮", layout="wide")


# ----------------------------------------------------------------- cached work
@st.cache_resource(show_spinner="Loading dataset and building the feature matrix...")
def load_system():
    """Load + clean the data and build the recommender ONCE per server session."""
    return build_recommender()


@st.cache_resource(show_spinner="Drawing charts...")
def load_charts(_games):
    """Draw every EDA chart once and reuse the figures on later reruns."""
    figures = {}
    for name, plot in eda.ALL_CHARTS.items():
        fig = plot(_games)
        plt.close(fig)  # keep the Figure object, but free pyplot's global registry
        figures[name] = fig
    return figures


# ------------------------------------------------------------ small formatters
def fmt_price(price: float) -> str:
    return "Free" if price == 0 else f"{CURRENCY_SYMBOL}{price:,.2f}"


def fmt_pct(value) -> str:
    return "N/A" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.0f}%"


def fmt_owners(low: int, high: int) -> str:
    return f"{low:,} - {high:,}"


# -------------------------------------------------------------------- results
def show_selected_game(info: dict) -> None:
    """Compact summary of the game the recommendations are based on."""
    with st.container(border=True):
        st.markdown(f"#### Based on: {info['name']}")
        st.caption(f"{info['developer']}  |  {info['release_year']}  |  "
                   f"{info['platforms'].replace(';', ', ')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", fmt_price(info["price"]))
        c2.metric("Positive reviews", fmt_pct(info["positive_pct"]))
        c3.metric("Reviews", f"{info['total_ratings']:,}")
        st.markdown("**Genres:** " + (", ".join(info["genres"]) or "-"))
        st.markdown("**Tags:** " + (", ".join(info["tags"]) or "-"))


def show_recommendation(row) -> None:
    """One recommendation card."""
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.subheader(f"{row['rank']}. {row['name']}")
            st.caption(f"{row['developer']}  |  {row['release_year']}  |  "
                       f"{row['platforms'].replace(';', ', ')}")
        with right:
            st.metric("Similarity", f"{row['similarity_pct']:.1f}%")
        st.progress(min(max(float(row["similarity"]), 0.0), 1.0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", fmt_price(row["price"]))
        c2.metric("Positive reviews", fmt_pct(row["positive_pct"]))
        c3.metric("Reviews", f"{row['total_ratings']:,}")
        c4.metric("Estimated owners", fmt_owners(row["owners_low"], row["owners_high"]))

        st.markdown("**Genres:** " + (", ".join(row["genres"]) or "-"))
        st.markdown("**Tags:** " + (", ".join(row["tags"]) or "-"))
        shared = ", ".join(row["shared_features"]) or "none (similarity comes from other features)"
        st.markdown(f"**Why it matches (shared genres/tags):** {shared}")
        st.markdown(f"[View on Steam](https://store.steampowered.com/app/{row['appid']})")


def render_results(recommender, request: dict) -> None:
    try:
        results = recommender.recommend(request["title"], DEFAULT_TOP_N, request["min_reviews"])
        info = recommender.describe_game(request["title"])
    except GameNotFoundError as exc:
        st.error(f"{exc} Please pick a game from the list.")
        return
    except ValueError as exc:
        st.error(f"Could not generate recommendations: {exc}")
        return

    show_selected_game(info)
    if results.empty:
        st.warning("No games match the current filter. Try lowering the minimum number of reviews.")
        return
    st.markdown(f"### Top {len(results)} similar games")
    for _, row in results.iterrows():
        show_recommendation(row)


# ------------------------------------------------------------------- the pages
def recommender_tab(recommender, min_reviews: int) -> None:
    st.markdown("Search for a Steam game, pick it from the list, and get the five most similar games.")

    query = st.text_input("Search for a game", key="query",
                          placeholder="e.g. Portal, Witcher, Civilization, Stardew")
    matches = recommender.search(query, limit=50)

    if not matches:
        # Clear error message when the game does not exist.
        st.error(f"No game found matching '{query}'. Check the spelling or try fewer words.")
        suggestions = recommender.suggest(query)
        if suggestions:
            st.info("Did you mean: " + " | ".join(suggestions))
        return

    label = "Select a game" if query.strip() else "Select a game (showing the most-reviewed games)"
    selected = st.selectbox(label, matches)
    if st.button("Recommend similar games", type="primary"):
        # Remember the request so results stay visible when other widgets change.
        st.session_state["request"] = {"title": selected, "min_reviews": min_reviews}

    request = st.session_state.get("request")
    if request:
        render_results(recommender, request)


def analysis_tab(recommender) -> None:
    games = recommender.games
    stats = eda.summary_stats(games)

    st.markdown("### Dataset overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games", f"{stats['n_games']:,}")
    c2.metric("Genres / tags", f"{stats['n_genres']} / {stats['n_tags']}")
    c3.metric("Free games", f"{stats['pct_free']:.1f}%")
    c4.metric("Median paid price", f"{CURRENCY_SYMBOL}{stats['median_paid_price']:.2f}")
    c5.metric("Avg positive reviews", f"{stats['avg_positive_pct']:.0f}%")
    st.caption(f"Games released {stats['year_min']}-{stats['year_max']}  |  "
               f"{stats['total_reviews']:,} reviews in total. "
               f"Average positive reviews uses games with at least "
               f"{eda.MIN_REVIEWS_FOR_RATING_CHART} reviews.")

    charts = load_charts(games)
    names = list(charts)
    for i in range(0, len(names), 2):
        cols = st.columns(2)
        for col, name in zip(cols, names[i:i + 2]):
            with col:
                st.pyplot(charts[name], clear_figure=False)


def about_tab(recommender, report: dict) -> None:
    sizes = recommender.space.block_sizes
    st.markdown(
        f"""
### How recommendations are calculated
1. **Clean** the Steam data (duplicates, missing values, inconsistent text).
2. **Vectorize** every game into one numeric vector:
   - **Tags** ({sizes['tags']} unique) - TF-IDF, weight **{FEATURE_WEIGHTS['tags']}**
   - **Genres** ({sizes['genres']} unique) - TF-IDF, weight **{FEATURE_WEIGHTS['genres']}**
   - **Categories** ({sizes['categories']} unique) - TF-IDF, weight **{FEATURE_WEIGHTS['categories']}**
   - **Numeric** ({', '.join(NUMERIC_FEATURES)}) - scaled to 0-1, weight **{FEATURE_WEIGHTS['numeric']}** each
3. **Cosine similarity** between the chosen game's vector and every other game's vector:
   `similarity = (A . B) / (|A| x |B|)` - 100% means identical direction, 0% means nothing in common.
4. **Rank** by similarity (ties broken by number of reviews) and show the top {DEFAULT_TOP_N}.

The similarity percentage is a *relative* measure of how alike two games' features are,
not a prediction of whether you will enjoy the game.
"""
    )
    st.markdown("### Data cleaning report")
    st.table({"Step": [k.replace("_", " ") for k in report], "Rows": [f"{v:,}" for v in report.values()]})


# ------------------------------------------------------------------------ main
def main() -> None:
    st.title("🎮 Steam Game Recommendation System")
    st.caption("Content-based recommendations using TF-IDF vectorization and cosine similarity")

    try:
        recommender, report = load_system()
    except DataLoadError as exc:
        st.error(f"Could not load the dataset: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        min_reviews = st.slider(
            "Minimum reviews for recommended games", 0, 1000, 10, step=10,
            help="Hides games with very few reviews. Applied when you click 'Recommend'.")
        st.divider()
        st.markdown(f"**{len(recommender.games):,}** games loaded")

    tab_rec, tab_eda, tab_about = st.tabs(["🎯 Recommender", "📊 Data Analysis", "ℹ️ How it works"])
    with tab_rec:
        recommender_tab(recommender, min_reviews)
    with tab_eda:
        analysis_tab(recommender)
    with tab_about:
        about_tab(recommender, report)


main()
