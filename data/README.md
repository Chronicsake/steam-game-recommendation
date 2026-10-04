# Dataset

The app reads **`data/steam.csv`** (path is relative to the project root, so nothing
needs to be configured).

The file is about 6 MB, so it is committed to the repository. That is what lets the app
run immediately after `git clone` and on Streamlit Community Cloud.

## Expected columns

The file has 18 columns. The pipeline requires these 13 (a clear error is shown if any
is missing):

`appid, name, release_date, developer, publisher, platforms, categories, genres,
steamspy_tags, positive_ratings, negative_ratings, owners, price`

The other columns (`english, required_age, achievements, average_playtime,
median_playtime`) are loaded but not used - see the main README for why.

## Using a different copy of the dataset

Replace `data/steam.csv` with another CSV that has the same column names. Multi-value
columns (`categories`, `genres`, `steamspy_tags`) must be `;`-separated, and `owners`
must be a range like `20000-50000`.

If your file is too large for GitHub (over ~50 MB), keep it out of the repo: add
`data/steam.csv` to `.gitignore`, and download it manually or in a startup script
instead.

## Source and licence

Add the link to the dataset page you downloaded this file from, and its licence, here
before publishing the repository.
