
import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------
# TMDB Configuration
# ---------------------------------------
API_KEY = "2e49de1c0d1a08f2711c8a3e882252fc"

IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500/"
FALLBACK_POSTER = (
    "https://placehold.co/500x750?text=Poster+Not+Available"
)

# ---------------------------------------
# TMDB API Request Helper
# ---------------------------------------
def tmdb_get(endpoint, params=None):
    if params is None:
        params = {}

    params["api_key"] = API_KEY

    response = requests.get(
        f"https://api.themoviedb.org/3/{endpoint}",
        params=params,
        timeout=12
    )

    response.raise_for_status()
    return response.json()


# ---------------------------------------
# Search Poster by Movie Title
# ---------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def search_poster_by_title(movie_title):
    if not movie_title or not str(movie_title).strip():
        return None

    try:
        data = tmdb_get(
            "search/movie",
            {
                "query": str(movie_title).strip(),
                "include_adult": "true",
                "language": "en-US",
                "page": 1
            }
        )

        results = data.get("results", [])

        if not results:
            return None

        # Exact title match ko preference do
        target = str(movie_title).strip().casefold()

        exact_matches = [
            item for item in results
            if item.get("title", "").strip().casefold() == target
            and item.get("poster_path")
        ]

        if exact_matches:
            return IMAGE_BASE_URL + exact_matches[0]["poster_path"]

        # Otherwise, closest title with an available poster
        poster_results = [
            item for item in results
            if item.get("poster_path")
        ]

        if not poster_results:
            return None

        best_match = max(
            poster_results,
            key=lambda item: SequenceMatcher(
                None,
                target,
                item.get("title", "").casefold()
            ).ratio()
        )

        return IMAGE_BASE_URL + best_match["poster_path"]

    except (requests.RequestException, ValueError, TypeError):
        return None


# ---------------------------------------
# Fetch Poster by TMDB Movie ID
# ---------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster(movie_id, movie_title):
    # First try the movie ID
    try:
        if pd.notna(movie_id):
            movie_id = int(movie_id)

            if movie_id > 0:
                data = tmdb_get(
                    f"movie/{movie_id}",
                    {"language": "en-US"}
                )

                poster_path = data.get("poster_path")

                if poster_path:
                    return IMAGE_BASE_URL + poster_path

    except (
        requests.RequestException,
        ValueError,
        TypeError
    ):
        pass

    # If ID failed, search by title
    poster = search_poster_by_title(movie_title)

    if poster:
        return poster

    return FALLBACK_POSTER


# ---------------------------------------
# Load Pickle Files
# ---------------------------------------
@st.cache_resource
def load_data():
    with open("movie_dict.pkl", "rb") as file:
        movies_data = pickle.load(file)

    with open("similarity.pkl", "rb") as file:
        similarity_data = pickle.load(file)

    if isinstance(movies_data, dict):
        movies_data = pd.DataFrame(movies_data)

    if not isinstance(movies_data, pd.DataFrame):
        raise TypeError(
            "movie_dict.pkl mein DataFrame nahi mila."
        )

    movies_data.columns = [
        str(col).strip() for col in movies_data.columns
    ]

    required_columns = ["title", "movie_id"]

    for column in required_columns:
        if column not in movies_data.columns:
            raise KeyError(
                f"'{column}' column nahi mila. "
                f"Available columns: {movies_data.columns.tolist()}"
            )

    similarity_data = np.asarray(similarity_data)

    if similarity_data.ndim != 2:
        raise ValueError(
            "similarity.pkl mein 2D matrix honi chahiye."
        )

    if similarity_data.shape != (
        len(movies_data),
        len(movies_data)
    ):
        raise ValueError(
            "Movies aur similarity matrix ka size match nahi karta. "
            "Dono pickle files same dataset se dobara banao."
        )

    return movies_data, similarity_data


# ---------------------------------------
# Recommendation Function
# ---------------------------------------
def recommend(movie, movies, similarity):
    titles = (
        movies["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    matches = np.flatnonzero(
        titles.to_numpy() == str(movie).strip()
    )

    if len(matches) == 0:
        return []

    movie_index = int(matches[0])
    scores = similarity[movie_index]

    # Highest similarity first
    sorted_indices = np.argsort(scores)[::-1]

    recommendations = []

    for index in sorted_indices:
        index = int(index)

        if index == movie_index:
            continue

        row = movies.iloc[index]
        title = str(row["title"]).strip()

        if not title:
            continue

        recommendations.append({
            "title": title,
            "movie_id": row["movie_id"]
        })

        if len(recommendations) >= 5:
            break

    return recommendations


# ---------------------------------------
# Main Application
# ---------------------------------------
st.title("🎬 Movie Recommender System")

st.write(
    "Select your favourite movie and discover "
    "five similar movies."
)

try:
    movies, similarity = load_data()

    movie_titles = sorted(
        movies["title"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda series: series != ""]
        .unique()
    )

    if not movie_titles:
        st.error("Dataset mein movie titles nahi hain.")
        st.stop()

    selected_movie = st.selectbox(
        "🎥 Type or select a movie",
        movie_titles
    )

    if st.button("Show Recommendation", type="primary"):

        recommendations = recommend(
            selected_movie,
            movies,
            similarity
        )

        if not recommendations:
            st.warning("Koi recommendation nahi mili.")
            st.stop()

        st.subheader(
            f"Recommended movies for {selected_movie}"
        )

        columns = st.columns(len(recommendations))

        for column, movie in zip(columns, recommendations):
            with column:
                with st.spinner("Loading poster..."):
                    poster = fetch_poster(
                        movie["movie_id"],
                        movie["title"]
                    )

                st.image(
                    poster,
                    use_container_width=True
                )
                st.caption(movie["title"])

                if poster == FALLBACK_POSTER:
                    st.caption(
                        "TMDB par poster nahi mila. "
                        "Movie ID/title verify karein."
                    )

except FileNotFoundError as error:
    st.error(f"Pickle file nahi mili: {error.filename}")
    st.info(
        "movie_dict.pkl aur similarity.pkl ko "
        "app.py ke same folder mein rakhein."
    )

except Exception as error:
    st.error(f"Application error: {error}")
    st.exception(error)
