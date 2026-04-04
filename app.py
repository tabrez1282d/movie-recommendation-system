import streamlit as st
import pickle
import pandas as pd
import requests
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from thefuzz import process


# Your TMDB API key
import os
from dotenv import load_dotenv
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()

sparse_matrix = data['sparse_matrix']
movie_index = data['movie_index']
movie_to_idx = data['movie_to_idx']
title_to_movie_id = data['title_to_movie_id']
movie_id_to_idx = data['movie_id_to_idx']
tfidf_matrix = data['tfidf_matrix']
movies = data['movies']

# Helper — get genre of a movie
def get_genre(title):
    if title in title_to_movie_id:
        mid = title_to_movie_id[title]
        row = movies[movies['movieId'] == mid]
        if not row.empty:
            return row.iloc[0]['genres'].replace('|', ' | ')
    return ""

# Helper — extract year from title
def extract_year(title):
    try:
        year = title[title.rfind('(')+1:title.rfind(')')]
        if year.isdigit():
            return year
    except:
        pass
    return ""

# Helper — clean title (remove year)
def clean_title(title):
    if '(' in title:
        return title[:title.rfind('(')].strip()
    return title

# Fetch poster from TMDB
def fetch_poster(movie_title):
    try:
        clean = clean_title(movie_title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w300{poster_path}"
    except:
        pass
    return None

# Functions
def find_movie_name(user_input):
    user_input = user_input.lower()
    exact = [t for t in movie_to_idx.keys() if user_input in t.lower()]
    if exact:
        return exact
    fuzzy = process.extract(user_input, movie_to_idx.keys(), limit=5)
    return [m[0] for m in fuzzy if m[1] > 50]

def hybrid_recommend(movie_name, n=5, rating_weight=0.7, genre_weight=0.3,):
    if movie_name not in movie_to_idx:
        return pd.DataFrame()

    idx = movie_to_idx[movie_name]
    movie_vector = sparse_matrix.T[idx]
    rating_scores = cosine_similarity(movie_vector, sparse_matrix.T).flatten()

    if movie_name in title_to_movie_id:
        movie_id = title_to_movie_id[movie_name]
        genre_idx = movie_id_to_idx[movie_id]
        genre_scores_raw = linear_kernel(tfidf_matrix[genre_idx], tfidf_matrix).flatten()
    else:
        genre_scores_raw = [0] * tfidf_matrix.shape[0]

    similar_indices = rating_scores.argsort()[::-1][1:]

    results = []
    for i in similar_indices:
        title = movie_index[i]
        genre = get_genre(title)


        r_score = rating_scores[i]
        g_score = 0
        if title in title_to_movie_id:
            mid = title_to_movie_id[title]
            gidx = movie_id_to_idx[mid]
            g_score = genre_scores_raw[gidx]
        final_score = (rating_weight * r_score) + (genre_weight * g_score)
        results.append({
            'title': title,
            'clean_title': clean_title(title),
            'year': extract_year(title),
            'genre': genre,
            'final_score': round(final_score, 3),
            'rating_sim': round(r_score, 3),
            'genre_sim': round(g_score, 3)
        })

        if len(results) == n:
            break

    return pd.DataFrame(results)

# Helper to show movie grid
def show_movie_grid(results):
    cols = st.columns(5)
    for i, row in results.iterrows():
        with cols[i % 5]:
            poster = fetch_poster(row['title'])
            if poster:
                st.image(poster, width=130)
            else:
                st.markdown("🎬 *No Poster Available*")
            st.markdown(f"**{row['clean_title']}**")
            st.caption(f"📅 {row['year']}")
            st.caption(f"🎭 {row['genre']}")
            st.caption(f"⭐ Score: {row['final_score']}")
            st.caption(f"🎬 Rating: {row['rating_sim']}")
            st.caption(f"🎭 Genre: {row['genre_sim']}")


# UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorites!")


# Auto sliding carousel
st.subheader("🎲 Try searching for:")
random_movies = random.sample(list(movie_to_idx.keys()), 5)
cols = st.columns(5)
for i, movie in enumerate(random_movies):
    with cols[i]:
        poster = fetch_poster(movie)
        if poster:
            st.image(poster, width=130)
        st.caption(f"{clean_title(movie)} ({extract_year(movie)})")

st.markdown("---")

# Search
user_input = st.text_input("Enter a movie name:")

if user_input:
    matches = find_movie_name(user_input)

    if len(matches) == 0:
        st.error("No movies found. Try a different name.")

    elif len(matches) == 1:
        selected = matches[0]
        st.success(f"Showing recommendations for: **{selected}**")
        results = hybrid_recommend(selected,)

        st.subheader("Selected Movie:")
        poster = fetch_poster(selected)
        if poster:
            st.image(poster, width=150)
        else:
            st.markdown("🎬 *No Poster Available*")
        st.markdown(f"**{clean_title(selected)}**")
        st.caption(f"📅 {extract_year(selected)}")
        st.caption(f"🎭 {get_genre(selected)}")

        st.subheader("Recommended Movies:")
        show_movie_grid(results)

    else:
        st.write("Did you mean:")
        selected = st.selectbox("Select a movie:", matches[:10])
        if selected:
            st.success(f"Showing recommendations for: **{selected}**")
            results = hybrid_recommend(selected)

            st.subheader("Selected Movie:")
            poster = fetch_poster(selected)
            if poster:
                st.image(poster, width=150)
            else:
                st.markdown("🎬 *No Poster Available*")
            st.markdown(f"**{clean_title(selected)}**")
            st.caption(f"📅 {extract_year(selected)}")
            st.caption(f"🎭 {get_genre(selected)}")

            st.subheader("Recommended Movies:")
            show_movie_grid(results)