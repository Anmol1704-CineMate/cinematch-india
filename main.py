import os
import json
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity as skl_cosine
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── API Keys ──────────────────────────────────────────
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

# ── Firebase Setup ────────────────────────────────────
if not firebase_admin._apps:
    firebase_key_json = os.environ.get("FIREBASE_KEY")
    firebase_key_dict = json.loads(firebase_key_json)
    cred = credentials.Certificate(firebase_key_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ── Load Movies ───────────────────────────────────────
def load_movies():
    docs = db.collection("movies").stream()
    movies = {}
    for doc in docs:
        movies[doc.id] = doc.to_dict()
    return movies

my_movies = load_movies()

# ── Build Fingerprints ────────────────────────────────
all_genres = sorted(set(g for m in my_movies.values() for g in m.get("genres", [])))

def build_fingerprint(movie):
    genres = movie.get("genres", [])
    genre_vector = [1 if g in genres else 0 for g in all_genres]
    rating = movie.get("rating", 0)
    year = int(movie.get("year", 2000))
    min_r, max_r = 0, 10
    min_y, max_y = 1990, 2024
    norm_rating = (rating - min_r) / (max_r - min_r)
    norm_year = (year - min_y) / (max_y - min_y)
    return genre_vector + [norm_rating, norm_year]

movies_vector = {title: build_fingerprint(m) for title, m in my_movies.items()}

# ── ML Functions ──────────────────────────────────────
def build_user_profile(username, df_matrix, movies_vector):
    if username not in df_matrix.index:
        return None
    user_ratings = df_matrix.loc[username]
    rated_movies = user_ratings.dropna()
    if len(rated_movies) == 0:
        return None
    profile_vector = None
    for movie, rating in rated_movies.items():
        if movie in movies_vector:
            fingerprint = movies_vector[movie]
            weighted = [x * rating for x in fingerprint]
            if profile_vector is None:
                profile_vector = weighted
            else:
                profile_vector = [profile_vector[i] + weighted[i] for i in range(len(weighted))]
    profile_vector = [x / len(rated_movies) for x in profile_vector]
    return profile_vector

def recommend_for_user(username, profile, df_matrix, movies_vector, top_n=10):
    if profile is None:
        return []
    user_ratings = df_matrix.loc[username]
    already_rated = set(user_ratings.dropna().index)
    scores = []
    for movie, fingerprint in movies_vector.items():
        if movie in already_rated:
            continue
        dot_product = sum(profile[i] * fingerprint[i] for i in range(len(profile)))
        magnitude_profile = sum(x**2 for x in profile) ** 0.5
        magnitude_movie = sum(x**2 for x in fingerprint) ** 0.5
        if magnitude_profile == 0 or magnitude_movie == 0:
            similarity = 0
        else:
            similarity = dot_product / (magnitude_profile * magnitude_movie)
        scores.append((movie, similarity))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# ── FastAPI ───────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("FastAPI app created!")

# ── Pydantic Request Models (Compatibility) ───────────
class RatingInput(BaseModel):
    username: str = None
    user_id: str = None
    movie: str = None
    rating: int = None
    label: str = None

# ── Endpoints ─────────────────────────────────────────
@app.get("/movies")
def get_movies():
    movies_ref = db.collection("movies").stream()
    movies = []
    for doc in movies_ref:
        movies.append(doc.to_dict())
    return {"movies": movies}

@app.get("/recommend")
def get_recommendations(username: str = None, user_id: str = None):
    # Support both username (new code) and user_id (frontend/old code)
    final_username = username or user_id
    if not final_username:
        return {"status": "error", "message": "username or user_id query parameter is required"}

    ratings_ref = db.collection("ratings").stream()
    all_ratings = {}
    for doc in ratings_ref:
        data = doc.to_dict()
        user = data.get("user_id")
        movie = data.get("movie")
        score = data.get("score")
        if user and movie and score is not None:
            if user not in all_ratings:
                all_ratings[user] = {}
            all_ratings[user][movie] = score
        
    df_matrix = pd.DataFrame(all_ratings).T
    df_matrix.index.name = "user"
    df_matrix.columns.name = "movie"
    
    if final_username not in df_matrix.index or df_matrix.loc[final_username].notna().sum() < 3:
        return {"status": "onboarding", "recommendations": []}
        
    profile = build_user_profile(final_username, df_matrix, movies_vector)
    recommendations = recommend_for_user(final_username, profile, df_matrix, movies_vector, top_n=10)
    
    # Adapt recommendations to list of dicts matching frontend expectations
    # The frontend expects a list of dicts with: title, genres, match, and poster_path
    formatted_recs = []
    for movie_title, score in recommendations[:10]:
        movie_data = my_movies.get(movie_title, {})
        formatted_recs.append({
            "title": movie_title,
            "genres": movie_data.get("genres", []),
            "match": round(float(score), 2),  # Rounded score for display
            "poster_path": movie_data.get("poster_path", "")
        })
        
    return {"status": "ok", "recommendations": formatted_recs}

@app.get("/rate")
@app.post("/rate")
def rate_movie(
    username: str = None, 
    movie: str = None, 
    rating: int = None, 
    data: RatingInput = None
):
    # Retrieve values from either JSON body (data) or query parameters
    final_username = username
    final_movie = movie
    final_rating = rating
    
    if data:
        if not final_username:
            final_username = data.username or data.user_id
        if not final_movie:
            final_movie = data.movie
        if final_rating is None:
            final_rating = data.rating
            if final_rating is None and data.label:
                # Map label to score
                label_map = {
                    "Loved it":    2,
                    "Liked it":    1,
                    "Neutral":     0,
                    "Didn't like": -1,
                    "Hated it":   -2,
                }
                direct_map = {
                    "SUPERB": 2,
                    "LOVE": 1,
                    "LIKE": 1,
                    "MEH": 0,
                    "DISLIKE": -1,
                }
                final_rating = label_map.get(data.label)
                if final_rating is None:
                    final_rating = direct_map.get(data.label.upper(), 0)

    if not final_username or not final_movie:
        return {"status": "error", "message": "Missing username or movie name"}
        
    if final_rating is None:
        final_rating = 0

    db.collection("ratings").document(f"{final_username}_{final_movie}").set(
        {"user_id": final_username, "movie": final_movie, "score": final_rating},
        merge=True
    )
    return {"status": "ok", "message": f"Rating saved for {final_movie}"}

# ── Local Server Startup ──────────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn
        from pyngrok import ngrok
        import nest_asyncio
        import asyncio

        nest_asyncio.apply()
        ngrok.set_auth_token("3FBEaL9gtGx9ikvmxumPppqefd0_2HrQiTufA8oWubBYb8wG9")
        public_url = ngrok.connect(8000)
        print(f"Public URL: {public_url}")

        config = uvicorn.Config(app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        asyncio.get_event_loop().run_until_complete(server.serve())
    except Exception as e:
        print(f"Ngrok/Local server start error or bypass: {e}")
