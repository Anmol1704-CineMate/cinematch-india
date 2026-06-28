# ═══════════════════════════════════════════════════════
# CINEMATCH INDIA — CLEAN NOTEBOOK
# Cell 1: Startup
# ═══════════════════════════════════════════════════════

import subprocess
subprocess.run(["pip", "install", "firebase-admin", "-q"])
subprocess.run(["pip", "install", "fastapi", "-q"])
subprocess.run(["pip", "install", "uvicorn", "-q"])
subprocess.run(["pip", "install", "pyngrok", "-q"])

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import firebase_admin
from firebase_admin import credentials, firestore
from google.colab import drive
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

print("All libraries imported!")

# ── API Keys ──────────────────────────────────────────
TMDB_API_KEY = "950551e0b37cd664fa188ec51a5a20d6"

# ── Mount Drive & Connect Firebase ───────────────────
drive.mount("/content/drive")

if not firebase_admin._apps:
    cred = credentials.Certificate("/content/drive/MyDrive/Cinemate/cinemate-a6e29-firebase-adminsdk-fbsvc-0c072c5e46.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
print("Firebase connected!")




# ── Load Movies from Firebase ─────────────────────────
my_movies = {}
docs = db.collection("movies").stream()

for doc in docs:
    data = doc.to_dict()
    title = data.get("title", doc.id)
    my_movies[title] = {
        "genres":      data.get("genres", []),
        "rating":      data.get("rating", 5.0),
        "year":        int(data.get("year", 2000)),
        "poster_path": data.get("poster_path", "")
    }

print(f"Loaded {len(my_movies)} movies from Firebase!")
# ── Build Movie Fingerprints ──────────────────────────

# Step 1: Find all unique genres
all_unique_genres_set = set()
for name, data in my_movies.items():
    for genre in data["genres"]:
        all_unique_genres_set.add(genre)
all_unique_genres = sorted(all_unique_genres_set)

# Step 2: Normalise ratings (min-max)
all_ratings_list = [data["rating"] for name, data in my_movies.items()]
min_rating       = min(all_ratings_list)
max_rating       = max(all_ratings_list)

# Step 3: Normalise years (min-max)
all_years_list = [data["year"] for name, data in my_movies.items()]
min_year       = min(all_years_list)
max_year       = max(all_years_list)

# Step 4: Build fingerprints
movies_vector = {}
for i, (movie_name, data) in enumerate(my_movies.items()):
    genre_vector = []
    for genre in all_unique_genres:
        if genre in data["genres"]:
            genre_vector.append(1)
        else:
            genre_vector.append(0)

    if max_rating == min_rating:
        norm_rating = 0.5
    else:
        norm_rating = (data["rating"] - min_rating) / (max_rating - min_rating)

    if max_year == min_year:
        norm_year = 0.5
    else:
        norm_year = (data["year"] - min_year) / (max_year - min_year)

    movies_vector[movie_name] = genre_vector + [norm_rating] + [norm_year]

print(f"Fingerprints built! Each movie has {len(list(movies_vector.values())[0])} features.")
#Block 5


# ── Build Similarity Matrix ───────────────────────────
movie_names   = list(movies_vector.keys())
matrix        = list(movies_vector.values())
sim_matrix    = sklearn_cosine(matrix)
df_similarity = pd.DataFrame(sim_matrix, index=movie_names, columns=movie_names)

print(f"Similarity matrix built! Shape: {df_similarity.shape}")




#Block 6

# ── User-Item Matrix ──────────────────────────────────

# Step 1: Pull ratings from Firebase
all_ratings_data = []
ratings_ref = db.collection("ratings").stream()

for doc in ratings_ref:
    data = doc.to_dict()
    if "user_id" not in data or "movie" not in data or "score" not in data:
        continue
    all_ratings_data.append({
        "user":   data["user_id"],
        "movie":  data["movie"],
        "rating": data["score"]
    })

print(f"Total ratings found: {len(all_ratings_data)}")

# Step 2: Build User-Item Matrix
df_ratings       = pd.DataFrame(all_ratings_data)
user_item_matrix = df_ratings.pivot_table(index="user", columns="movie", values="rating")

# Step 3: Keep only real users
real_users       = ["Anmol", "Om", "Om M", "Palindrome", "TestUser", "anmol_001"]
user_item_matrix = user_item_matrix.loc[real_users]

print(f"Matrix shape: {user_item_matrix.shape}")
print("Startup complete! Ready to recommend. 🎬")
#ML In Recommendations

# ═══════════════════════════════════════════════════════
# CELL 2 — ML Functions
# ═══════════════════════════════════════════════════════

def build_user_profile(username, df_matrix, movies_vector):
    if username not in df_matrix.index:
        print(f"User '{username}' not found.")
        return None

    user_ratings = df_matrix.loc[username]
    rated_movies = user_ratings.dropna()

    if len(rated_movies) == 0:
        print(f"User '{username}' has no ratings.")
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

def recommend_for_user(username, profile, df_matrix, movies_vector, top_n=5):
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
        magnitude_movie   = sum(x**2 for x in fingerprint) ** 0.5

        if magnitude_profile == 0 or magnitude_movie == 0:
            similarity = 0
        else:
            similarity = dot_product / (magnitude_profile * magnitude_movie)

        scores.append((movie, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]
# ═══════════════════════════════════════════════════════
# CELL 3 — FastAPI
# ═══════════════════════════════════════════════════════

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("FastAPI app created!")



#CELL 3 - Block 2


@app.get("/movies")
def get_movies():
    movies_ref = db.collection("movies").stream()
    movies = []
    for doc in movies_ref:
        movies.append(doc.to_dict())
    return {"movies": movies}

@app.get("/recommend")
def get_recommendations(username: str):
    # Step 1: Fetch ratings from Firebase
    ratings_ref = db.collection("ratings").stream()
    all_ratings = {}

    for doc in ratings_ref:
        data = doc.to_dict()
        user = data.get("user_id")
        movie = data.get("movie")
        score = data.get("score")

        if user not in all_ratings:
            all_ratings[user] = {}
        all_ratings[user][movie] = score

    # Step 2: Build User-Item Matrix
    df_matrix = pd.DataFrame(all_ratings).T
    df_matrix.index.name = "user"
    df_matrix.columns.name = "movie"

    # Step 3: Check onboarding
    if username not in df_matrix.index or df_matrix.loc[username].notna().sum() < 3:
        return {"status": "onboarding"}

    # Step 4: Build profile and recommend
    profile = build_user_profile(username, df_matrix, movies_vector)
    recommendations = recommend_for_user(username, profile, df_matrix, movies_vector)

    return {"status": "ok", "recommendations": recommendations[:5]}
@app.post("/rate")
def rate_movie(username: str, movie: str, rating: int):
    db.collection("ratings").document(username).set(
        {movie: rating},
        merge=True
    )
    return {"status": "ok", "message": f"Rating saved for {movie}"}
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
