import os
import json
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── API Keys ──────────────────────────────────────────
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ── Firebase Setup ────────────────────────────────────
if not firebase_admin._apps:
    firebase_key_json = os.environ.get("FIREBASE_KEY")
    firebase_key_dict = json.loads(firebase_key_json)
    cred = credentials.Certificate(firebase_key_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ── Load Movies From Firebase ─────────────────────────
def load_movies():
    movies = {}
    docs = db.collection("movies").stream()
    for doc in docs:
        movies[doc.id] = doc.to_dict()
    print(f"Loaded {len(movies)} movies from Firebase")
    return movies

my_movies = load_movies()

# ── FastAPI App ───────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper Functions ──────────────────────────────────
def fetch_movie(movie_id):
    response = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    )
    data = response.json()
    genres_list = [genre["name"] for genre in data.get("genres", [])]
    return {
        "title":       data["title"],
        "year":        data["release_date"][:4],
        "rating":      data["vote_average"],
        "genres":      genres_list,
        "language":    data["original_language"],
        "poster_path": data.get("poster_path", "")
    }

def search_movie(query):
    response = requests.get(
        f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    )
    data = response.json()
    results = []
    for movie in data["results"][:5]:
        results.append({
            "id":    movie["id"],
            "title": movie["title"],
            "year":  movie.get("release_date", "")[:4]
        })
    return results

def rate_movie(user_id, movie_name, label):
    label_map = {
        "SUPERB":      "Loved it",
        "MASTERPIECE": "Loved it",
        "LOVE":        "Liked it",
        "LIKE":        "Liked it",
        "MEH":         "Neutral",
        "DISLIKE":     "Didn't like",
        "HATED":       "Hated it",
    }
    label = label_map.get(label.upper(), label)
    scores = {
        "Loved it":    2,
        "Liked it":    1,
        "Neutral":     0,
        "Didn't like": -1,
        "Hated it":   -2,
        "Not watched": None
    }
    return {
        "user_id": user_id,
        "movie":   movie_name,
        "label":   label,
        "score":   scores.get(label, 0)
    }

def add_and_rate(user_id, movie_query, label):
    results    = search_movie(movie_query)
    movie_id   = results[0]["id"]
    movie_name = results[0]["title"]

    my_movies[movie_name] = fetch_movie(movie_id)

    rating = rate_movie(user_id, movie_name, label)

    doc_id = user_id + "_" + movie_name
    db.collection("ratings").document(doc_id).set(rating)

    return movie_name

# ── Endpoints ─────────────────────────────────────────
@app.get("/")
def home():
    return {"message": "CineMatch India is alive!"}

@app.get("/movies")
def get_movies():
    movies_list = []
    for name in my_movies:
        movies_list.append({
            "title":       name,
            "genres":      my_movies[name]["genres"],
            "year":        my_movies[name]["year"],
            "poster_path": my_movies[name].get("poster_path", "")
        })
    return {"movies": movies_list, "total": len(movies_list)}

@app.get("/recommend")
def get_recommendations(user_id: str = "anmol_001"):
    try:
        user_ratings = []
        docs = db.collection("ratings").where("user_id", "==", user_id).stream()
        for doc in docs:
            user_ratings.append(doc.to_dict())

        user_taste = {}
        for rating in user_ratings:
            score = rating["score"]
            if score is None:
                continue
            genres = my_movies.get(rating["movie"], {}).get("genres", [])
            for genre in genres:
                if genre not in user_taste:
                    user_taste[genre] = 0
                user_taste[genre] += score

        already_rated = [r["movie"] for r in user_ratings]

        result = []
        for movie_name in my_movies:
            if movie_name not in already_rated:
                match_score = 0
                for genre in my_movies[movie_name]["genres"]:
                    if genre in user_taste:
                        match_score += user_taste[genre]
                num_genres = len(my_movies[movie_name]["genres"])
                if num_genres > 0:
                    match_score = match_score / num_genres
                result.append({
                    "title":       movie_name,
                    "genres":      my_movies[movie_name]["genres"],
                    "match":       match_score,
                    "poster_path": my_movies[movie_name].get("poster_path", "")
                })

        result = sorted(result, key=lambda x: x["match"], reverse=True)
        return {"recommendations": result[:5], "user_id": user_id}

    except Exception as e:
        return {"error": str(e), "recommendations": []}

class RatingInput(BaseModel):
    user_id: str
    movie:   str
    label:   str

@app.post("/rate")
def submit_rating(data: RatingInput):
    movie_name = add_and_rate(data.user_id, data.movie, data.label)
    return {"status": "saved", "movie": movie_name}
