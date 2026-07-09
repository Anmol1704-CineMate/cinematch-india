import os
import json
import numpy as np
import pandas as pd
import requests
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS
from surprise import SVD, Dataset, Reader

# ── Firebase Setup ────────────────────────────────────
firebase_key_json = os.environ.get("FIREBASE_KEY")
if firebase_key_json:
    try:
        firebase_key_dict = json.loads(firebase_key_json)
        cred = credentials.Certificate(firebase_key_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
    except Exception as e:
        print(f"Error initializing Firebase with key: {e}")
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
else:
    if not firebase_admin._apps:
        try:
            firebase_admin.initialize_app()
        except Exception as e:
            print(f"Firebase default initialization failed: {e}")

db = firestore.client()

# ── Load Movies ───────────────────────────────────────
def load_movies():
    try:
        docs = db.collection("movies").stream()
        movies = {}
        for doc in docs:
            movies[doc.id] = doc.to_dict()
        return movies
    except Exception as e:
        print(f"Error loading movies from Firestore: {e}")
        return {}

my_movies = load_movies()

# ── SVD Movie Factors Loading ─────────────────────────
movie_factors = {}
if os.path.exists("movie_factors.json"):
    try:
        with open("movie_factors.json", "r") as f:
            movie_factors = json.load(f)
        print("Loaded movie_factors.json from local file.")
    except Exception as e:
        print(f"Error loading local movie_factors.json: {e}")

if not movie_factors and os.environ.get("MOVIE_FACTORS"):
    try:
        movie_factors = json.loads(os.environ.get("MOVIE_FACTORS"))
        print("Loaded movie_factors from MOVIE_FACTORS env variable.")
    except Exception as e:
        print(f"Error loading MOVIE_FACTORS env variable: {e}")

# ── Flickscore Ratings Loading ────────────────────────
flickscore_ratings = []
if os.path.exists("flickscore_ratings.json"):
    try:
        with open("flickscore_ratings.json", "r") as f:
            flickscore_ratings = json.load(f)
        print("Loaded flickscore_ratings.json from local file.")
    except Exception as e:
        print(f"Error loading local flickscore_ratings.json: {e}")
elif os.path.exists("flickscore_ratings.csv"):
    try:
        df_flick = pd.read_csv("flickscore_ratings.csv")
        flickscore_ratings = df_flick.to_dict(orient="records")
        print("Loaded flickscore_ratings.csv from local file.")
    except Exception as e:
        print(f"Error loading local flickscore_ratings.csv: {e}")

if not flickscore_ratings and os.environ.get("FLICKSCORE_RATINGS"):
    try:
        flickscore_ratings = json.loads(os.environ.get("FLICKSCORE_RATINGS"))
        print("Loaded flickscore_ratings from FLICKSCORE_RATINGS env variable.")
    except Exception as e:
        print(f"Error loading FLICKSCORE_RATINGS env variable: {e}")

# ── Latent Factors Cosine Similarity Logic ────────────
def build_user_profile_factors(username, df_matrix, movie_factors):
    if username not in df_matrix.index:
        return None
    user_ratings = df_matrix.loc[username]
    rated_movies = user_ratings.dropna()
    if len(rated_movies) == 0:
        return None
    
    first_key = next(iter(movie_factors.values()))
    factor_length = len(first_key)
    
    profile_vector = np.zeros(factor_length)
    valid_ratings_count = 0
    
    for movie, rating in rated_movies.items():
        if movie in movie_factors:
            factors = np.array(movie_factors[movie])
            profile_vector += factors * rating
            valid_ratings_count += 1
            
    if valid_ratings_count == 0:
        return None
        
    return profile_vector / valid_ratings_count

def recommend_for_user_factors(username, profile, df_matrix, movie_factors, top_n=10):
    if profile is None:
        return []
    user_ratings = df_matrix.loc[username]
    already_rated = set(user_ratings.dropna().index)
    
    scores = []
    profile_norm = np.linalg.norm(profile)
    if profile_norm == 0:
        return []
        
    for movie, factors in movie_factors.items():
        if movie in already_rated or movie not in my_movies:
            continue
        movie_vector = np.array(factors)
        movie_norm = np.linalg.norm(movie_vector)
        if movie_norm == 0:
            continue
        
        dot_product = np.dot(profile, movie_vector)
        similarity = dot_product / (profile_norm * movie_norm)
        scores.append((movie, similarity))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

# ── Surprise SVD Collaborative Filtering Logic ─────────
def get_surprise_recommendations(username, all_ratings, top_n=10):
    raw_data = []
    for user, movies in all_ratings.items():
        for movie, rating in movies.items():
            raw_data.append((user, movie, rating))
            
    if not raw_data:
        return []
        
    df = pd.DataFrame(raw_data, columns=["user", "item", "rating"])
    reader = Reader(rating_scale=(-1, 1))
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()
    
    algo = SVD()
    algo.fit(trainset)
    
    user_ratings = all_ratings.get(username, {})
    predictions = []
    for movie in my_movies.keys():
        if movie not in user_ratings:
            pred = algo.predict(username, movie)
            match_score = (pred.est + 1) / 2.0
            predictions.append((movie, match_score))
            
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]

# ── Flask Setup & Routing ─────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "CineMatch India Flask API is active",
        "has_movie_factors": len(movie_factors) > 0,
        "has_flickscore_ratings": len(flickscore_ratings) > 0
    })

@app.route("/movies", methods=["GET"])
def get_movies():
    global my_movies
    movies_list = list(my_movies.values())
    if not movies_list:
        my_movies = load_movies()
        movies_list = list(my_movies.values())
    return jsonify({"movies": movies_list})

@app.route("/onboarding", methods=["GET"])
def get_onboarding():
    onboarding_list = [
        "3 Idiots", "Sholay", "Dil Chahta Hai", 
        "Lagaan: Once Upon a Time in India", "Queen", 
        "Gangs of Wasseypur", "Taare Zameen Par", 
        "Andaz Apna Apna", "Pink", "Rang De Basanti"
    ]
    return jsonify({"movies": onboarding_list})

@app.route("/rate", methods=["GET", "POST"])
def rate_movie():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        username = data.get("username") or data.get("user_id") or request.args.get("username")
        movie = data.get("movie") or request.args.get("movie")
        rating = data.get("rating")
        if rating is None:
            rating = request.args.get("rating")
    else:
        username = request.args.get("username") or request.args.get("user_id")
        movie = request.args.get("movie")
        rating = request.args.get("rating")

    if not username or not movie:
        return jsonify({"status": "error", "message": "Missing username or movie name"}), 400

    try:
        rating_val = int(rating) if rating is not None else 0
    except ValueError:
        rating_val = 0

    try:
        db.collection("ratings").document(f"{username}_{movie}").set(
            {"user_id": username, "movie": movie, "score": rating_val},
            merge=True
        )
        return jsonify({"status": "ok", "message": f"Rating saved for {movie}"})
    except Exception as e:
        print(f"Error saving rating: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/recommend", methods=["GET"])
def get_recommendations():
    global my_movies
    username = request.args.get("username") or request.args.get("user_id")
    if not username:
        return jsonify({"status": "error", "message": "username query parameter is required"}), 400

    if not my_movies:
        my_movies = load_movies()

    try:
        # Load all ratings from Firestore
        ratings_ref = db.collection("ratings").stream()
        all_ratings = {}

        # 1. Seed with flickscore_ratings
        if flickscore_ratings:
            for r in flickscore_ratings:
                user = r.get("user_id") or r.get("userId") or r.get("username")
                movie = r.get("movie") or r.get("movieId") or r.get("title")
                score = r.get("score") or r.get("rating")
                if user and movie and score is not None:
                    if user not in all_ratings:
                        all_ratings[user] = {}
                    all_ratings[user][movie] = float(score)

        # 2. Overlay with live user ratings from Firestore
        for doc in ratings_ref:
            data = doc.to_dict()
            user = data.get("user_id")
            movie = data.get("movie")
            score = data.get("score")
            if user and movie and score is not None:
                if user not in all_ratings:
                    all_ratings[user] = {}
                all_ratings[user][movie] = float(score)

        # check user rating counts
        user_ratings = all_ratings.get(username, {})
        if len(user_ratings) < 3:
            return jsonify({"status": "onboarding", "recommendations": []})

        # Predict based on movie factors presence
        recommendations = []
        if movie_factors:
            df_matrix = pd.DataFrame(all_ratings).T
            df_matrix.index.name = "user"
            df_matrix.columns.name = "movie"
            profile = build_user_profile_factors(username, df_matrix, movie_factors)
            recommendations = recommend_for_user_factors(username, profile, df_matrix, movie_factors, top_n=10)
        else:
            # Fall back to surprise SVD model
            recommendations = get_surprise_recommendations(username, all_ratings, top_n=10)

        # Format output
        formatted_recs = []
        for movie_title, score in recommendations[:10]:
            movie_data = my_movies.get(movie_title, {})
            formatted_recs.append({
                "title": movie_title,
                "genres": movie_data.get("genres", []),
                "match": round(float(score), 2),
                "poster_path": movie_data.get("poster_path", "")
            })

        return jsonify({"status": "ok", "recommendations": formatted_recs})

    except Exception as e:
        print(f"Error in recommendation logic: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
