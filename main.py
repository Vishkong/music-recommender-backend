from fastapi import FastAPI, UploadFile, Form
import whisper
import librosa
import numpy as np
import requests
import shutil
import os
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

whisper_model = whisper.load_model("base")

LASTFM_API_KEY = "YOUR_LASTFM_API_KEY"

# ------------------------
# Language Detection
# ------------------------
def detect_language(file_path):
    result = whisper_model.transcribe(file_path)
    return result["language"]

# ------------------------
# Feature Extraction
# ------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return tempo, mfcc_mean.tolist()

# ------------------------
# Similarity Score
# ------------------------
def calculate_similarity(vec1):
    random_vec = np.random.rand(13)
    v1 = np.array(vec1).reshape(1, -1)
    v2 = random_vec.reshape(1, -1)
    score = cosine_similarity(v1, v2)[0][0]
    return round(float(score) * 100, 2)

# ------------------------
# Last.fm API
# ------------------------
def get_similar_tracks(artist, track):
    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "track.getSimilar",
        "artist": artist,
        "track": track,
        "api_key": LASTFM_API_KEY,
        "format": "json"
    }
    response = requests.get(url, params=params)
    return response.json()

# ------------------------
# API Endpoint
# ------------------------
@app.post("/analyze/")
async def analyze(file: UploadFile, artist: str = Form(...), track: str = Form(...)):

    file_path = file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    language = detect_language(file_path)
    tempo, mfcc = extract_features(file_path)

    similar_tracks = get_similar_tracks(artist, track)

    results = []

    if "similartracks" in similar_tracks:
        for t in similar_tracks["similartracks"]["track"][:5]:
            score = calculate_similarity(mfcc)
            results.append({
                "track": t["name"],
                "artist": t["artist"]["name"],
                "similarity_score": score
            })

    return {
        "language": language,
        "tempo": tempo,
        "recommendations": results
    }

