SongVibeRecommender 🎵

This project is a music recommendation web app built with Python, scikit-learn, and Flask. Unlike recommenders that focus on genre or artist, this tool finds songs that share a similar "vibe" by analyzing their raw audio features.

The entire process is detailed in my Medium article.

🚀 Live Demo

You can try the live application here:

https://music-recommender-xyh1.onrender.com

Please Note: The backend is hosted on Render's free tier. If the app hasn't been used in 15 minutes, it "sleeps" to save resources. It may take about 60 seconds to load the first time you visit it. Please be patient!

App Screenshot

(Inserisci qui uno screenshot della tua app! Sostituisci questo testo)

⚙️ How It Works

The recommendation engine is built on a simple but effective machine learning workflow:

Data Loading: Uses the Ultimate Spotify Tracks Dataset from Kaggle.

Preprocessing: Cleans the data, handles missing values, one-hot encodes categorical features (like key), and scales continuous audio features (like danceability, energy, tempo) using StandardScaler.

Embeddings: Converts each song into a numerical feature vector (embedding) that represents its audio profile.

Similarity: Uses Cosine Similarity (sklearn.metrics.pairwise.cosine_similarity) to calculate the "distance" between any two song vectors. Songs with similar vectors are considered to have a similar vibe.

API: A simple Flask server wraps this logic into two endpoints:

/suggest: Provides search suggestions as the user types.

/recommend: Takes a song index and returns the top 10 most similar tracks.

Frontend: A clean, Spotify-inspired interface built with vanilla HTML, CSS, and JavaScript that fetches data from the Flask API.

🛠️ Tech Stack

Backend: Python, Flask

Data Science: Pandas, NumPy, scikit-learn

Frontend: HTML, CSS, JavaScript

Deployment: Render
