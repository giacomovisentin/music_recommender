import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.metrics.pairwise import cosine_similarity

# --- App init & data load ---
app = Flask(__name__, template_folder="templates")

EMBEDDINGS_PATH = "embeddings.npy"
METADATA_PATH = "song_metadata.parquet"

# Load embeddings
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
emb = np.load(EMBEDDINGS_PATH)
print(f"Loaded embeddings shape: {emb.shape}")

# Load metadata (must contain track_name and artist_name and genre_* columns)
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
X = pd.read_parquet(METADATA_PATH)
X = X.reset_index(drop=True)
print(f"Loaded metadata rows: {len(X)}")

GENRE_COLS = [c for c in X.columns if c.startswith("genre_")]

# Precompute lowercase search fields for speed
X['_track_lower'] = X['track_name'].fillna('').str.lower()
X['_artist_lower'] = X['artist_name'].fillna('').str.lower()
# combine both for "both" searches
X['_both_lower'] = (X['_track_lower'] + " " + X['_artist_lower']).str.strip()

# --- Utility functions ---

def search_exact_index(title: str, artist: str):
    """Try exact match (case-insensitive). Return single index or None."""
    if title is None or artist is None:
        return None
    mask = (X['_track_lower'] == title.lower()) & (X['_artist_lower'] == artist.lower())
    matches = X[mask]
    if len(matches) == 1:
        return int(matches.index[0])
    return None

def fuzzy_find(title: str = None, artist: str = None, limit: int = 20):
    """
    Return up to `limit` candidate rows matching the query.
    If both title and artist given, search for rows where both substrings appear.
    If only one provided, search correspondingly.
    Results ordered by simple relevance heuristic (startswith > contains).
    """
    q_title = (title or "").strip().lower()
    q_artist = (artist or "").strip().lower()

    if q_title and q_artist:
        # prefer rows where both substrings appear
        both_mask = X['_both_lower'].str.contains(q_title) & X['_both_lower'].str.contains(q_artist)
        both = X[both_mask].copy()
        if len(both) >= limit:
            return both.head(limit)
        # fallback: any that contains both in any order
        either_mask = X['_both_lower'].str.contains(q_title) | X['_both_lower'].str.contains(q_artist)
        cand = pd.concat([both, X[either_mask].drop(both.index)]).drop_duplicates().head(limit)
        return cand
    elif q_title:
        starts = X[X['_track_lower'].str.startswith(q_title)].copy()
        contains = X[X['_track_lower'].str.contains(q_title) & ~X['_track_lower'].str.startswith(q_title)].copy()
        return pd.concat([starts, contains]).drop_duplicates().head(limit)
    elif q_artist:
        starts = X[X['_artist_lower'].str.startswith(q_artist)].copy()
        contains = X[X['_artist_lower'].str.contains(q_artist) & ~X['_artist_lower'].str.startswith(q_artist)].copy()
        return pd.concat([starts, contains]).drop_duplicates().head(limit)
    else:
        return X.head(limit)

def get_recommendations_by_index(idx: int, top_k: int = 10):
    """Compute cosine similarity against embeddings and return DataFrame of recommendations."""
    if idx < 0 or idx >= len(emb):
        raise IndexError("Index out of bounds for embeddings.")
    sims = cosine_similarity(emb[idx].reshape(1, -1), emb).flatten()
    order = sims.argsort()[::-1]
    # skip itself (first)
    top_idx = [i for i in order if i != idx][:top_k]
    recs = X.iloc[top_idx].copy()
    recs = recs.reset_index(drop=False)  # keep original index in 'index' column
    recs['similarity'] = sims[top_idx]
    # human-friendly genres column
    def extract_genres(row):
        genres = [c.replace('genre_', '') for c in GENRE_COLS if row.get(c, 0) > 0]
        return ", ".join(genres) if genres else "N/A"
    recs['genres'] = recs.apply(extract_genres, axis=1)
    recs['orig_index'] = recs['index']  # original dataframe index for client
    return recs[['track_name', 'artist_name', 'genres', 'similarity', 'orig_index']]

# --- API endpoints ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggest', methods=['GET'])
def suggest():
    """
    Query params:
      q: the typed string
      type: one of 'title', 'artist', 'both' (default 'both')
      limit: max suggestions
    Returns list of suggestions with fields: track_name, artist_name, index
    """
    q = request.args.get('q', '').strip()
    t = request.args.get('type', 'both')
    limit = int(request.args.get('limit', 10))

    if not q:
        return jsonify([])

    if t not in ('title', 'artist', 'both'):
        t = 'both'

    if t == 'both':
        # split heuristically: try to see if user typed "title - artist"
        if '-' in q:
            parts = [p.strip() for p in q.split('-', 1)]
            title_part = parts[0]
            artist_part = parts[1] if len(parts) > 1 else ''
            candidates = fuzzy_find(title=title_part, artist=artist_part, limit=limit)
        else:
            # search both combined
            candidates = X[X['_both_lower'].str.contains(q.lower())].head(limit)
            if candidates.empty:
                candidates = fuzzy_find(title=q, artist=None, limit=limit)
    elif t == 'title':
        candidates = fuzzy_find(title=q, artist=None, limit=limit)
    else:  # artist
        candidates = fuzzy_find(title=None, artist=q, limit=limit)

    # Build response
    suggestions = []
    for idx, row in candidates.iterrows():
        suggestions.append({
            "track_name": row['track_name'],
            "artist_name": row['artist_name'],
            "index": int(idx)
        })
        if len(suggestions) >= limit:
            break

    return jsonify(suggestions)

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Body JSON can contain either:
      { "index": <int> }   OR
      { "title": "...", "artist": "...", "mode": "exact|fuzzy" }
    Response:
      success: list of recommendations or list of matches if ambiguous
    """
    data = request.get_json(force=True)
    # 1) If index provided, use it directly
    if data.get('index') is not None:
        idx = int(data['index'])
        try:
            recs = get_recommendations_by_index(idx, top_k=int(data.get('top_k', 10)))
            return jsonify({"status": "success", "recommendations": recs.to_dict(orient='records')})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    # 2) If title/artist provided, attempt exact then fuzzy
    title = data.get('title', '').strip()
    artist = data.get('artist', '').strip()
    mode = data.get('mode', 'exact')  # exact or fuzzy

    if not title and not artist:
        return jsonify({"status": "error", "message": "Provide 'index' or at least one of 'title'/'artist'."}), 400

    # Try exact
    if title and artist:
        exact_idx = search_exact_index(title, artist)
        if exact_idx is not None:
            recs = get_recommendations_by_index(exact_idx, top_k=int(data.get('top_k', 10)))
            return jsonify({"status": "success", "recommendations": recs.to_dict(orient='records')})

    if mode == 'exact':
        # return list of candidates for user to choose
        candidates = fuzzy_find(title=title or None, artist=artist or None, limit=20)
        matches = []
        for idx, row in candidates.iterrows():
            matches.append({
                "track_name": row['track_name'],
                "artist_name": row['artist_name'],
                "index": int(idx)
            })
        return jsonify({"status": "ambiguous", "matches": matches})

    # mode == 'fuzzy': pick top fuzzy candidate (if any)
    candidates = fuzzy_find(title=title or None, artist=artist or None, limit=1)
    if len(candidates) == 0:
        return jsonify({"status": "error", "message": "No matches found."}), 404
    chosen_idx = int(candidates.index[0])
    recs = get_recommendations_by_index(chosen_idx, top_k=int(data.get('top_k', 10)))
    return jsonify({"status": "success", "recommendations": recs.to_dict(orient='records')})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
