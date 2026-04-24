"""
Step 4: Hybrid Recommendation Engine
--------------------------------------
Combines:
  1. NCF Deep Learning scores (collaborative filtering)
  2. TF-IDF Content Similarity (genre + language matching)
  3. Language Boost (Telugu/Hindi/English priority from user history)

When a user logs in, the engine:
  - Fetches their watch history
  - Detects their preferred language & genres
  - Scores ALL unseen movies using the hybrid formula
  - Returns top-K personalized recommendations
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

DLCASE = r"c:\Users\navee\OneDrive\Desktop\DLCASE"


class RecommendationEngine:
    """Hybrid Deep Learning Recommendation Engine."""

    def __init__(self):
        print("Loading Recommendation Engine...")

        # Load metadata
        with open("meta.pkl", "rb") as f:
            self.meta = pickle.load(f)

        # Load encoders
        with open("user_encoder.pkl", "rb") as f:
            self.user_enc = pickle.load(f)
        with open("movie_encoder.pkl", "rb") as f:
            self.movie_enc = pickle.load(f)

        # Load data
        self.movies = pd.read_csv("movies_clean.csv")
        self.ratings = pd.read_csv("ratings_clean.csv")
        self.survey = pd.read_csv("cleaned_survey.csv")

        # Sort movies by encoded ID for index alignment
        self.movies = self.movies.sort_values('movie_enc').reset_index(drop=True)

        # Load NCF model
        self.ncf = tf.keras.models.load_model("ncf_model.h5", compile=False)

        # Load TF-IDF content matrix
        self.content_matrix = load_npz("content_matrix.npz")

        self.num_users = self.meta['num_users']
        self.num_items = self.meta['num_items']

        print(f"  Engine ready: {self.num_users} users, {self.num_items} movies")

    # ------------------------------------------------------------------
    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Get all movies a user has previously watched with details."""
        user_ratings = self.ratings[self.ratings['userId'] == user_id].copy()
        if user_ratings.empty:
            return pd.DataFrame()

        # Merge with movie info
        history = user_ratings.merge(
            self.movies[['movieId', 'title', 'genres_str', 'primary_language',
                         'popularity', 'movie_enc']],
            on='movieId', how='left'
        )
        # Sort by rating (highest first)
        history = history.sort_values('rating', ascending=False)
        return history

    # ------------------------------------------------------------------
    # Known Telugu/Hindi movie keywords for language detection from survey
    TELUGU_KEYWORDS = [
        'dhurandhar', 'salaar', 'salar', 'rrr', 'pushpa', 'animal', 'dhruva',
        'baahubali', 'bahubali', 'raja rani', 'just married', 'godavari',
        'couple friendly', 'kishkindapuri', 'with love', 'sirai', 'og',
        'splitsvilla', '8 vasantalu', 'isha', 'jigris', 'saaiyara',
        'rajsaab', 'raj saab', 'ustad bhagat', 'vikram', 'hit 3', 'hit3',
        'dude', 'laapata', 'made in korea', 'naruto', 'durandhar', 'dhurandar',
        'sarvam maya', 'hi nana', 'gandhi talks', 'om shanti',
    ]
    HINDI_KEYWORDS = [
        'pushpa', 'animal', 'laapata ladies', 'stranger things',
        'money heist', 'dhurandhar', 'durandhar',
    ]

    def _detect_language_from_survey(self, user_id: str) -> list:
        """Detect user's preferred language from survey recent_favorite field."""
        survey_row = self.survey[self.survey['user_id'] == user_id]
        if survey_row.empty:
            return []

        favorite = str(survey_row.iloc[0].get('recent_favorite', '')).lower().strip()
        if not favorite or favorite == 'nan':
            return []

        # Check for Telugu movie keywords
        for kw in self.TELUGU_KEYWORDS:
            if kw in favorite:
                return ['Telugu', 'Hindi', 'English']

        # Check for Hindi movie keywords
        for kw in self.HINDI_KEYWORDS:
            if kw in favorite:
                return ['Hindi', 'Telugu', 'English']

        return ['English']

    # ------------------------------------------------------------------
    def get_user_preferences(self, user_id: str) -> dict:
        """Detect user's preferred language and genres from watch history + survey."""
        history = self.get_user_history(user_id)
        prefs = {'languages': [], 'genres': [], 'avg_rating': 0}

        # Always check survey for genre preferences
        survey_row = self.survey[self.survey['user_id'] == user_id]
        if not survey_row.empty:
            genres_str = survey_row.iloc[0].get('preferred_genres', '')
            if pd.notna(genres_str):
                prefs['genres'] = [g.strip() for g in str(genres_str).replace('|', ',').split(',') if g.strip()]

        # Detect language from survey favorites (Telugu/Hindi detection)
        survey_langs = self._detect_language_from_survey(user_id)

        if history.empty:
            prefs['languages'] = survey_langs if survey_langs else ['English']
            return prefs

        # Extract preferred languages from watch history
        if 'primary_language' in history.columns:
            lang_counts = history.groupby('primary_language')['rating'].agg(['count', 'mean'])
            lang_counts['score'] = lang_counts['count'] * lang_counts['mean']
            history_langs = lang_counts.sort_values('score', ascending=False).index.tolist()
        else:
            history_langs = []

        # Merge: survey language takes priority (since survey shows real preference)
        if survey_langs:
            # Put survey-detected language first, then add history languages
            combined = list(survey_langs)
            for lang in history_langs:
                if lang not in combined:
                    combined.append(lang)
            prefs['languages'] = combined
        else:
            prefs['languages'] = history_langs

        # Extract preferred genres from history (supplement survey genres)
        all_genres = []
        for gs in history['genres_str'].dropna():
            all_genres.extend([g.strip() for g in str(gs).split('|')])
        if all_genres:
            from collections import Counter
            genre_counts = Counter(all_genres)
            history_genres = [g for g, _ in genre_counts.most_common(5)]
            # Merge with survey genres
            for g in history_genres:
                if g not in prefs['genres']:
                    prefs['genres'].append(g)

        prefs['avg_rating'] = history['rating'].mean()
        return prefs

    # ------------------------------------------------------------------
    def _ncf_scores(self, user_enc: int, movie_encs: np.ndarray) -> np.ndarray:
        """Get NCF predicted scores for a user across multiple movies."""
        user_arr = np.full(len(movie_encs), user_enc)
        preds = self.ncf.predict([user_arr, movie_encs], verbose=0, batch_size=1024)
        return preds.flatten()

    # ------------------------------------------------------------------
    def _content_scores(self, watched_encs: list, candidate_encs: np.ndarray) -> np.ndarray:
        """Compute content similarity between watched movies and candidates."""
        if not watched_encs:
            return np.zeros(len(candidate_encs))

        # Average the TF-IDF vectors of all watched movies to get a user profile
        watched_vecs = self.content_matrix[watched_encs]
        user_profile = np.asarray(watched_vecs.mean(axis=0))
        if user_profile.ndim == 1:
            user_profile = user_profile.reshape(1, -1)

        # Compute similarity to all candidate movies
        candidate_vecs = self.content_matrix[candidate_encs]
        sims = cosine_similarity(user_profile, candidate_vecs)
        return sims.flatten()

    # ------------------------------------------------------------------
    def _language_boost(self, preferred_langs: list, candidate_encs: np.ndarray) -> np.ndarray:
        """Strong language boost — heavily rewards matching language, penalizes mismatch."""
        if not preferred_langs:
            return np.zeros(len(candidate_encs))

        boost = np.zeros(len(candidate_encs))
        top_lang = preferred_langs[0]  # primary preferred language

        for i, enc in enumerate(candidate_encs):
            if enc < len(self.movies):
                movie_lang = self.movies.iloc[enc]['primary_language']
                if movie_lang == top_lang:
                    boost[i] = 1.0  # full boost for #1 language
                elif movie_lang in preferred_langs:
                    rank = preferred_langs.index(movie_lang)
                    boost[i] = max(0.7 - rank * 0.15, 0.2)
                else:
                    boost[i] = -0.3  # penalize non-preferred languages
        return boost

    # ------------------------------------------------------------------
    def _filter_by_language(self, candidate_encs: np.ndarray,
                            preferred_langs: list, min_results: int = 20) -> np.ndarray:
        """Filter candidates to preferred language. Falls back if too few results."""
        if not preferred_langs:
            return candidate_encs

        top_lang = preferred_langs[0]

        # Try: only top preferred language
        lang_mask = np.array([
            self.movies.iloc[enc]['primary_language'] == top_lang
            if enc < len(self.movies) else False
            for enc in candidate_encs
        ])
        filtered = candidate_encs[lang_mask]

        if len(filtered) >= min_results:
            return filtered

        # Fallback: all preferred languages
        lang_mask = np.array([
            self.movies.iloc[enc]['primary_language'] in preferred_langs
            if enc < len(self.movies) else False
            for enc in candidate_encs
        ])
        filtered = candidate_encs[lang_mask]

        if len(filtered) >= min_results:
            return filtered

        # Last fallback: return all candidates (not enough in preferred lang)
        return candidate_encs

    # ------------------------------------------------------------------
    def _filter_by_genre(self, candidate_encs: np.ndarray,
                         genre_filter: list) -> np.ndarray:
        """Strict genre filter — only keep movies that contain at least one selected genre."""
        if not genre_filter:
            return candidate_encs

        mask = np.array([
            any(g.strip() in str(self.movies.iloc[enc].get('genres_str', '')).split('|')
                for g in genre_filter)
            if enc < len(self.movies) else False
            for enc in candidate_encs
        ])
        filtered = candidate_encs[mask]
        return filtered if len(filtered) > 0 else candidate_encs

    # ------------------------------------------------------------------
    def recommend(self, user_id: str, top_k: int = 10, genre_filter: list = None, lang_filter: str = None) -> list:
        """
        Generate top-K hybrid recommendations for a user.

        Strategy: LANGUAGE-FIRST filtering, then hybrid scoring.
        1. Detect user's preferred language from watch history
        2. Filter candidates to that language
        3. Score using: 0.5 * NCF + 0.3 * Content + 0.2 * Language Boost

        Returns a list of dicts with movie info + scores.
        """
        # Check if user exists
        if user_id not in self.user_enc.classes_:
            return self._cold_start_recommend(user_id, top_k, genre_filter=genre_filter, lang_filter=lang_filter)

        user_enc_id = self.user_enc.transform([user_id])[0]

        # Get watched movie encoded IDs from ratings (which has movie_enc)
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        watched_encs = set(user_ratings['movie_enc'].dropna().astype(int).tolist()) if not user_ratings.empty else set()

        # Get user preferences
        prefs = self.get_user_preferences(user_id)

        # Candidate movies = all movies NOT yet watched
        all_encs = np.arange(self.num_items)
        candidate_mask = ~np.isin(all_encs, list(watched_encs))
        candidate_encs = all_encs[candidate_mask]

        if len(candidate_encs) == 0:
            return []

        # LANGUAGE-FIRST: if UI lang_filter provided, use it; otherwise use user prefs
        if lang_filter and lang_filter != 'Any':
            preferred_langs = [lang_filter]
        else:
            preferred_langs = prefs.get('languages', [])
        candidate_encs = self._filter_by_language(candidate_encs, preferred_langs,
                                                   min_results=top_k)

        # GENRE FILTER: strict — only show movies matching selected genres
        if genre_filter:
            candidate_encs = self._filter_by_genre(candidate_encs, genre_filter)

        if len(candidate_encs) == 0:
            return []

        # Score 1: NCF Deep Learning
        ncf_scores = self._ncf_scores(user_enc_id, candidate_encs)

        # Score 2: Content Similarity (genre + language via TF-IDF)
        content_scores = self._content_scores(list(watched_encs), candidate_encs)

        # Score 3: Language Boost (still applied for ranking within filtered set)
        lang_boost = self._language_boost(preferred_langs, candidate_encs)

        # Hybrid combination — language has strong influence
        hybrid_scores = (
            0.45 * ncf_scores +
            0.30 * content_scores +
            0.25 * lang_boost
        )

        # Get top-K indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            enc_id = candidate_encs[idx]
            if enc_id >= len(self.movies):
                continue
            movie = self.movies.iloc[enc_id]
            results.append({
                'title': movie['title'],
                'year': movie.get('year', ''),
                'genres': movie.get('genres_str', 'Unknown'),
                'language': movie.get('primary_language', 'Unknown'),
                'popularity': float(movie.get('popularity', 0)),
                'ncf_score': float(ncf_scores[idx]),
                'content_score': float(content_scores[idx]),
                'lang_boost': float(lang_boost[idx]),
                'hybrid_score': float(hybrid_scores[idx]),
                'movie_enc': int(enc_id),
            })
        return results

    # ------------------------------------------------------------------
    def _cold_start_recommend(self, user_id: str, top_k: int,
                              genre_filter: list = None, lang_filter: str = None) -> list:
        """For new users not in training data, use survey preferences + popularity."""
        survey_row = self.survey[self.survey['user_id'] == user_id]
        preferred_genres = []
        if not survey_row.empty:
            gs = survey_row.iloc[0].get('preferred_genres', '')
            if pd.notna(gs):
                preferred_genres = [g.strip() for g in str(gs).replace('|', ',').split(',') if g.strip()]

        # If explicit genre filter provided, use that instead
        if genre_filter:
            preferred_genres = genre_filter

        # Score by genre match + popularity, FILTER by language FIRST
        results = []
        for _, movie in self.movies.iterrows():
            movie_lang = movie.get('primary_language', 'Unknown')

            # LANGUAGE FILTER: skip movies that don't match selected language
            if lang_filter and lang_filter != 'Any' and movie_lang != lang_filter:
                continue

            genres = str(movie.get('genres_str', '')).split('|')
            # Also handle comma-separated genres
            if len(genres) == 1 and ',' in genres[0]:
                genres = [g.strip() for g in genres[0].split(',')]

            genre_match = sum(1 for g in preferred_genres if g in genres)
            pop_score = float(movie.get('popularity', 0)) / 600.0
            score = 0.6 * (genre_match / max(len(preferred_genres), 1)) + 0.4 * min(pop_score, 1.0)
            results.append({
                'title': movie['title'],
                'year': movie.get('year', ''),
                'genres': movie.get('genres_str', 'Unknown'),
                'language': movie_lang,
                'popularity': float(movie.get('popularity', 0)),
                'ncf_score': 0.0,
                'content_score': score,
                'lang_boost': 0.0,
                'hybrid_score': score,
                'movie_enc': int(float(movie.get('movie_enc', 0))) if pd.notna(movie.get('movie_enc', 0)) else 0,
            })

        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    def get_all_genres(self) -> list:
        """Get all unique genres from the catalog."""
        all_g = set()
        for gs in self.movies['genres_str'].dropna():
            for g in str(gs).split('|'):
                g = g.strip()
                if g and g != 'Unknown':
                    all_g.add(g)
        return sorted(all_g)

    def get_all_languages(self) -> list:
        """Get supported languages — Telugu, Hindi, English only."""
        return ['English', 'Hindi', 'Telugu']


if __name__ == "__main__":
    engine = RecommendationEngine()
    print("\nTesting with user U1...")
    history = engine.get_user_history("U1")
    print(f"Watch history: {len(history)} movies")
    prefs = engine.get_user_preferences("U1")
    print(f"Preferred languages: {prefs['languages'][:3]}")
    print(f"Preferred genres: {prefs['genres'][:5]}")
    recs = engine.recommend("U1", top_k=5)
    print(f"\nTop 5 recommendations:")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r['title']} ({r['language']}) - {r['genres'][:40]} | Score: {r['hybrid_score']:.3f}")
