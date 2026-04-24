"""
Step 1: Data Preprocessing
--------------------------
- Loads DLDATA.csv (survey data) and movies.csv + ratings.csv (from DLCASE)
- Renames long feature names to short, clean names
- Maps languages: English, Hindi, Telugu
- Encodes genres: Action, Romance, Thriller, Drama, Family, Sci-Fi, Horror, Comedy
- Builds a clean dataset ready for deep learning
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

DLCASE = r"c:\Users\navee\OneDrive\Desktop\DLCASE"

def load_and_rename_survey():
    """Load DLDATA.csv and rename the ridiculously long column names."""
    df = pd.read_csv("DLDATA.csv")
    
    rename_map = {
        'user_id': 'user_id',
        'content_id': 'content_id',
        'Timestamp': 'timestamp',
        'Which genres do you enjoy the most? (Select up to 3)': 'preferred_genres',
        'What type of content do you prefer?': 'content_type',
        'When do you usually watch content? (Select all that apply)': 'watch_time',
        'What is your preferred episode length?': 'episode_length',
        'How do you usually find new content to watch?': 'discovery_method',
        'How often do you re-watch your favorite shows or movies?': 'rewatch_freq',
        'Which device do you primarily use for streaming?': 'device',
        'How satisfied are you with the current content recommendations?': 'satisfaction',
        'Based on your preferred content type, please share the name of a recent favorite you watched (e.g., a movie, TV show, or short film).': 'recent_favorite',
        'rating': 'rating'
    }
    
    df = df.rename(columns=rename_map)
    print(f"  Survey data loaded: {len(df)} users")
    print(f"  Columns after renaming: {list(df.columns)}")
    return df


def load_movies():
    """Load movies.csv from DLCASE."""
    path = os.path.join(DLCASE, "movies.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=['movieId', 'title'])
    df['movieId'] = df['movieId'].astype(int)
    
    # --- Language Mapping (ISO codes -> Full Names) ---
    lang_map = {
        'en': 'English', 'hi': 'Hindi', 'te': 'Telugu',
        'ta': 'Tamil', 'ml': 'Malayalam', 'kn': 'Kannada',
        'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
        'cn': 'Chinese', 'ru': 'Russian', 'it': 'Italian',
        'pt': 'Portuguese', 'ar': 'Arabic', 'nl': 'Dutch',
        'sv': 'Swedish', 'no': 'Norwegian', 'da': 'Danish',
        'pl': 'Polish', 'ro': 'Romanian', 'hu': 'Hungarian',
        'cs': 'Czech', 'el': 'Greek', 'he': 'Hebrew',
        'th': 'Thai', 'vi': 'Vietnamese', 'uk': 'Ukrainian',
        'bg': 'Bulgarian', 'la': 'Latin', 'eo': 'Esperanto',
        'gd': 'Scottish Gaelic', 'ga': 'Irish', 'gl': 'Galician',
        'eu': 'Basque', 'si': 'Sinhala', 'bn': 'Bengali',
        'ur': 'Urdu', 'ne': 'Nepali', 'my': 'Burmese',
        'km': 'Khmer', 'bo': 'Tibetan', 'sq': 'Albanian',
        'sw': 'Swahili', 'yi': 'Yiddish', 'ja': 'Japanese',
    }
    
    def get_primary_language(lang_str):
        """Get the primary (first) language as full name."""
        if pd.isna(lang_str):
            return 'English'  # default
        parts = str(lang_str).split('|')
        first = parts[0].strip().lower()
        return lang_map.get(first, first.upper())
    
    def get_all_languages(lang_str):
        """Get all languages as full names."""
        if pd.isna(lang_str):
            return 'English'
        parts = str(lang_str).split('|')
        mapped = [lang_map.get(p.strip().lower(), p.strip().upper()) for p in parts]
        return '|'.join(mapped)
    
    df['primary_language'] = df['language_str'].apply(get_primary_language)
    df['all_languages'] = df['language_str'].apply(get_all_languages)
    
    # Clean genres
    df['genres_str'] = df['genres_str'].fillna('Unknown')
    df['genres_list'] = df['genres_str'].str.split('|')
    
    # Create combined content text for TF-IDF similarity
    # Language is repeated 3x to boost its weight in similarity scoring
    df['content_text'] = (
        df['genres_str'].str.replace('|', ' ', regex=False) + ' ' +
        df['primary_language'] + ' ' +
        df['primary_language'] + ' ' +
        df['primary_language']
    )
    
    print(f"  Movies loaded: {len(df)} movies")
    print(f"  Languages found: {df['primary_language'].nunique()} unique")
    print(f"  Top 3 languages: {df['primary_language'].value_counts().head(3).to_dict()}")
    return df


def load_ratings():
    """Load ratings.csv from DLCASE."""
    path = os.path.join(DLCASE, "ratings.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=['userId', 'movieId', 'rating'])
    df['movieId'] = df['movieId'].astype(int)
    print(f"  Ratings loaded: {len(df)} interactions")
    return df


def main():
    print("=" * 60)
    print("🚀 STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    
    # --- Load all datasets ---
    print("\n[1/6] Loading survey data (DLDATA.csv)...")
    survey_df = load_and_rename_survey()
    
    print("\n[2/6] Loading movies catalog...")
    movies_df = load_movies()
    
    print("\n[3/6] Loading ratings/interactions...")
    ratings_df = load_ratings()
    
    # --- Filter ratings to valid movies only ---
    valid_ids = set(movies_df['movieId'].unique())
    ratings_df = ratings_df[ratings_df['movieId'].isin(valid_ids)].copy()
    print(f"  Valid ratings after filtering: {len(ratings_df)}")
    
    # --- Encode Users and Movies ---
    print("\n[4/6] Encoding users and movies for neural network...")
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    # Fit user encoder on ALL users from both survey and ratings
    all_users = list(set(
        list(ratings_df['userId'].unique()) + 
        list(survey_df['user_id'].unique())
    ))
    user_encoder.fit(all_users)
    ratings_df['user_enc'] = user_encoder.transform(ratings_df['userId'])
    
    # Fit movie encoder on ALL movies
    movie_encoder.fit(movies_df['movieId'])
    ratings_df['movie_enc'] = movie_encoder.transform(ratings_df['movieId'])
    movies_df['movie_enc'] = movie_encoder.transform(movies_df['movieId'])
    
    # Normalize ratings to 0-1 for sigmoid output
    r_min, r_max = ratings_df['rating'].min(), ratings_df['rating'].max()
    ratings_df['rating_norm'] = (ratings_df['rating'] - r_min) / (r_max - r_min + 1e-9)
    
    num_users = len(user_encoder.classes_)
    num_items = len(movie_encoder.classes_)
    print(f"  Encoded Users: {num_users}")
    print(f"  Encoded Movies: {num_items}")
    
    # --- Build TF-IDF Content Matrix ---
    print("\n[5/6] Building TF-IDF content similarity matrix...")
    # Sort movies by encoded ID so matrix index = encoded ID
    movies_df = movies_df.sort_values('movie_enc').reset_index(drop=True)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    content_matrix = tfidf.fit_transform(movies_df['content_text'])
    print(f"  Content matrix shape: {content_matrix.shape}")
    
    # --- Save Everything ---
    print("\n[6/6] Saving all preprocessed data...")
    
    # Clean survey data
    survey_df.to_csv("cleaned_survey.csv", index=False)
    
    # Movies and ratings
    movies_df.to_csv("movies_clean.csv", index=False)
    ratings_df.to_csv("ratings_clean.csv", index=False)
    
    # Encoders
    with open("user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)
    with open("movie_encoder.pkl", "wb") as f:
        pickle.dump(movie_encoder, f)
    
    # TF-IDF artifacts
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    save_npz("content_matrix.npz", content_matrix)
    
    # Save metadata
    meta = {
        'num_users': num_users,
        'num_items': num_items,
        'r_min': r_min,
        'r_max': r_max
    }
    with open("meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\nFiles saved:")
    print("  📄 cleaned_survey.csv    - Survey data with clean column names")
    print("  📄 movies_clean.csv      - Movies with language & genre tags")
    print("  📄 ratings_clean.csv     - User-movie interactions (encoded)")
    print("  📦 user_encoder.pkl      - User ID encoder")
    print("  📦 movie_encoder.pkl     - Movie ID encoder")
    print("  📦 tfidf_vectorizer.pkl  - TF-IDF model for content similarity")
    print("  📦 content_matrix.npz    - Precomputed content vectors")
    print("  📦 meta.pkl              - Metadata (counts, normalization)")


if __name__ == "__main__":
    main()
