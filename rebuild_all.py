"""
Full Pipeline Rebuild: Merges fixed Telugu/Hindi movies, rebuilds ALL artifacts, retrains model.
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

DLCASE = r"c:\Users\navee\OneDrive\Desktop\DLCASE"

print("=" * 60)
print("FULL PIPELINE REBUILD")
print("=" * 60)

# ── 1. Load original DLCASE movies ──
print("\n[1/8] Loading original DLCASE movies...")
movies_df = pd.read_csv(os.path.join(DLCASE, "movies.csv"))
movies_df = movies_df.dropna(subset=['movieId', 'title'])
movies_df['movieId'] = movies_df['movieId'].astype(int)

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
    'sw': 'Swahili', 'bn': 'Bengali', 'ur': 'Urdu',
}

def get_primary_language(lang_str):
    if pd.isna(lang_str):
        return 'English'
    first = str(lang_str).split('|')[0].strip().lower()
    return lang_map.get(first, first.upper())

movies_df['primary_language'] = movies_df['language_str'].apply(get_primary_language)
movies_df['genres_str'] = movies_df['genres_str'].fillna('Unknown')
print(f"  Original DLCASE movies: {len(movies_df)}")

# ── 2. Load fixed Telugu/Hindi movies ──
print("\n[2/8] Loading fixed Telugu/Hindi movies from movies_real_2000_2026.csv...")
new_df = pd.read_csv("movies_real_2000_2026.csv")
# Only take Telugu and Hindi from the new file (English already in DLCASE)
new_indian = new_df[new_df['primary_language'].isin(['Telugu', 'Hindi'])].copy()
print(f"  New Indian movies to merge: {len(new_indian)}")
print(f"    Telugu: {len(new_indian[new_indian['primary_language']=='Telugu'])}")
print(f"    Hindi:  {len(new_indian[new_indian['primary_language']=='Hindi'])}")

# Prepare columns
new_indian['movieId'] = range(9000000, 9000000 + len(new_indian))
new_indian['original_language'] = new_indian['primary_language'].map(
    {'Telugu': 'te', 'Hindi': 'hi'}
)
if 'popularity' not in new_indian.columns:
    new_indian['popularity'] = 20.0
new_indian['spoken_languages'] = new_indian['primary_language']
new_indian['genres_str'] = new_indian['genres_str'].fillna('Drama')
# Normalize genre separator to pipe
new_indian['genres_str'] = new_indian['genres_str'].str.replace(', ', '|')

# ── 3. Merge ──
print("\n[3/8] Merging catalogs...")
combined = pd.concat([movies_df, new_indian], ignore_index=True)
combined = combined.drop_duplicates(subset=['title'], keep='last')
combined = combined.reset_index(drop=True)
combined['genres_str'] = combined['genres_str'].fillna('Unknown')
combined['genres_list'] = combined['genres_str'].str.split('|')
print(f"  Merged catalog: {len(combined)} movies")
print(f"  Telugu: {len(combined[combined['primary_language']=='Telugu'])}")
print(f"  Hindi:  {len(combined[combined['primary_language']=='Hindi'])}")
print(f"  English: {len(combined[combined['primary_language']=='English'])}")

# ── 4. Load ratings ──
print("\n[4/8] Loading ratings...")
ratings_df = pd.read_csv(os.path.join(DLCASE, "ratings.csv"))
ratings_df = ratings_df.dropna(subset=['userId', 'movieId', 'rating'])
ratings_df['movieId'] = ratings_df['movieId'].astype(int)
valid_ids = set(combined['movieId'].unique())
ratings_df = ratings_df[ratings_df['movieId'].isin(valid_ids)].copy()
print(f"  Valid ratings: {len(ratings_df)}")

# ── 5. Encode users and movies ──
print("\n[5/8] Encoding users and movies...")
survey_df = pd.read_csv("DLDATA.csv")
rename_map = {
    'user_id': 'user_id', 'content_id': 'content_id', 'Timestamp': 'timestamp',
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
survey_df = survey_df.rename(columns=rename_map)
survey_df.to_csv("cleaned_survey.csv", index=False)

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

all_users = list(set(
    list(ratings_df['userId'].unique()) +
    list(survey_df['user_id'].unique())
))
user_encoder.fit(all_users)
ratings_df['user_enc'] = user_encoder.transform(ratings_df['userId'])

movie_encoder.fit(combined['movieId'])
ratings_df['movie_enc'] = movie_encoder.transform(ratings_df['movieId'])
combined['movie_enc'] = movie_encoder.transform(combined['movieId'])

r_min, r_max = ratings_df['rating'].min(), ratings_df['rating'].max()
ratings_df['rating_norm'] = (ratings_df['rating'] - r_min) / (r_max - r_min + 1e-9)

num_users = len(user_encoder.classes_)
num_items = len(movie_encoder.classes_)
print(f"  Users: {num_users}, Movies: {num_items}")

# ── 6. Build TF-IDF content matrix ──
print("\n[6/8] Building TF-IDF content matrix...")
combined = combined.sort_values('movie_enc').reset_index(drop=True)
combined['content_text'] = (
    combined['genres_str'].str.replace('|', ' ', regex=False) + ' ' +
    combined['primary_language'] + ' ' +
    combined['primary_language'] + ' ' +
    combined['primary_language']
)
tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
content_matrix = tfidf.fit_transform(combined['content_text'])
print(f"  Content matrix shape: {content_matrix.shape}")

# ── 7. Save everything ──
print("\n[7/8] Saving all artifacts...")
combined.to_csv("movies_clean.csv", index=False)
ratings_df.to_csv("ratings_clean.csv", index=False)

with open("user_encoder.pkl", "wb") as f:
    pickle.dump(user_encoder, f)
with open("movie_encoder.pkl", "wb") as f:
    pickle.dump(movie_encoder, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
save_npz("content_matrix.npz", content_matrix)

meta = {'num_users': num_users, 'num_items': num_items, 'r_min': r_min, 'r_max': r_max}
with open("meta.pkl", "wb") as f:
    pickle.dump(meta, f)

# ── 8. Train NCF Model ──
print("\n[8/8] Training NCF model...")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from step2_model import build_ncf
from sklearn.model_selection import train_test_split

X_user = ratings_df['user_enc'].values
X_item = ratings_df['movie_enc'].values
y = ratings_df['rating_norm'].values.astype(np.float32)

Xu_tr, Xu_te, Xi_tr, Xi_te, y_tr, y_te = train_test_split(
    X_user, X_item, y, test_size=0.2, random_state=42
)

model = build_ncf(num_users, num_items, embed_dim=64)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, verbose=1
    )
]

model.fit(
    [Xu_tr, Xi_tr], y_tr,
    validation_data=([Xu_te, Xi_te], y_te),
    epochs=30, batch_size=256,
    callbacks=callbacks, verbose=1
)

model.save("ncf_model.h5")

# ── Verify ──
print("\n" + "=" * 60)
print("REBUILD COMPLETE!")
print("=" * 60)
print(f"  Movies: {num_items} (Telugu: {len(combined[combined['primary_language']=='Telugu'])}, "
      f"Hindi: {len(combined[combined['primary_language']=='Hindi'])}, "
      f"English: {len(combined[combined['primary_language']=='English'])})")
print(f"  Users: {num_users}")
print(f"  Ratings: {len(ratings_df)}")

# Quick test
print("\nQuick genre check:")
for title in ['RRR', 'Pushpa: The Rise', 'Dangal', 'Pathaan', 'Baahubali: The Beginning']:
    row = combined[combined['title'] == title]
    if not row.empty:
        r = row.iloc[0]
        print(f"  {title:40s} | {r['primary_language']:8s} | {r['genres_str']}")
