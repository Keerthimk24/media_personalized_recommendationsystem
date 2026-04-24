"""
Fix user watch histories to match their survey language preferences.
Users who listed Telugu/Hindi favorites should have Telugu/Hindi movies in their history.
"""
import pandas as pd
import numpy as np
import random
random.seed(42)

print("Loading data...")
survey = pd.read_csv('cleaned_survey.csv')
movies = pd.read_csv('movies_clean.csv')
ratings = pd.read_csv('ratings_clean.csv')

# Telugu keywords from survey recent_favorite
TELUGU_KW = [
    'with love', 'salaar', 'salar', 'rrr', 'pushpa', 'dhruva', 'dhurandhar', 'durandhar',
    'baahubali', 'bahubali', 'just married', 'godavari', 'couple friendly', 'kishkindapuri',
    'splitsvilla', '8 vasantalu', 'isha', 'jigris', 'saaiyara', 'rajsaab', 'raj saab',
    'ustad bhagat', 'hit 3', 'hit3', 'dude', 'laapata', 'made in korea', 'naruto',
    'sarvam maya', 'hi nana', 'gandhi talks', 'om shanti', 'og', 'sirai',
    'maa parivar', 'ugadi', 'couple friendly',
]
HINDI_KW = [
    'animal', 'stranger things', 'money heist', 'attack on titan',
    'smile', 'shutter island',
]

telugu_movies = movies[movies['primary_language'] == 'Telugu'].copy()
hindi_movies = movies[movies['primary_language'] == 'Hindi'].copy()
english_movies = movies[movies['primary_language'] == 'English'].copy()

print(f"Telugu movies available: {len(telugu_movies)}")
print(f"Hindi movies available: {len(hindi_movies)}")
print(f"English movies available: {len(english_movies)}")

def detect_lang(row):
    fav = str(row.get('recent_favorite', '')).lower().strip()
    if not fav or fav == 'nan' or fav == '.':
        return 'English'
    for kw in TELUGU_KW:
        if kw in fav:
            return 'Telugu'
    for kw in HINDI_KW:
        if kw in fav:
            return 'Hindi'
    return 'English'

def get_genre_list(genre_str):
    if pd.isna(genre_str):
        return []
    return [g.strip() for g in str(genre_str).replace('|', ',').split(',') if g.strip()]

# For each user, generate watch history matching their language
new_ratings = []
for _, user in survey.iterrows():
    uid = user['user_id']
    lang = detect_lang(user)
    pref_genres = get_genre_list(user.get('preferred_genres', ''))
    
    # Pick movie pool based on detected language
    if lang == 'Telugu':
        pool = telugu_movies
        mix_pool = hindi_movies  # secondary
        mix_ratio = 0.2
    elif lang == 'Hindi':
        pool = hindi_movies
        mix_pool = telugu_movies
        mix_ratio = 0.2
    else:
        pool = english_movies
        mix_pool = pd.DataFrame()
        mix_ratio = 0.0
    
    # Score movies by genre match
    scored = []
    for _, m in pool.iterrows():
        m_genres = get_genre_list(m.get('genres_str', ''))
        match = sum(1 for g in pref_genres if g in m_genres)
        scored.append((m, match))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Pick 30-50 movies for this user
    n_movies = random.randint(30, 50)
    n_primary = int(n_movies * (1 - mix_ratio))
    n_secondary = n_movies - n_primary
    
    # Primary language movies (genre-matched first, then random)
    primary_picks = [s[0] for s in scored[:n_primary]]
    if len(primary_picks) < n_primary:
        extra = pool.sample(min(n_primary - len(primary_picks), len(pool)), random_state=hash(uid) % 10000)
        primary_picks.extend([r for _, r in extra.iterrows()])
    
    # Secondary language movies
    secondary_picks = []
    if not mix_pool.empty and n_secondary > 0:
        sec_sample = mix_pool.sample(min(n_secondary, len(mix_pool)), random_state=hash(uid) % 10000)
        secondary_picks = [r for _, r in sec_sample.iterrows()]
    
    all_picks = primary_picks[:n_primary] + secondary_picks[:n_secondary]
    
    for m in all_picks:
        m_genres = get_genre_list(m.get('genres_str', ''))
        genre_match = sum(1 for g in pref_genres if g in m_genres)
        # Higher rating for genre-matched movies
        base_rating = 3.0 + genre_match * 0.5 + random.uniform(-0.5, 0.5)
        rating = round(min(max(base_rating, 1.0), 5.0), 1)
        
        new_ratings.append({
            'userId': uid,
            'movieId': int(m['movieId']),
            'rating': rating,
        })

new_df = pd.DataFrame(new_ratings)
print(f"\nGenerated {len(new_df)} new ratings")

# Show language distribution per user
for uid in ['U1', 'U2', 'U3', 'U4', 'U5']:
    user_movies = new_df[new_df['userId'] == uid].merge(
        movies[['movieId', 'primary_language']], on='movieId', how='left'
    )
    lang_counts = user_movies['primary_language'].value_counts().to_dict()
    fav = survey[survey['user_id']==uid].iloc[0].get('recent_favorite','?')
    print(f"  {uid} (fav: {str(fav)[:25]:25s}) -> {lang_counts}")

# Combine with existing ratings but remove old entries for users who now have new data
users_with_new = set(new_df['userId'].unique())
old_ratings = ratings[~ratings['userId'].isin(users_with_new)]
combined = pd.concat([old_ratings[['userId', 'movieId', 'rating']], new_df], ignore_index=True)

# Re-encode
from sklearn.preprocessing import LabelEncoder
import pickle

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

all_users = list(set(combined['userId'].unique()))
user_encoder.fit(all_users)
combined['user_enc'] = user_encoder.transform(combined['userId'])

movie_encoder.fit(movies['movieId'])
# Only keep ratings for movies that exist in our catalog
valid_movie_ids = set(movies['movieId'].unique())
combined = combined[combined['movieId'].isin(valid_movie_ids)].copy()
combined['movie_enc'] = movie_encoder.transform(combined['movieId'])

r_min, r_max = combined['rating'].min(), combined['rating'].max()
combined['rating_norm'] = (combined['rating'] - r_min) / (r_max - r_min + 1e-9)

num_users = len(user_encoder.classes_)
num_items = len(movie_encoder.classes_)

print(f"\nFinal: {num_users} users, {num_items} movies, {len(combined)} ratings")

# Save
combined.to_csv('ratings_clean.csv', index=False)
with open('user_encoder.pkl', 'wb') as f:
    pickle.dump(user_encoder, f)
with open('movie_encoder.pkl', 'wb') as f:
    pickle.dump(movie_encoder, f)
meta = {'num_users': num_users, 'num_items': num_items, 'r_min': r_min, 'r_max': r_max}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("\nDone! Now run: python step3_train.py")
