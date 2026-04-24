"""Fix remaining misclassified movies and rebuild content matrix"""
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

df = pd.read_csv('movies_clean.csv')

# These are English movies wrongly tagged as Hindi
english_fixes = [
    'Close Encounters of the Third Kind',
    'The Best Exotic Marigold Hotel', 
    'Million Dollar Arm',
    'The Hundred-Foot Journey',
]

print("Fixing English movies wrongly tagged as Hindi...")
for title in english_fixes:
    mask = df['title'] == title
    if mask.any():
        df.loc[mask, 'primary_language'] = 'English'
        df.loc[mask, 'content_text'] = (
            df.loc[mask, 'genres_str'].str.replace('|', ' ', regex=False) + 
            ' English English English'
        )
        print(f"  FIXED: '{title}' [Hindi] -> [English]")

# Rebuild content_text for ALL movies to be consistent
print("\nRebuilding content_text for all movies...")
df['content_text'] = (
    df['genres_str'].str.replace('|', ' ', regex=False) + ' ' +
    df['primary_language'] + ' ' +
    df['primary_language'] + ' ' +
    df['primary_language']
)

# Save updated CSV
df.to_csv('movies_clean.csv', index=False)

# Rebuild TF-IDF matrix
print("Rebuilding TF-IDF content matrix...")
df = df.sort_values('movie_enc').reset_index(drop=True)
tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
content_matrix = tfidf.fit_transform(df['content_text'])
print(f"  Content matrix shape: {content_matrix.shape}")

# Save
save_npz('content_matrix.npz', content_matrix)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Final count
print("\nFinal language counts:")
for lang in ['Telugu', 'Hindi', 'English']:
    count = len(df[df['primary_language'] == lang])
    print(f"  {lang}: {count} movies")

print("\nDone! movies_clean.csv + content_matrix.npz updated.")
