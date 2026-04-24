import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import joblib

def merge_and_update():
    print("Loading existing catalog...")
    clean_df = pd.read_csv('movies_clean.csv')
    
    print(f"Original catalog size: {len(clean_df)}")
    
    # Ensure all new movies have necessary columns
    new_df = pd.read_csv('movies_real_2000_2026.csv')
    print(f"New movies to merge: {len(new_df)}")
    
    # Map columns to match clean_df
    # clean_df has: movieId,title,genres,original_language,popularity,primary_language,spoken_languages,genres_list,genres_str,new_movieId
    
    new_df['original_language'] = new_df['primary_language'].apply(lambda x: 'te' if x == 'Telugu' else ('hi' if x == 'Hindi' else 'en'))
    new_df['popularity'] = 20.0  # Give them a baseline popularity
    new_df['spoken_languages'] = new_df['primary_language']
    new_df['genres'] = new_df['genres_str']
    
    # Append to clean_df
    combined_df = pd.concat([clean_df, new_df], ignore_index=True)
    
    # Drop duplicates by title (keep the first, which will be the original if it exists, or update it)
    combined_df = combined_df.drop_duplicates(subset=['title'], keep='last')
    
    # Generate new sequential new_movieId to ensure matrix indices align
    combined_df = combined_df.reset_index(drop=True)
    combined_df['new_movieId'] = combined_df.index
    
    print(f"Merged catalog size: {len(combined_df)}")
    
    # Rebuild TF-IDF Matrix for content-based filtering
    print("Rebuilding TF-IDF Matrix...")
    tfidf = TfidfVectorizer(stop_words='english')
    combined_df['genres_str'] = combined_df['genres_str'].fillna('')
    content_matrix = tfidf.fit_transform(combined_df['genres_str'])
    
    # Save everything
    combined_df.to_csv('movies_clean.csv', index=False)
    scipy.sparse.save_npz('content_matrix.npz', content_matrix)
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    
    print("Successfully updated movies_clean.csv, content_matrix.npz, and tfidf_vectorizer.pkl")

if __name__ == '__main__':
    merge_and_update()
