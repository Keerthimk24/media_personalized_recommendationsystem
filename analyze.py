import pandas as pd
import json

def analyze_genres():
    # Load movies clean to get the current catalog's primary language and genres
    clean_df = pd.read_csv('movies_clean.csv')
    clean_df['movieId'] = clean_df['movieId'].astype(str)
    
    # Load metadata to get release dates
    meta_df = pd.read_csv('../DLCASE/movies_metadata.csv', low_memory=False)
    meta_df['id'] = pd.to_numeric(meta_df['id'], errors='coerce')
    meta_df = meta_df.dropna(subset=['id'])
    meta_df['id'] = meta_df['id'].astype(int).astype(str)
    
    # Merge
    merged = pd.merge(clean_df, meta_df[['id', 'release_date']], left_on='movieId', right_on='id', how='left')
    
    # Extract year
    merged['release_date'] = pd.to_datetime(merged['release_date'], errors='coerce')
    merged['year'] = merged['release_date'].dt.year
    
    target_langs = ['Telugu', 'Hindi', 'English']
    recent = merged[merged['primary_language'].isin(target_langs)]
    
    results = {}
    
    for lang in target_langs:
        lang_df = recent[recent['primary_language'] == lang]
        
        # Calculate year distribution
        years = lang_df['year'].dropna().astype(int)
        year_dist = {
            'min_year': int(years.min()) if not years.empty else None,
            'max_year': int(years.max()) if not years.empty else None,
            'recent_2010_2020': int(len(years[(years >= 2010) & (years <= 2020)])),
            'movies_2020_2026': int(len(years[(years >= 2020) & (years <= 2026)])),
        }
        
        genre_counts = {}
        for genres in lang_df['genres_str'].dropna():
            g_list = [g.strip() for g in genres.split(',')]
            for g in g_list:
                genre_counts[g] = genre_counts.get(g, 0) + 1
                
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        results[lang] = {
            'total_movies_in_dataset': len(lang_df),
            'year_stats': year_dist,
            'top_10_genres': sorted_genres[:10]
        }
        
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    analyze_genres()
