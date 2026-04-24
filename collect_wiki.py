import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os

def fetch_wiki_movies(language, years):
    headers = {
        'User-Agent': 'MovieDataCollector/1.0 (studying_genres@example.com)'
    }
    
    all_movies = []
    
    for year in years:
        if language == 'English':
            url = f'https://en.wikipedia.org/wiki/List_of_American_films_of_{year}'
        else:
            url = f'https://en.wikipedia.org/wiki/List_of_{language}_films_of_{year}'
            
        print(f"Fetching {url}")
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            for table in tables:
                try:
                    df = pd.read_html(str(table))[0]
                    # Flatten multi-index columns if any
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(-1)
                    
                    df.columns = [str(c).lower().strip() for c in df.columns]
                    
                    title_col = None
                    genre_col = None
                    
                    for c in df.columns:
                        if 'title' in c: title_col = c
                        if 'genre' in c: genre_col = c
                        
                    if title_col and genre_col:
                        for _, row in df.iterrows():
                            title = str(row[title_col])
                            genre = str(row[genre_col])
                            if title != 'nan' and genre != 'nan':
                                all_movies.append({
                                    'title': title,
                                    'primary_language': language,
                                    'year': year,
                                    'genres_str': genre
                                })
                except Exception as e:
                    pass
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            
    return all_movies

def main():
    years = list(range(2020, 2027))
    all_data = []
    
    for lang in ['Telugu', 'Hindi', 'English']:
        movies = fetch_wiki_movies(lang, years)
        all_data.extend(movies)
        
    df = pd.DataFrame(all_data)
    
    # clean genres
    import re
    def clean_genre(g):
        g = re.sub(r'\[.*?\]', '', g)
        parts = re.split(r',|/| and |-', g)
        cleaned = []
        for p in parts:
            p = p.strip().title()
            if p and len(p) > 2:
                cleaned.append(p)
        return "|".join(cleaned)
        
    df['genres_list'] = df['genres_str'].apply(clean_genre)
    # Filter out bad rows
    df = df[df['genres_list'] != '']
    
    df.to_csv('movies_recent_2020_2026.csv', index=False)
    print(f"Saved {len(df)} movies to movies_recent_2020_2026.csv")
    
    # Analyze
    for lang in ['Telugu', 'Hindi', 'English']:
        lang_df = df[df['primary_language'] == lang]
        all_g = []
        for gl in lang_df['genres_list']:
            all_g.extend(gl.split('|'))
        from collections import Counter
        top = Counter(all_g).most_common(10)
        print(f"\n{lang} Top Genres (Total {len(lang_df)} movies):")
        for g, count in top:
            print(f"  {g}: {count}")

if __name__ == '__main__':
    main()
