import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import time

def fetch_wiki_genres(language, years):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_genres = {}
    total_movies = 0
    
    for year in years:
        if language == 'English':
            url = f'https://en.wikipedia.org/wiki/List_of_American_films_of_{year}'
        else:
            url = f'https://en.wikipedia.org/wiki/List_of_{language}_films_of_{year}'
            
        print(f"Fetching {url}")
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to fetch {url} - Status: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            for table in tables:
                try:
                    df = pd.read_html(str(table))[0]
                    # Find genre column
                    genre_col = None
                    for col in df.columns:
                        if isinstance(col, tuple):
                            col_name = col[-1].lower()
                        else:
                            col_name = str(col).lower()
                            
                        if 'genre' in col_name:
                            genre_col = col
                            break
                            
                    if genre_col is not None:
                        genres = df[genre_col].dropna().astype(str).tolist()
                        for g in genres:
                            # clean up citations like [1], clean up multiple genres
                            import re
                            g = re.sub(r'\[.*?\]', '', g)
                            parts = re.split(r',|/| and |-', g)
                            for p in parts:
                                p = p.strip().title()
                                if p and len(p) > 2 and p != 'Nan':
                                    all_genres[p] = all_genres.get(p, 0) + 1
                                    total_movies += 1
                except Exception as e:
                    # Ignore tables that fail parsing
                    pass
            time.sleep(1) # politeness delay
        except Exception as e:
            print(f"Error processing {url}: {e}")
            
    sorted_genres = sorted(all_genres.items(), key=lambda x: x[1], reverse=True)
    return total_movies, sorted_genres[:15]

def main():
    years = list(range(2020, 2027))
    results = {}
    
    for lang in ['Telugu', 'Hindi', 'English']:
        print(f"Processing {lang}...")
        total, top_genres = fetch_wiki_genres(lang, years)
        results[lang] = {
            'estimated_movies_processed': total,
            'top_genres': top_genres
        }
        
    with open('wiki_genres_2020_2026.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Done! Check wiki_genres_2020_2026.json")

if __name__ == '__main__':
    main()
