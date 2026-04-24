import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random

def fetch_titles(language, year):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f'https://en.wikipedia.org/wiki/List_of_{language}_films_of_{year}'
    movies = []
    
    try:
        r = requests.get(url, headers=headers)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table', {'class': 'wikitable'})
        
        for table in tables:
            df = pd.read_html(str(table))[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            cols = [str(c).lower().strip() for c in df.columns]
            df.columns = cols
            
            title_col = None
            for c in cols:
                if 'title' in c: title_col = c
                
            if title_col:
                for _, row in df.iterrows():
                    title = str(row[title_col])
                    if title != 'nan' and len(title) > 1 and title.lower() != 'title':
                        # Clean up title citations like [1]
                        import re
                        title = re.sub(r'\[.*?\]', '', title).strip()
                        movies.append(title)
    except Exception as e:
        pass
    return list(set(movies)) # deduplicate

def test_fetch():
    telugu_2023 = fetch_titles('Telugu', 2023)
    print(f"Found {len(telugu_2023)} real Telugu movies in 2023.")
    print(telugu_2023[:10])

if __name__ == '__main__':
    test_fetch()
