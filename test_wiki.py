import pandas as pd
import requests
from bs4 import BeautifulSoup

def fetch_test():
    headers = {'User-Agent': 'MovieDataCollector/1.0 (test@example.com)'}
    url = 'https://en.wikipedia.org/wiki/List_of_Telugu_films_of_2023'
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    print(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables):
        try:
            df = pd.read_html(str(table))[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            cols = [str(c).lower().strip() for c in df.columns]
            df.columns = cols
            print(f"\nTable {i} cols:", cols)
            if 'genre' in cols or 'genre(s)' in cols or any('genre' in c for c in cols):
                print(df.head(2))
        except Exception as e:
            print("Error parsing table", i, e)

if __name__ == '__main__':
    fetch_test()
