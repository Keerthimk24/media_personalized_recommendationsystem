import pandas as pd

df = pd.read_csv('movies_real_2000_2026.csv')
print(f"TOTAL MOVIES IN movies_real_2000_2026.csv: {len(df)}")
print()
print("=== LANGUAGE BREAKDOWN ===")
print(df['primary_language'].value_counts().to_string())
print()

# Telugu
te = df[df['primary_language']=='Telugu'][['title','genres_str']].reset_index(drop=True)
print(f"{'='*80}")
print(f" TELUGU MOVIES ({len(te)} total)")
print(f"{'='*80}")
for i, row in te.iterrows():
    t = str(row['title'])[:48]
    g = str(row['genres_str'])
    print(f"  {i+1:3d}. {t:50s} | {g}")

print()

# Hindi
hi = df[df['primary_language']=='Hindi'][['title','genres_str']].reset_index(drop=True)
print(f"{'='*80}")
print(f" HINDI MOVIES ({len(hi)} total)")
print(f"{'='*80}")
for i, row in hi.iterrows():
    t = str(row['title'])[:48]
    g = str(row['genres_str'])
    print(f"  {i+1:3d}. {t:50s} | {g}")

print()

# English (first 50)
en = df[df['primary_language']=='English'][['title','genres_str']].reset_index(drop=True)
print(f"{'='*80}")
print(f" ENGLISH MOVIES ({len(en)} total) - showing first 50")
print(f"{'='*80}")
for i, row in en.head(50).iterrows():
    t = str(row['title'])[:48]
    g = str(row['genres_str'])
    print(f"  {i+1:3d}. {t:50s} | {g}")

# Check for WRONG genres
print()
print(f"{'='*80}")
print(" GENRE PROBLEMS - Random/Wrong genres detected")
print(f"{'='*80}")
# These are known big movies - check if genres make sense
check = {
    'RRR': 'Should be Action, Drama',
    'Pushpa: The Rise': 'Should be Action, Thriller, Drama',
    'Baahubali: The Beginning': 'Should be Action, Drama, Fantasy',
    'Dangal': 'Should be Biography, Drama, Sport',
    'Jawan': 'Should be Action, Thriller',
    'Pathaan': 'Should be Action, Thriller',
    'Animal': 'Should be Action, Crime, Drama',
    'Kalki 2898 AD': 'Should be Action, Sci-Fi',
}
for title, correct in check.items():
    row = df[df['title'] == title]
    if not row.empty:
        actual = row.iloc[0]['genres_str']
        print(f"  {title:40s} | ACTUAL: {actual:40s} | {correct}")
