import pandas as pd

df = pd.read_csv('movies_real_2000_2026.csv')

# Find duplicate titles
dupes = df[df.duplicated(subset=['title'], keep=False)].sort_values('title')
print(f"TOTAL MOVIES: {len(df)}")
print(f"UNIQUE TITLES: {df['title'].nunique()}")
print(f"DUPLICATE TITLES: {len(dupes)}")
print()

if len(dupes) > 0:
    print("=" * 100)
    print(" DUPLICATE MOVIES FOUND")
    print("=" * 100)
    for title in dupes['title'].unique():
        rows = df[df['title'] == title]
        print(f"\n  '{title}' appears {len(rows)} times:")
        for _, r in rows.iterrows():
            print(f"    Language: {r['primary_language']:10s} | Genre: {str(r['genres_str']):50s} | Year: {r.get('year','?')}")

# Also check movies_clean.csv for duplicates
print("\n\n" + "=" * 100)
print(" DUPLICATES IN movies_clean.csv")
print("=" * 100)
df2 = pd.read_csv('movies_clean.csv')
dupes2 = df2[df2.duplicated(subset=['title'], keep=False)].sort_values('title')
print(f"TOTAL: {len(df2)}, UNIQUE: {df2['title'].nunique()}, DUPLICATES: {len(dupes2)}")

if len(dupes2) > 0:
    for title in dupes2['title'].unique()[:30]:  # Show first 30
        rows = df2[df2['title'] == title]
        print(f"\n  '{title}' appears {len(rows)} times:")
        for _, r in rows.iterrows():
            print(f"    Language: {r.get('primary_language','?'):10s} | Genre: {str(r.get('genres_str','')):50s}")
