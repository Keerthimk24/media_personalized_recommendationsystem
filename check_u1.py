import pandas as pd
survey = pd.read_csv('cleaned_survey.csv')
u1 = survey[survey['user_id']=='U1']
if not u1.empty:
    r = u1.iloc[0]
    print('U1 Survey:')
    print(f'  Preferred genres: {r.get("preferred_genres","N/A")}')
    print(f'  Recent favorite: {r.get("recent_favorite","N/A")}')
    print(f'  Content type: {r.get("content_type","N/A")}')

ratings = pd.read_csv('ratings_clean.csv')
movies = pd.read_csv('movies_clean.csv')
u1r = ratings[ratings['userId']=='U1']
print(f'\nU1 watch history: {len(u1r)} movies')
merged = u1r.merge(movies[['movieId','title','primary_language','genres_str']], on='movieId', how='left')
print('Language breakdown:')
print(merged['primary_language'].value_counts().to_string())
print('\nTop 10 watched:')
for _,row in merged.sort_values('rating', ascending=False).head(10).iterrows():
    print(f'  {str(row["title"])[:40]:42s} | {row["primary_language"]:10s} | {str(row["genres_str"])[:30]}')
