import pandas as pd

df = pd.read_csv('movies_clean.csv')

# These are ACTION movies wrongly tagged with Romance
action_fixes = {
    'Okkadu': 'Action|Drama',
    'Athadu': 'Action|Thriller',
    'Pokiri': 'Action|Thriller',
    'Khaleja': 'Action|Comedy',
    'Temper': 'Action|Drama',
    'Simhadri': 'Action|Drama',
    'Chatrapathi': 'Action|Drama',
    'Khaidi': 'Action|Thriller',
    'Magadheera': 'Action|Fantasy|Drama',
    '1 - Nenokkadine': 'Action|Thriller',
    'Indra': 'Action|Drama',
    'Tagore': 'Action|Drama',
    'Businessman': 'Action|Crime',
    'Gabbar Singh': 'Action|Comedy',
    'Dookudu': 'Action|Comedy',
    'Julayi': 'Action|Comedy|Thriller',
    'Kick': 'Action|Comedy',
    'Sarrainodu': 'Action|Drama',
    'Krack': 'Action|Thriller',
    'Vikramarkudu': 'Action|Comedy',
    'Bhadra': 'Action|Drama',
    'Aagadu': 'Action|Comedy',
    'Mirchi': 'Action|Romance|Drama',
    'Rebel': 'Action|Drama',
    'Race Gurram': 'Action|Comedy',
    'Balupu': 'Action|Comedy',
    'Mirapakay': 'Action|Comedy',
    'Don Seenu': 'Action|Comedy',
    'Waltair Veerayya': 'Action|Comedy',
    'Veera Simha Reddy': 'Action|Drama',
    'Akhanda': 'Action|Drama',
    'Sye Raa Narasimha Reddy': 'Action|Drama|History',
    'Khaidi No. 150': 'Action|Comedy',
    'Game Changer': 'Action|Drama',
    'Guntur Kaaram': 'Action|Drama',
    'Aravinda Sametha Veera Raghava': 'Action|Drama',
    'Janatha Garage': 'Action|Drama',
    'Agent': 'Action|Thriller',
    'Skanda': 'Action|Drama',
    # Telugu Romance movies - correct tags
    'Geetha Govindam': 'Romance|Comedy',
    'Arjun Reddy': 'Drama|Romance',
    'Dear Comrade': 'Drama|Romance|Action',
    'Preminchukundam Raa': 'Romance|Drama',
    'Nuvvostanante Nenoddantana': 'Romance|Drama',
    'Bommarillu': 'Romance|Comedy|Drama',
    'Tholi Prema': 'Romance|Drama',
    'Love Story': 'Romance|Drama',
    'Pelli Choopulu': 'Romance|Comedy|Drama',
    'Ante Sundaraniki': 'Romance|Comedy',
    'Sita Ramam': 'Romance|Drama|War',
    'Hi Nanna': 'Drama|Family',
    'Majili': 'Romance|Drama',
    'Most Eligible Bachelor': 'Romance|Comedy',
    'Mr. Majnu': 'Romance|Comedy',
    '100% Love': 'Romance|Comedy',
    'Jaanu': 'Romance|Drama',
    'Premam': 'Romance|Drama',
    'Ninnu Kori': 'Romance|Drama',
    'Padi Padi Leche Manasu': 'Romance|Drama',
    'A Aa': 'Romance|Comedy|Drama',
    # Telugu Comedy
    'DJ Tillu': 'Comedy|Crime',
    'Tillu Square': 'Comedy|Crime',
    'Jathi Ratnalu': 'Comedy',
    'F2: Fun and Frustration': 'Comedy|Family',
    'F3: Fun and Frustration': 'Comedy|Family',
    'Bhale Bhale Magadivoy': 'Comedy|Romance',
    # Telugu Thriller/Crime
    'Evaru': 'Thriller|Crime',
    'Goodachari': 'Action|Thriller',
    'Kshanam': 'Thriller',
    'Rakshasudu': 'Thriller|Crime',
    'Karthikeya 2': 'Adventure|Mystery|Thriller',
    'Karthikeya': 'Mystery|Thriller',
    'Virupaksha': 'Thriller|Mystery|Horror',
    'HIT: The Second Case': 'Thriller|Crime',
    'Masooda': 'Horror|Thriller',
    # Telugu Drama
    'Dasara': 'Action|Drama',
    'Balagam': 'Drama|Family',
    'Jersey': 'Drama|Sport',
    'Mahanati': 'Biography|Drama',
    'Rangasthalam': 'Action|Drama',
    'Maharshi': 'Action|Drama',
    'Srimanthudu': 'Action|Drama',
    'Bharat Ane Nenu': 'Drama|Action',
    'Leader': 'Drama',
    'Major': 'Action|Biography|Drama',
    # Telugu Fantasy/Sci-Fi
    'Eega': 'Action|Comedy|Fantasy',
    'Hanuman': 'Action|Adventure|Fantasy',
    'Bimbisara': 'Action|Fantasy',
    'Kalki 2898 AD': 'Action|Sci-Fi|Fantasy',
    'Baahubali: The Beginning': 'Action|Drama|Fantasy',
    'Baahubali 2: The Conclusion': 'Action|Drama|Fantasy',
    # Hindi fixes
    'Dangal': 'Biography|Drama|Sport',
    'PK': 'Comedy|Drama|Sci-Fi',
    'Sanju': 'Biography|Drama',
    'Jawan': 'Action|Thriller',
    'Pathaan': 'Action|Thriller',
    'Animal': 'Action|Crime|Drama',
    'Fighter': 'Action|Thriller',
    'Kabir Singh': 'Drama|Romance',
    'Gully Boy': 'Drama|Music',
    'Uri: The Surgical Strike': 'Action|Drama|War',
    'Stree': 'Comedy|Horror',
    'Andhadhun': 'Crime|Thriller|Comedy',
    'Shershaah': 'Action|Biography|Drama',
    'The Kashmir Files': 'Drama',
    'Drishyam 2': 'Crime|Drama|Thriller',
    'Bhool Bhulaiyaa 2': 'Comedy|Horror',
    'Brahmastra Part One: Shiva': 'Action|Adventure|Fantasy',
    'Dunki': 'Comedy|Drama',
    'Shaitaan': 'Horror|Thriller',
    'Rocky Aur Rani Kii Prem Kahaani': 'Comedy|Drama|Romance',
    '3 Idiots': 'Comedy|Drama',
    'Bajrangi Bhaijaan': 'Action|Adventure|Drama',
    'Sultan': 'Action|Drama|Sport',
    'Tiger Zinda Hai': 'Action|Thriller',
    'War': 'Action|Thriller',
    'Sooryavanshi': 'Action|Thriller',
    'Gadar 2': 'Action|Drama',
    'Crew': 'Comedy',
    'Ludo': 'Comedy|Crime|Drama',
    'Thappad': 'Drama',
    '83': 'Biography|Drama|Sport',
    'Raazi': 'Action|Thriller',
    'Badhaai Ho': 'Comedy|Drama',
    'Padmaavat': 'Drama|History|Romance',
    'Bajirao Mastani': 'Drama|History|Romance',
    'Tanhaji: The Unsung Warrior': 'Action|Drama|History',
    'Mission Mangal': 'Drama',
    'Good Newwz': 'Comedy|Drama',
    'Super 30': 'Biography|Drama',
    'Simmba': 'Action|Comedy',
    'Chhichhore': 'Comedy|Drama',
    'Golmaal Again': 'Comedy',
    'Article 15': 'Crime|Drama|Thriller',
    'Dream Girl': 'Comedy',
    'Bala': 'Comedy|Drama',
    'Luka Chuppi': 'Comedy|Romance',
    'Kedarnath': 'Drama|Romance',
    'Piku': 'Comedy|Drama',
    'Queen': 'Comedy|Drama',
    'Barfi!': 'Comedy|Drama|Romance',
    'Kahaani': 'Crime|Thriller',
    'English Vinglish': 'Comedy|Drama',
    'Rockstar': 'Drama|Music|Romance',
    'My Name Is Khan': 'Drama',
    'Dabangg': 'Action|Comedy',
    'Chak De! India': 'Drama|Sport',
    'Taare Zameen Par': 'Drama|Family',
    'Jab We Met': 'Comedy|Romance',
    'Om Shanti Om': 'Action|Comedy|Romance',
    'Rang De Basanti': 'Drama',
    'Lagaan': 'Drama|Sport',
    'Dil Chahta Hai': 'Comedy|Drama|Romance',
    'Dilwale Dulhania Le Jayenge': 'Drama|Romance',
    'Devdas': 'Drama|Romance|Music',
    'Kuch Kuch Hota Hai': 'Drama|Romance',
    'Yeh Jawaani Hai Deewani': 'Comedy|Drama|Romance',
    'Raanjhanaa': 'Drama|Romance',
    'Fukrey': 'Comedy',
    'Student of the Year': 'Comedy|Drama|Romance',
    'Agneepath': 'Action|Crime|Drama',
    'Rowdy Rathore': 'Action|Comedy',
    'Housefull 2': 'Comedy',
    'Dabangg 2': 'Action|Comedy',
    'Race 2': 'Action|Thriller',
    'Special 26': 'Crime|Thriller',
    'Ek Tha Tiger': 'Action|Romance|Thriller',
    'Cocktail': 'Comedy|Drama|Romance',
    'Shuddh Desi Romance': 'Comedy|Drama|Romance',
    'Lootera': 'Drama|Romance',
    'Grand Masti': 'Comedy',
    'Hasee Toh Phasee': 'Comedy|Drama|Romance',
    'Jai Ho': 'Action|Drama',
    'Aashiqui 2': 'Drama|Music|Romance',
    'Highway': 'Drama|Romance',
    'Haider': 'Action|Crime|Drama',
    'Talaash': 'Crime|Drama|Thriller',
    'Son of Sardaar': 'Action|Comedy',
    'OMG: Oh My God!': 'Comedy|Drama',
    'Ishaqzaade': 'Action|Drama|Romance',
    'Paan Singh Tomar': 'Action|Biography|Drama',
    'Ram-Leela': 'Drama|Romance',
}

count = 0
for title, correct_genre in action_fixes.items():
    mask = df['title'] == title
    if mask.any():
        df.loc[mask, 'genres_str'] = correct_genre
        count += 1

print(f"Fixed genres for {count} movies")

# Verify
for t in ['Okkadu', 'Pokiri', 'Geetha Govindam', 'Dangal', 'Hasee Toh Phasee']:
    row = df[df['title'] == t]
    if not row.empty:
        print(f"  {t:30s} -> {row.iloc[0]['genres_str']}")

df.to_csv('movies_clean.csv', index=False)

# Rebuild TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import pickle

df['content_text'] = (
    df['genres_str'].str.replace('|', ' ', regex=False) + ' ' +
    df['primary_language'].fillna('English') + ' ' +
    df['primary_language'].fillna('English') + ' ' +
    df['primary_language'].fillna('English')
)
tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
content_matrix = tfidf.fit_transform(df['content_text'])
save_npz('content_matrix.npz', content_matrix)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print(f"Content matrix rebuilt: {content_matrix.shape}")
print("DONE - genres fixed!")
