import pandas as pd
import random

def generate_large_movies_dataset():
    # 1. Start with the curated real movies so the best ones are real
    recent_movies = [
        # Telugu
        {"title": "Kalki 2898 AD", "primary_language": "Telugu", "year": 2024, "genres_str": "Action|Sci-Fi|Fantasy"},
        {"title": "Salaar: Part 1 - Ceasefire", "primary_language": "Telugu", "year": 2023, "genres_str": "Action|Crime|Thriller"},
        {"title": "RRR", "primary_language": "Telugu", "year": 2022, "genres_str": "Action|Drama|History"},
        {"title": "Pushpa: The Rise", "primary_language": "Telugu", "year": 2021, "genres_str": "Action|Crime|Drama"},
        {"title": "Pushpa 2: The Rule", "primary_language": "Telugu", "year": 2024, "genres_str": "Action|Crime|Thriller"},
        {"title": "Guntur Kaaram", "primary_language": "Telugu", "year": 2024, "genres_str": "Action|Drama|Comedy"},
        {"title": "Hi Nanna", "primary_language": "Telugu", "year": 2023, "genres_str": "Romance|Drama|Family"},
        {"title": "Dasara", "primary_language": "Telugu", "year": 2023, "genres_str": "Action|Drama"},
        # Hindi
        {"title": "Jawan", "primary_language": "Hindi", "year": 2023, "genres_str": "Action|Thriller"},
        {"title": "Pathaan", "primary_language": "Hindi", "year": 2023, "genres_str": "Action|Thriller|Spy"},
        {"title": "Animal", "primary_language": "Hindi", "year": 2023, "genres_str": "Action|Crime|Drama"},
        {"title": "Fighter", "primary_language": "Hindi", "year": 2024, "genres_str": "Action|Thriller"},
        {"title": "Brahmastra Part One: Shiva", "primary_language": "Hindi", "year": 2022, "genres_str": "Action|Fantasy|Adventure"},
        # English
        {"title": "Oppenheimer", "primary_language": "English", "year": 2023, "genres_str": "Biography|Drama|History"},
        {"title": "Barbie", "primary_language": "English", "year": 2023, "genres_str": "Comedy|Fantasy"},
        {"title": "Spider-Man: No Way Home", "primary_language": "English", "year": 2021, "genres_str": "Action|Adventure|Fantasy"},
        {"title": "Dune: Part Two", "primary_language": "English", "year": 2024, "genres_str": "Action|Adventure|Sci-Fi"},
    ]

    # 2. Procedural Generation for the remaining ~1000 movies
    telugu_prefixes = ["Veera", "Maha", "Raja", "Prema", "Yodha", "Samara", "Bhaag", "Nava", "Kotha", "Simha"]
    telugu_suffixes = ["Raju", "Vamsam", "Katha", "Desam", "Poratam", "Sagar", "Nayak", "Babu", "Rao", "Siva"]
    
    hindi_prefixes = ["Dil", "Pyaar", "Ek", "Maha", "Khooni", "Raat", "Din", "Mera", "Apna", "Naya"]
    hindi_suffixes = ["Deewana", "Zindagi", "Khiladi", "Hindustani", "Safar", "Kahani", "Dost", "Dushman", "Sapna"]
    
    english_prefixes = ["The Last", "Return of", "Rise of", "Fall of", "Dark", "Silent", "Hidden", "Secret", "Lost", "Beyond"]
    english_suffixes = ["Shadow", "Hero", "Night", "Day", "Dawn", "City", "World", "Star", "Legend", "Mystery"]

    base_genres = ["Action", "Drama", "Comedy", "Romance", "Thriller", "Horror", "Sci-Fi", "Fantasy", "Crime", "Adventure"]
    
    random.seed(42) # For reproducible random titles

    # Generate 400 Telugu movies
    for _ in range(400):
        title = f"{random.choice(telugu_prefixes)} {random.choice(telugu_suffixes)} {random.randint(1, 3) if random.random() > 0.8 else ''}".strip()
        g_count = random.randint(1, 3)
        genres = random.sample(base_genres, g_count)
        if "Romance" in genres and "Action" not in genres and random.random() > 0.5:
            genres.append("Comedy")
            
        recent_movies.append({
            "title": title,
            "primary_language": "Telugu",
            "year": random.randint(2020, 2026),
            "genres_str": "|".join(genres)
        })

    # Generate 400 Hindi movies
    for _ in range(400):
        title = f"{random.choice(hindi_prefixes)} {random.choice(hindi_suffixes)} {random.randint(1, 3) if random.random() > 0.8 else ''}".strip()
        g_count = random.randint(1, 3)
        genres = random.sample(base_genres, g_count)
        recent_movies.append({
            "title": title,
            "primary_language": "Hindi",
            "year": random.randint(2020, 2026),
            "genres_str": "|".join(genres)
        })

    # Generate 400 English movies
    for _ in range(400):
        title = f"{random.choice(english_prefixes)} {random.choice(english_suffixes)} {random.randint(1, 3) if random.random() > 0.8 else ''}".strip()
        g_count = random.randint(1, 3)
        genres = random.sample(base_genres, g_count)
        recent_movies.append({
            "title": title,
            "primary_language": "English",
            "year": random.randint(2020, 2026),
            "genres_str": "|".join(genres)
        })

    df = pd.DataFrame(recent_movies)
    
    # Ensure titles are unique
    df['title'] = df['title'] + " (" + df['year'].astype(str) + ") - " + (df.index + 1).astype(str)
    
    # Clean up the few hardcoded real movies to not have the index
    for i in range(17):
        # We manually inputted 17 real movies at the top
        df.loc[i, 'title'] = recent_movies[i]['title']
    
    # Generate unique fake movie IDs
    df['movieId'] = range(9000000, 9000000 + len(df))
    df['movieId'] = df['movieId'].astype(str)
    
    df['genres_list'] = df['genres_str'].apply(lambda x: [g.strip() for g in x.split('|')])
    df['genres_str'] = df['genres_list'].apply(lambda x: ', '.join(x))
    
    # Save to CSV
    output_file = 'movies_recent_2020_2026.csv'
    df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(df)} movies.")

if __name__ == '__main__':
    generate_large_movies_dataset()
