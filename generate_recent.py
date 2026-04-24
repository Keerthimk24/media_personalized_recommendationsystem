import pandas as pd
import random

def generate_recent_movies_dataset():
    # Curated list of major movies from 2020 to 2024+
    # with accurate real-world genres.
    
    recent_movies = [
        # --- TELUGU ---
        {"title": "Kalki 2898 AD", "primary_language": "Telugu", "year": 2024, "genres_str": "Action|Sci-Fi|Fantasy"},
        {"title": "Salaar: Part 1 - Ceasefire", "primary_language": "Telugu", "year": 2023, "genres_str": "Action|Crime|Thriller"},
        {"title": "RRR", "primary_language": "Telugu", "year": 2022, "genres_str": "Action|Drama|History"},
        {"title": "Pushpa: The Rise", "primary_language": "Telugu", "year": 2021, "genres_str": "Action|Crime|Drama"},
        {"title": "Pushpa 2: The Rule", "primary_language": "Telugu", "year": 2024, "genres_str": "Action|Crime|Thriller"},
        {"title": "Guntur Kaaram", "primary_language": "Telugu", "year": 2024, "genres_str": "Action|Drama|Comedy"},
        {"title": "Hi Nanna", "primary_language": "Telugu", "year": 2023, "genres_str": "Romance|Drama|Family"},
        {"title": "Dasara", "primary_language": "Telugu", "year": 2023, "genres_str": "Action|Drama"},
        {"title": "Bhagavanth Kesari", "primary_language": "Telugu", "year": 2023, "genres_str": "Action|Comedy|Drama"},
        {"title": "Sita Ramam", "primary_language": "Telugu", "year": 2022, "genres_str": "Romance|Drama|Action"},
        {"title": "Bimbisara", "primary_language": "Telugu", "year": 2022, "genres_str": "Action|Fantasy"},
        {"title": "Karthikeya 2", "primary_language": "Telugu", "year": 2022, "genres_str": "Adventure|Mystery|Thriller"},
        {"title": "DJ Tillu", "primary_language": "Telugu", "year": 2022, "genres_str": "Comedy|Crime|Romance"},
        {"title": "Tillu Square", "primary_language": "Telugu", "year": 2024, "genres_str": "Comedy|Crime|Romance"},
        {"title": "Hanuman", "primary_language": "Telugu", "year": 2024, "genres_str": "Action|Fantasy|Superhero"},
        {"title": "V", "primary_language": "Telugu", "year": 2020, "genres_str": "Action|Thriller|Mystery"},
        {"title": "Ala Vaikunthapurramuloo", "primary_language": "Telugu", "year": 2020, "genres_str": "Action|Drama|Comedy"},
        {"title": "Sarileru Neekevvaru", "primary_language": "Telugu", "year": 2020, "genres_str": "Action|Comedy"},
        {"title": "Bheeshma", "primary_language": "Telugu", "year": 2020, "genres_str": "Romance|Comedy|Action"},
        {"title": "Love Story", "primary_language": "Telugu", "year": 2021, "genres_str": "Romance|Drama"},
        {"title": "Akhanda", "primary_language": "Telugu", "year": 2021, "genres_str": "Action|Drama"},
        {"title": "Jathi Ratnalu", "primary_language": "Telugu", "year": 2021, "genres_str": "Comedy"},
        {"title": "Major", "primary_language": "Telugu", "year": 2022, "genres_str": "Action|Biography|Drama"},
        {"title": "Waltair Veerayya", "primary_language": "Telugu", "year": 2023, "genres_str": "Action|Comedy|Drama"},
        {"title": "Veera Simha Reddy", "primary_language": "Telugu", "year": 2023, "genres_str": "Action|Drama"},
        {"title": "Baby", "primary_language": "Telugu", "year": 2023, "genres_str": "Romance|Drama"},
        {"title": "Devara: Part 1", "primary_language": "Telugu", "year": 2024, "genres_str": "Action|Drama|Thriller"},
        {"title": "Game Changer", "primary_language": "Telugu", "year": 2024, "genres_str": "Action|Political|Drama"},

        # --- HINDI ---
        {"title": "Jawan", "primary_language": "Hindi", "year": 2023, "genres_str": "Action|Thriller"},
        {"title": "Pathaan", "primary_language": "Hindi", "year": 2023, "genres_str": "Action|Thriller|Spy"},
        {"title": "Animal", "primary_language": "Hindi", "year": 2023, "genres_str": "Action|Crime|Drama"},
        {"title": "Fighter", "primary_language": "Hindi", "year": 2024, "genres_str": "Action|Thriller"},
        {"title": "Brahmastra Part One: Shiva", "primary_language": "Hindi", "year": 2022, "genres_str": "Action|Fantasy|Adventure"},
        {"title": "Drishyam 2", "primary_language": "Hindi", "year": 2022, "genres_str": "Crime|Thriller|Drama"},
        {"title": "Bhool Bhulaiyaa 2", "primary_language": "Hindi", "year": 2022, "genres_str": "Comedy|Horror"},
        {"title": "The Kashmir Files", "primary_language": "Hindi", "year": 2022, "genres_str": "Drama|History"},
        {"title": "Sooryavanshi", "primary_language": "Hindi", "year": 2021, "genres_str": "Action|Crime|Thriller"},
        {"title": "Shershaah", "primary_language": "Hindi", "year": 2021, "genres_str": "Action|Biography|War"},
        {"title": "83", "primary_language": "Hindi", "year": 2021, "genres_str": "Biography|Drama|Sport"},
        {"title": "Tanhaji: The Unsung Warrior", "primary_language": "Hindi", "year": 2020, "genres_str": "Action|Biography|History"},
        {"title": "Thappad", "primary_language": "Hindi", "year": 2020, "genres_str": "Drama"},
        {"title": "Ludo", "primary_language": "Hindi", "year": 2020, "genres_str": "Action|Comedy|Crime"},
        {"title": "Gadar 2", "primary_language": "Hindi", "year": 2023, "genres_str": "Action|Drama"},
        {"title": "Rocky Aur Rani Kii Prem Kahaani", "primary_language": "Hindi", "year": 2023, "genres_str": "Comedy|Romance|Family"},
        {"title": "Dunki", "primary_language": "Hindi", "year": 2023, "genres_str": "Comedy|Drama"},
        {"title": "Tiger 3", "primary_language": "Hindi", "year": 2023, "genres_str": "Action|Thriller|Spy"},
        {"title": "Bade Miyan Chote Miyan", "primary_language": "Hindi", "year": 2024, "genres_str": "Action|Thriller"},
        {"title": "Crew", "primary_language": "Hindi", "year": 2024, "genres_str": "Comedy"},
        {"title": "Shaitaan", "primary_language": "Hindi", "year": 2024, "genres_str": "Horror|Thriller"},

        # --- ENGLISH ---
        {"title": "Oppenheimer", "primary_language": "English", "year": 2023, "genres_str": "Biography|Drama|History"},
        {"title": "Barbie", "primary_language": "English", "year": 2023, "genres_str": "Comedy|Fantasy"},
        {"title": "Spider-Man: No Way Home", "primary_language": "English", "year": 2021, "genres_str": "Action|Adventure|Fantasy"},
        {"title": "Top Gun: Maverick", "primary_language": "English", "year": 2022, "genres_str": "Action|Drama"},
        {"title": "Avatar: The Way of Water", "primary_language": "English", "year": 2022, "genres_str": "Action|Adventure|Fantasy"},
        {"title": "The Batman", "primary_language": "English", "year": 2022, "genres_str": "Action|Crime|Drama"},
        {"title": "Dune", "primary_language": "English", "year": 2021, "genres_str": "Action|Adventure|Sci-Fi"},
        {"title": "Dune: Part Two", "primary_language": "English", "year": 2024, "genres_str": "Action|Adventure|Sci-Fi"},
        {"title": "John Wick: Chapter 4", "primary_language": "English", "year": 2023, "genres_str": "Action|Crime|Thriller"},
        {"title": "Guardians of the Galaxy Vol. 3", "primary_language": "English", "year": 2023, "genres_str": "Action|Adventure|Comedy"},
        {"title": "Spider-Man: Across the Spider-Verse", "primary_language": "English", "year": 2023, "genres_str": "Animation|Action|Adventure"},
        {"title": "Everything Everywhere All at Once", "primary_language": "English", "year": 2022, "genres_str": "Action|Adventure|Comedy"},
        {"title": "No Time to Die", "primary_language": "English", "year": 2021, "genres_str": "Action|Adventure|Thriller"},
        {"title": "Tenet", "primary_language": "English", "year": 2020, "genres_str": "Action|Sci-Fi|Thriller"},
        {"title": "Soul", "primary_language": "English", "year": 2020, "genres_str": "Animation|Adventure|Comedy"},
        {"title": "A Quiet Place Part II", "primary_language": "English", "year": 2020, "genres_str": "Drama|Horror|Sci-Fi"},
        {"title": "The Super Mario Bros. Movie", "primary_language": "English", "year": 2023, "genres_str": "Animation|Adventure|Comedy"},
        {"title": "Mission: Impossible - Dead Reckoning Part One", "primary_language": "English", "year": 2023, "genres_str": "Action|Adventure|Thriller"},
        {"title": "Deadpool & Wolverine", "primary_language": "English", "year": 2024, "genres_str": "Action|Comedy|Sci-Fi"},
        {"title": "Godzilla x Kong: The New Empire", "primary_language": "English", "year": 2024, "genres_str": "Action|Adventure|Sci-Fi"},
        {"title": "Furiosa: A Mad Max Saga", "primary_language": "English", "year": 2024, "genres_str": "Action|Adventure|Sci-Fi"},
        {"title": "Inside Out 2", "primary_language": "English", "year": 2024, "genres_str": "Animation|Adventure|Comedy"}
    ]

    df = pd.DataFrame(recent_movies)
    
    # Generate unique fake movie IDs that don't conflict with TMDB
    df['movieId'] = range(9000000, 9000000 + len(df))
    df['movieId'] = df['movieId'].astype(str)
    
    # Process genres list
    df['genres_list'] = df['genres_str'].apply(lambda x: [g.strip() for g in x.split('|')])
    df['genres_str'] = df['genres_list'].apply(lambda x: ', '.join(x))
    
    # Save to CSV
    output_file = 'movies_recent_2020_2026.csv'
    df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(df)} movies.")
    
    # Print Genre Distribution
    for lang in ['Telugu', 'Hindi', 'English']:
        lang_df = df[df['primary_language'] == lang]
        all_genres = []
        for gl in lang_df['genres_list']:
            all_genres.extend(gl)
            
        from collections import Counter
        top_genres = Counter(all_genres).most_common()
        
        print(f"\n[{lang}] Movies 2020-2026 Genre Representation:")
        for g, count in top_genres:
            print(f"  - {g}: {count} movies")

if __name__ == '__main__':
    generate_recent_movies_dataset()
