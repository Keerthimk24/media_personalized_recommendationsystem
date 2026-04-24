import pandas as pd

def generate_real_combined_dataset():
    print("Loading local Kaggle dataset for 2000-2020 movies...")
    # Load movies clean
    clean_df = pd.read_csv('movies_clean.csv')
    clean_df['movieId'] = clean_df['movieId'].astype(str)
    
    # Load metadata
    meta_df = pd.read_csv('../DLCASE/movies_metadata.csv', low_memory=False)
    meta_df['id'] = pd.to_numeric(meta_df['id'], errors='coerce')
    meta_df = meta_df.dropna(subset=['id'])
    meta_df['id'] = meta_df['id'].astype(int).astype(str)
    
    # Merge to get years
    merged = pd.merge(clean_df, meta_df[['id', 'release_date']], left_on='movieId', right_on='id', how='inner')
    merged['year'] = pd.to_datetime(merged['release_date'], errors='coerce').dt.year
    
    # Filter 2000 to 2020
    recent = merged[(merged['year'] >= 2000) & (merged['primary_language'].isin(['Telugu', 'Hindi', 'English']))].copy()
    recent['year'] = recent['year'].astype(int)
    
    # Format the local movies to match our target CSV
    # We need: title, primary_language, year, genres_str, movieId, genres_list
    local_movies = recent[['title', 'primary_language', 'year', 'genres_str', 'movieId']].copy()
    
    print(f"Found {len(local_movies)} real movies from 2000-2020 in local database.")
    
    # Now append the 2020-2026 curated list
    curated_recent_movies = [
        # --- TELUGU ---
        {"title": "Kalki 2898 AD", "primary_language": "Telugu", "year": 2024, "genres_str": "Action, Sci-Fi, Fantasy", "movieId": "9000001"},
        {"title": "Salaar: Part 1 - Ceasefire", "primary_language": "Telugu", "year": 2023, "genres_str": "Action, Crime, Thriller", "movieId": "9000002"},
        {"title": "RRR", "primary_language": "Telugu", "year": 2022, "genres_str": "Action, Drama, History", "movieId": "9000003"},
        {"title": "Pushpa: The Rise", "primary_language": "Telugu", "year": 2021, "genres_str": "Action, Crime, Drama", "movieId": "9000004"},
        {"title": "Pushpa 2: The Rule", "primary_language": "Telugu", "year": 2024, "genres_str": "Action, Crime, Thriller", "movieId": "9000005"},
        {"title": "Guntur Kaaram", "primary_language": "Telugu", "year": 2024, "genres_str": "Action, Drama, Comedy", "movieId": "9000006"},
        {"title": "Hi Nanna", "primary_language": "Telugu", "year": 2023, "genres_str": "Romance, Drama, Family", "movieId": "9000007"},
        {"title": "Dasara", "primary_language": "Telugu", "year": 2023, "genres_str": "Action, Drama", "movieId": "9000008"},
        {"title": "Bhagavanth Kesari", "primary_language": "Telugu", "year": 2023, "genres_str": "Action, Comedy, Drama", "movieId": "9000009"},
        {"title": "Sita Ramam", "primary_language": "Telugu", "year": 2022, "genres_str": "Romance, Drama, Action", "movieId": "9000010"},
        {"title": "Bimbisara", "primary_language": "Telugu", "year": 2022, "genres_str": "Action, Fantasy", "movieId": "9000011"},
        {"title": "Karthikeya 2", "primary_language": "Telugu", "year": 2022, "genres_str": "Adventure, Mystery, Thriller", "movieId": "9000012"},
        {"title": "DJ Tillu", "primary_language": "Telugu", "year": 2022, "genres_str": "Comedy, Crime, Romance", "movieId": "9000013"},
        {"title": "Tillu Square", "primary_language": "Telugu", "year": 2024, "genres_str": "Comedy, Crime, Romance", "movieId": "9000014"},
        {"title": "Hanuman", "primary_language": "Telugu", "year": 2024, "genres_str": "Action, Fantasy, Superhero", "movieId": "9000015"},
        {"title": "V", "primary_language": "Telugu", "year": 2020, "genres_str": "Action, Thriller, Mystery", "movieId": "9000016"},
        {"title": "Ala Vaikunthapurramuloo", "primary_language": "Telugu", "year": 2020, "genres_str": "Action, Drama, Comedy", "movieId": "9000017"},
        {"title": "Sarileru Neekevvaru", "primary_language": "Telugu", "year": 2020, "genres_str": "Action, Comedy", "movieId": "9000018"},
        {"title": "Bheeshma", "primary_language": "Telugu", "year": 2020, "genres_str": "Romance, Comedy, Action", "movieId": "9000019"},
        {"title": "Love Story", "primary_language": "Telugu", "year": 2021, "genres_str": "Romance, Drama", "movieId": "9000020"},
        {"title": "Akhanda", "primary_language": "Telugu", "year": 2021, "genres_str": "Action, Drama", "movieId": "9000021"},
        {"title": "Jathi Ratnalu", "primary_language": "Telugu", "year": 2021, "genres_str": "Comedy", "movieId": "9000022"},
        {"title": "Major", "primary_language": "Telugu", "year": 2022, "genres_str": "Action, Biography, Drama", "movieId": "9000023"},
        {"title": "Waltair Veerayya", "primary_language": "Telugu", "year": 2023, "genres_str": "Action, Comedy, Drama", "movieId": "9000024"},
        {"title": "Veera Simha Reddy", "primary_language": "Telugu", "year": 2023, "genres_str": "Action, Drama", "movieId": "9000025"},
        {"title": "Baby", "primary_language": "Telugu", "year": 2023, "genres_str": "Romance, Drama", "movieId": "9000026"},
        {"title": "Devara: Part 1", "primary_language": "Telugu", "year": 2024, "genres_str": "Action, Drama, Thriller", "movieId": "9000027"},
        {"title": "Game Changer", "primary_language": "Telugu", "year": 2024, "genres_str": "Action, Political, Drama", "movieId": "9000028"},

        # --- HINDI ---
        {"title": "Jawan", "primary_language": "Hindi", "year": 2023, "genres_str": "Action, Thriller", "movieId": "9000029"},
        {"title": "Pathaan", "primary_language": "Hindi", "year": 2023, "genres_str": "Action, Thriller, Spy", "movieId": "9000030"},
        {"title": "Animal", "primary_language": "Hindi", "year": 2023, "genres_str": "Action, Crime, Drama", "movieId": "9000031"},
        {"title": "Fighter", "primary_language": "Hindi", "year": 2024, "genres_str": "Action, Thriller", "movieId": "9000032"},
        {"title": "Brahmastra Part One: Shiva", "primary_language": "Hindi", "year": 2022, "genres_str": "Action, Fantasy, Adventure", "movieId": "9000033"},
        {"title": "Drishyam 2", "primary_language": "Hindi", "year": 2022, "genres_str": "Crime, Thriller, Drama", "movieId": "9000034"},
        {"title": "Bhool Bhulaiyaa 2", "primary_language": "Hindi", "year": 2022, "genres_str": "Comedy, Horror", "movieId": "9000035"},
        {"title": "The Kashmir Files", "primary_language": "Hindi", "year": 2022, "genres_str": "Drama, History", "movieId": "9000036"},
        {"title": "Sooryavanshi", "primary_language": "Hindi", "year": 2021, "genres_str": "Action, Crime, Thriller", "movieId": "9000037"},
        {"title": "Shershaah", "primary_language": "Hindi", "year": 2021, "genres_str": "Action, Biography, War", "movieId": "9000038"},
        {"title": "83", "primary_language": "Hindi", "year": 2021, "genres_str": "Biography, Drama, Sport", "movieId": "9000039"},
        {"title": "Tanhaji: The Unsung Warrior", "primary_language": "Hindi", "year": 2020, "genres_str": "Action, Biography, History", "movieId": "9000040"},
        {"title": "Thappad", "primary_language": "Hindi", "year": 2020, "genres_str": "Drama", "movieId": "9000041"},
        {"title": "Ludo", "primary_language": "Hindi", "year": 2020, "genres_str": "Action, Comedy, Crime", "movieId": "9000042"},
        {"title": "Gadar 2", "primary_language": "Hindi", "year": 2023, "genres_str": "Action, Drama", "movieId": "9000043"},
        {"title": "Rocky Aur Rani Kii Prem Kahaani", "primary_language": "Hindi", "year": 2023, "genres_str": "Comedy, Romance, Family", "movieId": "9000044"},
        {"title": "Dunki", "primary_language": "Hindi", "year": 2023, "genres_str": "Comedy, Drama", "movieId": "9000045"},
        {"title": "Tiger 3", "primary_language": "Hindi", "year": 2023, "genres_str": "Action, Thriller, Spy", "movieId": "9000046"},
        {"title": "Bade Miyan Chote Miyan", "primary_language": "Hindi", "year": 2024, "genres_str": "Action, Thriller", "movieId": "9000047"},
        {"title": "Crew", "primary_language": "Hindi", "year": 2024, "genres_str": "Comedy", "movieId": "9000048"},
        {"title": "Shaitaan", "primary_language": "Hindi", "year": 2024, "genres_str": "Horror, Thriller", "movieId": "9000049"},

        # --- ENGLISH ---
        {"title": "Oppenheimer", "primary_language": "English", "year": 2023, "genres_str": "Biography, Drama, History", "movieId": "9000050"},
        {"title": "Barbie", "primary_language": "English", "year": 2023, "genres_str": "Comedy, Fantasy", "movieId": "9000051"},
        {"title": "Spider-Man: No Way Home", "primary_language": "English", "year": 2021, "genres_str": "Action, Adventure, Fantasy", "movieId": "9000052"},
        {"title": "Top Gun: Maverick", "primary_language": "English", "year": 2022, "genres_str": "Action, Drama", "movieId": "9000053"},
        {"title": "Avatar: The Way of Water", "primary_language": "English", "year": 2022, "genres_str": "Action, Adventure, Fantasy", "movieId": "9000054"},
        {"title": "The Batman", "primary_language": "English", "year": 2022, "genres_str": "Action, Crime, Drama", "movieId": "9000055"},
        {"title": "Dune", "primary_language": "English", "year": 2021, "genres_str": "Action, Adventure, Sci-Fi", "movieId": "9000056"},
        {"title": "Dune: Part Two", "primary_language": "English", "year": 2024, "genres_str": "Action, Adventure, Sci-Fi", "movieId": "9000057"},
        {"title": "John Wick: Chapter 4", "primary_language": "English", "year": 2023, "genres_str": "Action, Crime, Thriller", "movieId": "9000058"},
        {"title": "Guardians of the Galaxy Vol. 3", "primary_language": "English", "year": 2023, "genres_str": "Action, Adventure, Comedy", "movieId": "9000059"},
        {"title": "Spider-Man: Across the Spider-Verse", "primary_language": "English", "year": 2023, "genres_str": "Animation, Action, Adventure", "movieId": "9000060"},
        {"title": "Everything Everywhere All at Once", "primary_language": "English", "year": 2022, "genres_str": "Action, Adventure, Comedy", "movieId": "9000061"},
        {"title": "No Time to Die", "primary_language": "English", "year": 2021, "genres_str": "Action, Adventure, Thriller", "movieId": "9000062"},
        {"title": "Tenet", "primary_language": "English", "year": 2020, "genres_str": "Action, Sci-Fi, Thriller", "movieId": "9000063"},
        {"title": "Soul", "primary_language": "English", "year": 2020, "genres_str": "Animation, Adventure, Comedy", "movieId": "9000064"},
        {"title": "A Quiet Place Part II", "primary_language": "English", "year": 2020, "genres_str": "Drama, Horror, Sci-Fi", "movieId": "9000065"},
        {"title": "The Super Mario Bros. Movie", "primary_language": "English", "year": 2023, "genres_str": "Animation, Adventure, Comedy", "movieId": "9000066"},
        {"title": "Mission: Impossible - Dead Reckoning Part One", "primary_language": "English", "year": 2023, "genres_str": "Action, Adventure, Thriller", "movieId": "9000067"},
        {"title": "Deadpool & Wolverine", "primary_language": "English", "year": 2024, "genres_str": "Action, Comedy, Sci-Fi", "movieId": "9000068"},
        {"title": "Godzilla x Kong: The New Empire", "primary_language": "English", "year": 2024, "genres_str": "Action, Adventure, Sci-Fi", "movieId": "9000069"},
        {"title": "Furiosa: A Mad Max Saga", "primary_language": "English", "year": 2024, "genres_str": "Action, Adventure, Sci-Fi", "movieId": "9000070"},
        {"title": "Inside Out 2", "primary_language": "English", "year": 2024, "genres_str": "Animation, Adventure, Comedy", "movieId": "9000071"}
    ]
    curated_df = pd.DataFrame(curated_recent_movies)
    print(f"Added {len(curated_df)} curated real movies from 2020-2026.")
    
    # Combine
    final_df = pd.concat([local_movies, curated_df], ignore_index=True)
    
    # Drop duplicates just in case
    final_df = final_df.drop_duplicates(subset=['title', 'year'])
    
    # Create lists
    final_df['genres_str'] = final_df['genres_str'].fillna('Unknown')
    final_df['genres_list'] = final_df['genres_str'].apply(lambda x: [g.strip() for g in x.replace('|', ',').split(',')])
    final_df['genres_str'] = final_df['genres_list'].apply(lambda x: ', '.join(x))
    
    output_file = 'movies_real_2000_2026.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\nFinal dataset created: {output_file}")
    print(f"Total REAL Movies (2000-2026): {len(final_df)}")
    print(final_df.groupby(['primary_language']).size())

if __name__ == '__main__':
    generate_real_combined_dataset()
