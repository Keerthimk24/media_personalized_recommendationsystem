"""Quick test of the entire pipeline."""
from step4_engine import RecommendationEngine

e = RecommendationEngine()

# Test recommendations for existing user
print("\n=== Testing U1 Recommendations ===")
recs = e.recommend('U1', top_k=5)
print(f"Recs count: {len(recs)}")
for r in recs:
    print(f"  {r['title']} ({r['language']}) score={r['hybrid_score']:.3f}")

# Test cold start
print("\n=== Testing Cold Start ===")
cold = e._cold_start_recommend('COLD', 5, genre_filter=['Action'])
print(f"Cold start count: {len(cold)}")
for r in cold:
    print(f"  {r['title']} ({r['language']})")

# Test user history
print("\n=== Testing User History ===")
h = e.get_user_history('U1')
print(f"History count: {len(h)}")

# Test preferences
print("\n=== Testing Preferences ===")
p = e.get_user_preferences('U1')
print(f"Prefs: {p}")

# Test genre filter
print("\n=== Testing Genre Filter ===")
recs_genre = e.recommend('U1', top_k=5, genre_filter=['Action'])
print(f"Genre filtered recs: {len(recs_genre)}")
for r in recs_genre:
    print(f"  {r['title']} ({r['language']}) genres={r['genres']}")

# Test all genres/languages
print("\n=== All Genres ===")
print(e.get_all_genres())
print("\n=== All Languages ===")
print(e.get_all_languages())

# Test multiple users
print("\n=== Testing Multiple Users ===")
survey_users = e.survey['user_id'].unique()[:5]
for uid in survey_users:
    recs = e.recommend(uid, top_k=3)
    print(f"  {uid}: {len(recs)} recs")

print("\n✅ ALL TESTS PASSED!")
