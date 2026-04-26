"""
Model Evaluation & Architecture Report
-----------------------------------------
Shows all models, encoders/transformers used, and computes
accuracy metrics for the trained NCF hybrid recommendation system.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█   MEDIASTREAM — COMPLETE MODEL EVALUATION REPORT" + " "*18 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    # ─── 1. MODELS USED ────────────────────────────────────────────────
    separator("1. DEEP LEARNING MODEL — Neural Collaborative Filtering (NCF)")

    # Load metadata
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)

    num_users = meta['num_users']
    num_items = meta['num_items']
    r_min = meta['r_min']
    r_max = meta['r_max']

    print(f"""
    Model Type      : Neural Collaborative Filtering (NCF)
    Framework       : TensorFlow / Keras {tf.__version__}
    Task            : Rating Prediction (Regression via Sigmoid)
    Loss Function   : Binary Cross-Entropy
    Optimizer       : Adam (lr=0.001)

    ┌─────────────────────────────────────────────────────────────┐
    │                    NCF ARCHITECTURE                         │
    ├─────────────────────────────────────────────────────────────┤
    │  Input: User ID (1,)        Input: Movie ID (1,)           │
    │         ↓                            ↓                     │
    │  User Embedding (64-d)      Item Embedding (64-d)          │
    │         ↓                            ↓                     │
    │         └──────── Concatenate ────────┘                     │
    │                      ↓ (128-d)                             │
    │              Dense 256 + ReLU                               │
    │              BatchNormalization                             │
    │              Dropout (0.30)                                 │
    │                      ↓                                     │
    │              Dense 128 + ReLU                               │
    │              BatchNormalization                             │
    │              Dropout (0.25)                                 │
    │                      ↓                                     │
    │              Dense 64 + ReLU                                │
    │              BatchNormalization                             │
    │              Dropout (0.20)                                 │
    │                      ↓                                     │
    │              Dense 1 + Sigmoid                              │
    │              (Predicted Rating 0–1)                         │
    └─────────────────────────────────────────────────────────────┘

    Regularization  : L2 (1e-5) on embeddings, Dropout, BatchNorm
    Callbacks       : EarlyStopping (patience=5), ReduceLROnPlateau
    Epochs          : Up to 30 (with early stopping)
    Batch Size      : 256
    Embedding Dim   : 64
    """)

    # Load and display model summary
    model = tf.keras.models.load_model("ncf_model.h5", compile=False)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    print("    Keras Model Summary:")
    print("    " + "─"*55)
    model.summary(print_fn=lambda x: print(f"    {x}"))

    total_params = model.count_params()
    print(f"\n    Total Trainable Parameters: {total_params:,}")

    # ─── 2. ENCODERS / TRANSFORMERS USED ────────────────────────────────
    separator("2. ENCODERS & TRANSFORMERS")

    # Label Encoders
    with open("user_encoder.pkl", "rb") as f:
        user_enc = pickle.load(f)
    with open("movie_encoder.pkl", "rb") as f:
        movie_enc = pickle.load(f)

    print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │              ENCODER / TRANSFORMER PIPELINE                 │
    ├──────────────────┬──────────────────────────────────────────┤
    │  Component       │  Details                                 │
    ├──────────────────┼──────────────────────────────────────────┤
    │  User Encoder    │  sklearn.preprocessing.LabelEncoder      │
    │                  │  Maps user IDs → integer indices          │
    │                  │  Num Classes: {num_users:<27}│
    ├──────────────────┼──────────────────────────────────────────┤
    │  Movie Encoder   │  sklearn.preprocessing.LabelEncoder      │
    │                  │  Maps movie IDs → integer indices         │
    │                  │  Num Classes: {num_items:<27}│
    ├──────────────────┼──────────────────────────────────────────┤
    │  TF-IDF          │  sklearn.feature_extraction.text          │
    │  Vectorizer      │      .TfidfVectorizer                    │
    │                  │  Converts genre+language text → vectors   │
    │                  │  Max Features: 2000                       │
    │                  │  Stop Words: english                      │
    ├──────────────────┼──────────────────────────────────────────┤
    │  Rating          │  Min-Max Normalization                    │
    │  Normalizer      │  rating_norm = (r - r_min)/(r_max-r_min) │
    │                  │  r_min={r_min}, r_max={r_max}                        │
    ├──────────────────┼──────────────────────────────────────────┤
    │  User Embedding  │  tf.keras.layers.Embedding               │
    │                  │  Input: {num_users} users → 64-d vectors          │
    │                  │  L2 regularization: 1e-5                  │
    ├──────────────────┼──────────────────────────────────────────┤
    │  Item Embedding  │  tf.keras.layers.Embedding               │
    │                  │  Input: {num_items} movies → 64-d vectors        │
    │                  │  L2 regularization: 1e-5                  │
    ├──────────────────┼──────────────────────────────────────────┤
    │  Cosine          │  sklearn.metrics.pairwise                 │
    │  Similarity      │      .cosine_similarity                  │
    │                  │  Content-based similarity scoring         │
    └──────────────────┴──────────────────────────────────────────┘
    """)

    # TF-IDF details
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    content_matrix = load_npz("content_matrix.npz")
    vocab_size = len(tfidf.vocabulary_)
    print(f"    TF-IDF Vocabulary Size   : {vocab_size}")
    print(f"    Content Matrix Shape     : {content_matrix.shape}")
    print(f"    Content Matrix Non-zeros : {content_matrix.nnz:,}")
    print(f"    Sparsity                 : {(1 - content_matrix.nnz / (content_matrix.shape[0]*content_matrix.shape[1]))*100:.2f}%")

    # Show sample vocabulary
    top_terms = sorted(tfidf.vocabulary_.items(), key=lambda x: x[1])[:20]
    print(f"\n    Top TF-IDF Terms (sample): {', '.join([t[0] for t in top_terms])}")

    # ─── 3. HYBRID SCORING FORMULA ──────────────────────────────────────
    separator("3. HYBRID RECOMMENDATION FORMULA")

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │               HYBRID SCORING FORMULA                        │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │   hybrid_score = 0.45 × NCF_score                          │
    │                + 0.30 × Content_similarity (TF-IDF)         │
    │                + 0.25 × Language_boost                      │
    │                                                             │
    ├─────────────────────────────────────────────────────────────┤
    │  NCF Score      : Deep Learning collaborative prediction    │
    │  Content Score  : Cosine similarity of TF-IDF vectors       │
    │  Language Boost : +1.0 primary lang, +0.55 secondary,       │
    │                   -0.3 penalty for non-preferred             │
    ├─────────────────────────────────────────────────────────────┤
    │  Cold Start     : 0.60 × Genre_match + 0.40 × Popularity   │
    └─────────────────────────────────────────────────────────────┘
    """)

    # ─── 4. ACCURACY METRICS ────────────────────────────────────────────
    separator("4. MODEL ACCURACY & EVALUATION METRICS")

    # Load ratings data
    ratings = pd.read_csv("ratings_clean.csv")
    print(f"\n    Total Ratings        : {len(ratings):,}")
    print(f"    Unique Users         : {ratings['userId'].nunique()}")
    print(f"    Unique Movies        : {ratings['movieId'].nunique()}")
    print(f"    Rating Range (norm)  : [{ratings['rating_norm'].min():.3f}, {ratings['rating_norm'].max():.3f}]")
    print(f"    Rating Range (raw)   : [{ratings['rating'].min()}, {ratings['rating'].max()}]")

    # Split data same way as training
    X_user = ratings['user_enc'].values
    X_item = ratings['movie_enc'].values
    y = ratings['rating_norm'].values.astype(np.float32)

    Xu_tr, Xu_te, Xi_tr, Xi_te, y_tr, y_te = train_test_split(
        X_user, X_item, y, test_size=0.2, random_state=42
    )

    print(f"\n    Train Set Size       : {len(y_tr):,}")
    print(f"    Test Set Size        : {len(y_te):,}")

    # ── 4a. Rating Prediction Metrics ──
    print(f"\n    {'─'*50}")
    print(f"    4a. RATING PREDICTION METRICS (on test set)")
    print(f"    {'─'*50}")

    # Predict on test set
    y_pred_test = model.predict([Xu_te, Xi_te], verbose=0, batch_size=1024).flatten()
    y_pred_train = model.predict([Xu_tr, Xi_tr], verbose=0, batch_size=1024).flatten()

    # Metrics on normalized scale (0-1)
    test_mae_norm = mean_absolute_error(y_te, y_pred_test)
    test_rmse_norm = np.sqrt(mean_squared_error(y_te, y_pred_test))
    train_mae_norm = mean_absolute_error(y_tr, y_pred_train)
    train_rmse_norm = np.sqrt(mean_squared_error(y_tr, y_pred_train))

    # Convert back to original scale
    scale = r_max - r_min
    test_mae_orig = test_mae_norm * scale
    test_rmse_orig = test_rmse_norm * scale
    train_mae_orig = train_mae_norm * scale
    train_rmse_orig = train_rmse_norm * scale

    # Evaluate using model.evaluate
    print("\n    Running model.evaluate() on test set...")
    test_loss, test_mae_eval, test_rmse_eval = model.evaluate(
        [Xu_te, Xi_te], y_te, verbose=0, batch_size=1024
    )
    train_loss, train_mae_eval, train_rmse_eval = model.evaluate(
        [Xu_tr, Xi_tr], y_tr, verbose=0, batch_size=1024
    )

    print(f"""
    ┌───────────────────────────────────────────────────────────────┐
    │            RATING PREDICTION ACCURACY                         │
    ├─────────────────────┬──────────────────┬──────────────────────┤
    │  Metric             │  Train Set       │  Test Set            │
    ├─────────────────────┼──────────────────┼──────────────────────┤
    │  Loss (BCE)         │  {train_loss:<16.6f}│  {test_loss:<20.6f}│
    │  MAE (normalized)   │  {train_mae_norm:<16.6f}│  {test_mae_norm:<20.6f}│
    │  RMSE (normalized)  │  {train_rmse_norm:<16.6f}│  {test_rmse_norm:<20.6f}│
    │  MAE (orig 1-5)     │  {train_mae_orig:<16.6f}│  {test_mae_orig:<20.6f}│
    │  RMSE (orig 1-5)    │  {test_rmse_orig:<16.6f}│  {test_rmse_orig:<20.6f}│
    └─────────────────────┴──────────────────┴──────────────────────┘
    """)

    # ── 4b. Classification / Ranking Metrics ──
    print(f"    {'─'*50}")
    print(f"    4b. RANKING / RECOMMENDATION METRICS (on test set)")
    print(f"    {'─'*50}")

    # Threshold: if predicted > 0.6 = "liked", actual > 0.6 = "liked"
    threshold = 0.6
    y_liked_actual = (y_te >= threshold).astype(int)
    y_liked_pred = (y_pred_test >= threshold).astype(int)

    tp = np.sum((y_liked_pred == 1) & (y_liked_actual == 1))
    fp = np.sum((y_liked_pred == 1) & (y_liked_actual == 0))
    fn = np.sum((y_liked_pred == 0) & (y_liked_actual == 1))
    tn = np.sum((y_liked_pred == 0) & (y_liked_actual == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_te)

    print(f"""
    Threshold for "liked" : >= {threshold} (normalized)  [= {threshold * scale + r_min:.1f}/5 original]

    ┌───────────────────────────────────────────────────────────────┐
    │             CLASSIFICATION METRICS                            │
    ├─────────────────────┬─────────────────────────────────────────┤
    │  Accuracy           │  {accuracy*100:<8.2f}%                          │
    │  Precision          │  {precision*100:<8.2f}%                          │
    │  Recall             │  {recall*100:<8.2f}%                          │
    │  F1-Score           │  {f1*100:<8.2f}%                          │
    ├─────────────────────┼─────────────────────────────────────────┤
    │  True Positives     │  {tp:<8,}                               │
    │  False Positives    │  {fp:<8,}                               │
    │  True Negatives     │  {tn:<8,}                               │
    │  False Negatives    │  {fn:<8,}                               │
    └─────────────────────┴─────────────────────────────────────────┘
    """)

    # ── 4c. Per-User Hit Rate & NDCG@K ──
    print(f"    {'─'*50}")
    print(f"    4c. TOP-K RECOMMENDATION METRICS")
    print(f"    {'─'*50}")

    K_VALUES = [5, 10]

    # Group test set by user
    test_df = pd.DataFrame({
        'user_enc': Xu_te, 'movie_enc': Xi_te,
        'actual': y_te, 'predicted': y_pred_test
    })

    for K in K_VALUES:
        hits = 0
        ndcg_scores = []
        total_users = 0

        for user_id, group in test_df.groupby('user_enc'):
            if len(group) < 2:
                continue
            total_users += 1

            # Actual top-K items (by true rating)
            actual_top = set(group.nlargest(K, 'actual')['movie_enc'].values)

            # Predicted top-K items
            pred_top = group.nlargest(K, 'predicted')['movie_enc'].values

            # Hit Rate: does at least 1 actual liked item appear in predicted top-K?
            hit = len(set(pred_top) & actual_top) > 0
            if hit:
                hits += 1

            # NDCG@K
            relevance = [1.0 if m in actual_top else 0.0 for m in pred_top]
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
            ideal = sorted(relevance, reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)

        hit_rate = hits / total_users if total_users > 0 else 0
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0

        header = f"TOP-{K} METRICS  (evaluated on {total_users} users)"
        pad = 62 - len(header) - 4
        hr_label = f"Hit Rate@{K}"
        ndcg_label = f"NDCG@{K}"
        print(f"""
    +---------------------------------------------------------------+
    |  {header}{' '*pad}|
    +---------------------+-----------------------------------------+
    |  {hr_label:<19} |  {hit_rate*100:<8.2f}%                          |
    |  {ndcg_label:<19} |  {avg_ndcg:<8.4f}                           |
    +---------------------+-----------------------------------------+""")

    # ─── 5. DATASET STATISTICS ──────────────────────────────────────────
    separator("5. DATASET STATISTICS")

    movies = pd.read_csv("movies_clean.csv")
    survey = pd.read_csv("cleaned_survey.csv")

    print(f"""
    ┌───────────────────────────────────────────────────────────────┐
    │                    DATASET OVERVIEW                            │
    ├─────────────────────┬─────────────────────────────────────────┤
    │  Survey Responses   │  {len(survey):<8,}                               │
    │  Total Movies       │  {len(movies):<8,}                               │
    │  Total Ratings      │  {len(ratings):<8,}                               │
    │  Unique Users       │  {ratings['userId'].nunique():<8,}                               │
    │  Unique Languages   │  {movies['primary_language'].nunique():<8,}                               │
    ├─────────────────────┼─────────────────────────────────────────┤
    │  Avg Rating (raw)   │  {ratings['rating'].mean():<8.2f}                               │
    │  Ratings Std Dev    │  {ratings['rating'].std():<8.2f}                               │
    │  Sparsity           │  {(1 - len(ratings)/(ratings['userId'].nunique()*ratings['movieId'].nunique()))*100:<8.2f}%                          │
    └─────────────────────┴─────────────────────────────────────────┘
    """)

    # Language distribution
    print("    Language Distribution in Catalog:")
    lang_dist = movies['primary_language'].value_counts().head(10)
    for lang, count in lang_dist.items():
        pct = count / len(movies) * 100
        bar = "█" * int(pct / 2)
        print(f"      {lang:<15} {count:>6,}  ({pct:5.1f}%)  {bar}")

    # Genre distribution
    print("\n    Top Genre Distribution in Catalog:")
    all_genres = []
    for gs in movies['genres_str'].dropna():
        all_genres.extend([g.strip() for g in str(gs).split('|') if g.strip() and g.strip() != 'Unknown'])
    from collections import Counter
    genre_counts = Counter(all_genres).most_common(10)
    for genre, count in genre_counts:
        pct = count / len(movies) * 100
        bar = "█" * int(pct / 2)
        print(f"      {genre:<15} {count:>6,}  ({pct:5.1f}%)  {bar}")

    # ─── 6. COMPLETE PIPELINE SUMMARY ───────────────────────────────────
    separator("6. COMPLETE PIPELINE SUMMARY")

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                 END-TO-END PIPELINE                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Step 1: Data Preprocessing (step1_preprocess.py)           │
    │    ├── Load DLDATA.csv (survey), movies.csv, ratings.csv    │
    │    ├── LabelEncoder → user IDs & movie IDs to integers      │
    │    ├── Min-Max Normalization → ratings to [0, 1]            │
    │    ├── TfidfVectorizer → genre+language text to vectors     │
    │    └── Save: encoders, TF-IDF, content matrix, clean CSVs   │
    │                                                             │
    │  Step 2: Model Architecture (step2_model.py)                │
    │    └── NCF: Embedding → Concat → Dense(256,128,64) → σ      │
    │                                                             │
    │  Step 3: Training (step3_train.py)                          │
    │    ├── 80/20 train/test split                               │
    │    ├── Adam optimizer, Binary Cross-Entropy loss            │
    │    ├── EarlyStopping + ReduceLROnPlateau                    │
    │    └── Save: ncf_model.h5                                   │
    │                                                             │
    │  Step 4: Hybrid Engine (step4_engine.py)                    │
    │    ├── NCF scores (collaborative filtering)                 │
    │    ├── TF-IDF Cosine Similarity (content-based)             │
    │    ├── Language Boost (survey + history-aware)               │
    │    └── Hybrid: 0.45×NCF + 0.30×Content + 0.25×LangBoost    │
    │                                                             │
    │  Step 5: Streamlit Dashboard (step5_app.py)                 │
    │    ├── Glassmorphism UI with animations                     │
    │    ├── User profile, watch history, preferences             │
    │    ├── Cold start support for new users                     │
    │    └── Explainable AI (reason for each recommendation)      │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)

    print("█"*70)
    print("█  EVALUATION COMPLETE                                              █")
    print("█"*70 + "\n")


if __name__ == "__main__":
    main()
