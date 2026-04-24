"""
Step 3: Train the NCF Deep Learning Model
------------------------------------------
Trains on preprocessed ratings data with EarlyStopping and LR scheduling.
Run AFTER step1_preprocess.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from step2_model import build_ncf


def train():
    print("=" * 60)
    print("STEP 3: TRAINING NCF MODEL")
    print("=" * 60)

    # Load preprocessed data
    print("\n[1/4] Loading preprocessed data...")
    ratings = pd.read_csv("ratings_clean.csv")

    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)

    num_users = meta['num_users']
    num_items = meta['num_items']
    print(f"  Users: {num_users}, Items: {num_items}, Ratings: {len(ratings)}")

    # Prepare arrays
    X_user = ratings['user_enc'].values
    X_item = ratings['movie_enc'].values
    y = ratings['rating_norm'].values.astype(np.float32)

    # Train/Val split
    Xu_tr, Xu_te, Xi_tr, Xi_te, y_tr, y_te = train_test_split(
        X_user, X_item, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(y_tr)}, Validation: {len(y_te)}")

    # Build model
    print("\n[2/4] Building NCF model...")
    model = build_ncf(num_users, num_items, embed_dim=64)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=2, verbose=1
        )
    ]

    # Train
    print("\n[3/4] Training...")
    history = model.fit(
        [Xu_tr, Xi_tr], y_tr,
        validation_data=([Xu_te, Xi_te], y_te),
        epochs=30, batch_size=256,
        callbacks=callbacks, verbose=1
    )

    # Save
    print("\n[4/4] Saving trained model...")
    model.save("ncf_model.h5")

    # Print final metrics
    val_loss = min(history.history['val_loss'])
    val_mae = min(history.history['val_mae'])
    print(f"\n  Best Val Loss : {val_loss:.4f}")
    print(f"  Best Val MAE  : {val_mae:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE! Model saved -> ncf_model.h5")
    print("=" * 60)


if __name__ == "__main__":
    train()
