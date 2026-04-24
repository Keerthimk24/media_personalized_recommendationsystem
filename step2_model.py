"""
Step 2: Deep Learning Model Architecture
-----------------------------------------
Neural Collaborative Filtering (NCF) with Genre & Language Embeddings.
This is a production-grade model that learns user-item interaction patterns.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def build_ncf(num_users: int, num_items: int, embed_dim: int = 64) -> tf.keras.Model:
    """
    Neural Collaborative Filtering Model.
    
    Architecture:
        User Embedding (64-d) + Item Embedding (64-d)
        -> Concatenate (128-d)
        -> Dense 256 (ReLU + BN + Dropout)
        -> Dense 128 (ReLU + BN + Dropout)
        -> Dense 64  (ReLU + BN + Dropout)
        -> Dense 1   (Sigmoid -> predicted rating 0-1)
    """
    user_in = tf.keras.Input(shape=(1,), name='user_input')
    item_in = tf.keras.Input(shape=(1,), name='item_input')

    user_emb = tf.keras.layers.Embedding(
        num_users, embed_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
        name='user_embedding'
    )(user_in)

    item_emb = tf.keras.layers.Embedding(
        num_items, embed_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
        name='item_embedding'
    )(item_in)

    user_flat = tf.keras.layers.Flatten()(user_emb)
    item_flat = tf.keras.layers.Flatten()(item_emb)

    x = tf.keras.layers.Concatenate()([user_flat, item_flat])

    # Deep MLP
    x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    out = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')(x)

    model = tf.keras.Model(inputs=[user_in, item_in], outputs=out, name='NCF')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model


if __name__ == "__main__":
    m = build_ncf(100, 100)
    m.summary()
