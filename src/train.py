"""
train.py
--------
Training script for the LSTM Sentiment Classifier.

Includes:
- Loading preprocessed dataset splits
- Model building and compilation
- Keras callbacks (EarlyStopping, ModelCheckpoint)
- Training & validation history logging
- Serializing the best model to models/sentiment_lstm.keras
"""

import os
import json
import pickle
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocess import load_and_preprocess_data
from model import build_sentiment_lstm_model, compile_model


def train():
    # 1. Define paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_data_file = os.path.join(base_dir, "data", "raw", "imdb_reviews.csv")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_save_path = os.path.join(models_dir, "sentiment_lstm.keras")
    tokenizer_save_path = os.path.join(models_dir, "tokenizer.pickle")
    history_save_path = os.path.join(results_dir, "training_history.json")

    # 2. Hyperparameters
    VOCAB_SIZE = 10000
    MAX_LEN = 200
    EMBEDDING_DIM = 64
    LSTM_UNITS = 64
    DENSE_UNITS = 32
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 5

    print("=" * 60)
    print("STAGE: PREPROCESSING & DATA PREPARATION")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = load_and_preprocess_data(
        data_path=raw_data_file,
        num_words=VOCAB_SIZE,
        max_len=MAX_LEN,
        test_size=0.20,
        val_size=0.10,
        random_state=42,
        save_tokenizer_path=tokenizer_save_path
    )

    # Cache preprocessed test set for evaluate.py
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    print(f"[INFO] Cached test data to {processed_dir} for evaluation.")

    print("\n" + "=" * 60)
    print("STAGE: MODEL CONSTRUCTION & COMPILATION")
    print("=" * 60)
    model = build_sentiment_lstm_model(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_LEN,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE
    )
    model = compile_model(model, learning_rate=LEARNING_RATE)
    model.summary()

    # 3. Callbacks for robust training
    callbacks = [
        # EarlyStopping: halts training if validation loss does not improve for 3 consecutive epochs
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        # ModelCheckpoint: saves only the model with the highest validation accuracy
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        # ReduceLROnPlateau: decreases learning rate if loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print("\n" + "=" * 60)
    print("STAGE: TRAINING LSTM CLASSIFIER")
    print("=" * 60)
    print(f"Batch Size: {BATCH_SIZE} | Epochs: {EPOCHS}")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history for visualization
    history_dict = {
        "accuracy": [float(x) for x in history.history.get("accuracy", [])],
        "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        "loss": [float(x) for x in history.history.get("loss", [])],
        "val_loss": [float(x) for x in history.history.get("val_loss", [])]
    }
    with open(history_save_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"\n[INFO] Training history saved to: {history_save_path}")
    print(f"[SUCCESS] Best model checkpoint saved to: {model_save_path}")


if __name__ == "__main__":
    train()
