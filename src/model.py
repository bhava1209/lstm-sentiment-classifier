"""
model.py
--------
Defines the LSTM Neural Network Architecture for Binary Sentiment Classification.

Architecture:
  Input (Token IDs) [batch_size, max_len]
        │
  Embedding Layer   [batch_size, max_len, embedding_dim]
        │
  SpatialDropout1D  [batch_size, max_len, embedding_dim] (Feature-map regularization)
        │
  LSTM Layer        [batch_size, lstm_units] (Captures sequential & temporal word dependencies)
        │
  Dense Layer       [batch_size, dense_units] (Non-linear feature combination with ReLU)
        │
  Dropout Layer     [batch_size, dense_units] (Prevents overfitting)
        │
  Output Layer      [batch_size, 1] (Sigmoid activation: probability of positive sentiment)
"""

import keras
from keras import layers, models, optimizers


def build_sentiment_lstm_model(
    vocab_size: int = 10000,
    embedding_dim: int = 64,
    max_len: int = 200,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.3
) -> keras.Model:
    """
    Constructs an LSTM sentiment classification model using Keras Functional API.

    Parameters:
      vocab_size: Total number of unique tokens in the dictionary.
      embedding_dim: Dimension of the dense word embedding vectors.
      max_len: Length of the input integer sequences.
      lstm_units: Number of memory units in the LSTM layer.
      dense_units: Number of hidden units in the intermediate dense layer.
      dropout_rate: Dropout fraction for regularization.

    Returns:
      keras.Model: Uncompiled Keras model instance.
    """
    # 1. Input Layer: Accepts fixed-length integer token ID sequences
    input_layer = layers.Input(shape=(max_len,), dtype="int32", name="token_input")

    # 2. Embedding Layer: Converts sparse word IDs into dense 64-D vectors
    embedding_layer = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="word_embedding"
    )(input_layer)

    # 3. Spatial Dropout: Drops 1D feature maps to prevent embedding co-adaptation
    spatial_dropout = layers.SpatialDropout1D(
        rate=dropout_rate,
        name="spatial_dropout"
    )(embedding_layer)

    # 4. LSTM Layer: Captures context and temporal sequence dynamics
    lstm_layer = layers.LSTM(
        units=lstm_units,
        dropout=dropout_rate,
        recurrent_dropout=0.0,
        name="lstm_layer"
    )(spatial_dropout)

    # 5. Intermediate Dense Layer: Non-linear feature combination
    dense_layer = layers.Dense(
        units=dense_units,
        activation="relu",
        name="dense_layer"
    )(lstm_layer)

    # 6. Dropout: Regularization before the final classification
    dropout_layer = layers.Dropout(
        rate=dropout_rate,
        name="classification_dropout"
    )(dense_layer)

    # 7. Output Layer: Sigmoid activation squashes output into [0, 1] probability
    output_layer = layers.Dense(
        units=1,
        activation="sigmoid",
        name="sentiment_output"
    )(dropout_layer)

    model = models.Model(inputs=input_layer, outputs=output_layer, name="Sentiment_LSTM_Classifier")
    return model


def compile_model(model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
    """
    Compiles the model with Adam optimizer, Binary Cross-Entropy loss,
    and Accuracy metric.
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing LSTM Model Architecture Definition")
    print("=" * 60)

    model = build_sentiment_lstm_model(
        vocab_size=10000,
        embedding_dim=64,
        max_len=200,
        lstm_units=64,
        dense_units=32,
        dropout_rate=0.3
    )
    model = compile_model(model, learning_rate=0.001)

    model.summary()
    print("\n[SUCCESS] Model built and compiled successfully!")
