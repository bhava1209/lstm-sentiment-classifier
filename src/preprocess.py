"""
preprocess.py
-------------
Text preprocessing and tokenization pipeline for the LSTM Sentiment Classifier.

Pipeline Steps:
1. Text Cleaning: Lowercase, HTML removal, punctuation/special character stripping.
2. Tokenization & Vocabulary: Word frequency mapping with Out-Of-Vocabulary (<OOV>) support.
3. Sequence Padding/Truncation: Uniform sequence length for LSTM tensor batches.
4. Stratified Train/Val/Test Split: Prevents data leakage and ensures balanced sets.
5. Tokenizer Serialization: Saves the fitted tokenizer to models/tokenizer.pickle.
"""

import os
import re
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences


def clean_text(text: str) -> str:
    """
    Cleans raw review text for natural language modeling.
    - Converts text to lowercase.
    - Removes HTML tags (e.g., <br />, <div>).
    - Removes URLs and hyperlinks.
    - Strips special characters, numbers, and excess punctuation.
    - Collapses multiple whitespace characters.
    """
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)                      # Remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)                # Keep letters only
    text = re.sub(r"\s+", " ", text).strip()                 # Normalize spaces
    return text


class ReviewTokenizer:
    """
    Custom vocabulary builder and sequence tokenizer.
    Assigns:
      - 0 to <PAD> (padding token)
      - 1 to <OOV> (out-of-vocabulary words)
      - 2 to num_words for the most frequent words.
    """

    def __init__(self, num_words: int = 10000, max_len: int = 200, oov_token: str = "<OOV>"):
        self.num_words = num_words
        self.max_len = max_len
        self.oov_token = oov_token
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}

    def fit_on_texts(self, texts: List[str]):
        """
        Builds vocabulary from a collection of raw text documents.
        """
        counter = Counter()
        for text in texts:
            tokens = clean_text(text).split()
            counter.update(tokens)

        # Reserve 0 for <PAD> and 1 for <OOV>
        self.word_index = {self.oov_token: 1}
        # Take the top (num_words - 2) most frequent words
        most_common = counter.most_common(self.num_words - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_index[word] = idx

        # Reverse mapping: integer ID to word
        self.index_word = {idx: word for word, idx in self.word_index.items()}
        self.index_word[0] = "<PAD>"

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Transforms texts into integer ID sequences using the learned vocabulary.
        """
        sequences = []
        oov_idx = self.word_index.get(self.oov_token, 1)
        for text in texts:
            tokens = clean_text(text).split()
            seq = [self.word_index.get(token, oov_idx) for token in tokens]
            sequences.append(seq)
        return sequences

    def pad(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Pads or truncates sequences to a uniform length (max_len).
        Uses 'pre' padding so that zero-padding appears at the start of the sequence,
        allowing the LSTM's final recurrent state to capture the actual words.
        """
        return pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="pre",
            truncating="pre"
        )

    def sequences_to_text(self, sequence: List[int]) -> str:
        """
        Decodes a sequence of integer IDs back into words for inspection.
        """
        return " ".join([self.index_word.get(idx, self.oov_token) for idx in sequence if idx != 0])

    def save(self, file_path: str):
        """
        Serializes tokenizer instance to disk via pickle.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"[INFO] Tokenizer saved successfully to: {file_path}")

    @staticmethod
    def load(file_path: str) -> "ReviewTokenizer":
        """
        Loads a serialized tokenizer instance from disk.
        """
        with open(file_path, "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer


def load_and_preprocess_data(
    data_path: str,
    num_words: int = 10000,
    max_len: int = 200,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    save_tokenizer_path: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ReviewTokenizer]:
    """
    Loads dataset, performs text preprocessing, fits tokenizer, pads sequences,
    and returns train/validation/test arrays along with the tokenizer.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}. Run download_data.py first.")

    print(f"[INFO] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Convert binary sentiment labels to integers: positive -> 1, negative -> 0
    label_map = {"positive": 1, "negative": 0}
    df["label"] = df["sentiment"].map(label_map)

    texts = df["review"].values
    labels = df["label"].values.astype(np.float32)

    # 1st split: Separate Test set (e.g. 20% -> 10,000 samples)
    X_temp, X_test_raw, y_temp, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    # 2nd split: From the remaining 80%, separate Validation set (val_size / (1 - test_size))
    # 0.10 / 0.80 = 0.125 of remaining -> exactly 5,000 validation samples, leaving 35,000 train samples
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=y_temp
    )

    print(f"[INFO] Dataset splits:")
    print(f"       Train set:      {len(X_train_raw)} samples")
    print(f"       Validation set: {len(X_val_raw)} samples")
    print(f"       Test set:       {len(X_test_raw)} samples")

    # Fit Tokenizer ONLY on training set to avoid data leakage
    print(f"[INFO] Fitting Tokenizer on training text (vocab size: {num_words}, max_len: {max_len})...")
    tokenizer = ReviewTokenizer(num_words=num_words, max_len=max_len)
    tokenizer.fit_on_texts(X_train_raw)

    # Transform all splits to padded integer sequences
    X_train = tokenizer.pad(tokenizer.texts_to_sequences(X_train_raw))
    X_val = tokenizer.pad(tokenizer.texts_to_sequences(X_val_raw))
    X_test = tokenizer.pad(tokenizer.texts_to_sequences(X_test_raw))

    if save_tokenizer_path:
        tokenizer.save(save_tokenizer_path)

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_data_file = os.path.join(base_dir, "data", "raw", "imdb_reviews.csv")
    tokenizer_file = os.path.join(base_dir, "models", "tokenizer.pickle")

    print("=" * 60)
    print("Testing Text Preprocessing & Tokenization Pipeline")
    print("=" * 60)

    # Test individual text cleaning
    sample_dirty = "Awesome movie!! Loved every bit of it.<br /><br />Visit http://imdb.com for more."
    sample_clean = clean_text(sample_dirty)
    print(f"Raw text:     '{sample_dirty}'")
    print(f"Cleaned text: '{sample_clean}'\n")

    # Run full dataset preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, tok = load_and_preprocess_data(
        data_path=raw_data_file,
        num_words=10000,
        max_len=200,
        save_tokenizer_path=tokenizer_file
    )

    print("\n--- Preprocessed Tensor Shapes ---")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    print("\n--- Sample Tokenized Sequence (First 15 IDs of sample 0) ---")
    print("Sequence IDs:", X_train[0][:15])
    print("Decoded back:", tok.sequences_to_text(X_train[0][:15]))
    print("\n[SUCCESS] Preprocessing pipeline verified successfully!")
