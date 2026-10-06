"""
download_data.py
----------------
Module to download the IMDB 50k Movie Reviews dataset (CSV format)
into the data/raw/ directory for sentiment classification training.
"""

import os
import urllib.request
import pandas as pd


DATA_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
DEST_FILE = os.path.join(RAW_DATA_DIR, "imdb_reviews.csv")


def download_dataset(url: str = DATA_URL, destination: str = DEST_FILE) -> str:
    """
    Downloads the dataset from the URL if not already present,
    and validates its structure.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if os.path.exists(destination):
        print(f"[INFO] Dataset already exists at: {destination}")
    else:
        print(f"[INFO] Downloading dataset from: {url}")
        print("[INFO] This may take a minute (approx. 66 MB)...")
        urllib.request.urlretrieve(url, destination)
        print(f"[SUCCESS] Dataset downloaded and saved to: {destination}")

    # Validate dataset contents
    df = pd.read_csv(destination)
    print("\n--- Dataset Summary ---")
    print(f"Total Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nClass Distribution:")
    print(df['sentiment'].value_counts())
    print("\nFirst sample preview:")
    print(f"Review: {df['review'].iloc[0][:200]}...")
    print(f"Label: {df['sentiment'].iloc[0]}")

    return destination


if __name__ == "__main__":
    download_dataset()
