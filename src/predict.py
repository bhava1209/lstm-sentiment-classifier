"""
predict.py
----------
Inference module for classifying new customer comments and reviews.

Features:
- Loads the trained LSTM model (models/sentiment_lstm.keras)
- Loads the serialized ReviewTokenizer (models/tokenizer.pickle)
- Preprocesses and cleans raw user strings on the fly
- Outputs binary sentiment (Positive / Negative) with confidence % and raw probability
- Supports programmatic API, batch test demo, and interactive terminal input
"""

import os
import argparse
from typing import Dict, Any, List

import numpy as np
import keras
from preprocess import ReviewTokenizer, clean_text


class SentimentPredictor:
    """
    Production-ready wrapper for real-time sentiment inference.
    """

    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "sentiment_lstm.keras")
        if tokenizer_path is None:
            tokenizer_path = os.path.join(base_dir, "models", "tokenizer.pickle")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}. Run preprocessing first.")

        print(f"[INFO] Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)

        print(f"[INFO] Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = ReviewTokenizer.load(tokenizer_path)

    def predict(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Classifies a single text string.

        Returns:
            dict containing text, cleaned_text, sentiment, confidence_pct, and probability.
        """
        cleaned = clean_text(text)
        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = self.tokenizer.pad(seq)

        raw_prob = float(self.model.predict(padded, verbose=0)[0][0])

        if raw_prob >= threshold:
            sentiment = "Positive"
            confidence = raw_prob * 100.0
        else:
            sentiment = "Negative"
            confidence = (1.0 - raw_prob) * 100.0

        return {
            "text": text,
            "cleaned_text": cleaned,
            "sentiment": sentiment,
            "confidence": confidence,
            "probability": raw_prob
        }

    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Classifies a list of text strings efficiently.
        """
        cleaned_list = [clean_text(t) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(cleaned_list)
        padded = self.tokenizer.pad(seqs)

        raw_probs = self.model.predict(padded, batch_size=len(texts), verbose=0).flatten()

        results = []
        for text, cleaned, prob in zip(texts, cleaned_list, raw_probs):
            prob = float(prob)
            if prob >= threshold:
                sentiment = "Positive"
                confidence = prob * 100.0
            else:
                sentiment = "Negative"
                confidence = (1.0 - prob) * 100.0

            results.append({
                "text": text,
                "cleaned_text": cleaned,
                "sentiment": sentiment,
                "confidence": confidence,
                "probability": prob
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis LSTM Inference")
    parser.add_argument("--text", type=str, help="Single review text to analyze")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI loop")
    args = parser.parse_args()

    predictor = SentimentPredictor()

    if args.text:
        res = predictor.predict(args.text)
        print("\n" + "=" * 50)
        print(f"Review:      {res['text']}")
        print(f"Sentiment:   {res['sentiment']}")
        print(f"Confidence:  {res['confidence']:.2f}% (Prob: {res['probability']:.4f})")
        print("=" * 50)

    elif args.interactive:
        print("\n" + "=" * 60)
        print(" Interactive Sentiment Classifier (type 'exit' or 'quit' to stop)")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nEnter customer review: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting predictor. Goodbye!")
                    break
                res = predictor.predict(user_input)
                print(f"-> Sentiment:   {res['sentiment']}")
                print(f"-> Confidence:  {res['confidence']:.2f}% (Raw Score: {res['probability']:.4f})")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        # Default Demonstration with test cases from requirements
        print("\n" + "=" * 70)
        print(" Running Demonstration on Benchmark Customer Reviews")
        print("=" * 70)
        demo_reviews = [
            "The product quality is excellent and I am very happy.",
            "The product is very poor and I am disappointed.",
            "Super fast delivery! The build quality surpassed my expectations.",
            "Complete waste of money. It stopped working on the second day.",
            "Brilliant acting and stunning cinematography, a true cinematic masterpiece.",
            "The movie was utterly boring, dreadful plot and terrible direction."
        ]

        results = predictor.predict_batch(demo_reviews)
        for i, res in enumerate(results, 1):
            print(f"\n[Sample {i}]")
            print(f"Input:       \"{res['text']}\"")
            print(f"Sentiment:   {res['sentiment']}")
            print(f"Confidence:  {res['confidence']:.2f}% (Model Probability: {res['probability']:.4f})")
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
