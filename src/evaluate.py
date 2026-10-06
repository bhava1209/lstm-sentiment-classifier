"""
evaluate.py
-----------
Comprehensive model evaluation and performance visualization.

Computes:
- Test Accuracy & Test Loss
- Precision, Recall, and F1-score
- Confusion Matrix & Classification Report
- Training vs. Validation Loss and Accuracy plots (results/accuracy_loss_curve.png)
- Confusion Matrix Heatmap (results/confusion_matrix.png)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "sentiment_lstm.keras")
    processed_dir = os.path.join(base_dir, "data", "processed")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    X_test_path = os.path.join(processed_dir, "X_test.npy")
    y_test_path = os.path.join(processed_dir, "y_test.npy")
    history_path = os.path.join(results_dir, "training_history.json")

    # 1. Validate files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Run train.py first.")
    if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError(f"Test data not found in {processed_dir}. Run train.py first.")

    # 2. Load model and test data
    print(f"[INFO] Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    print(f"[INFO] Loading test data from: {processed_dir}")
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    # 3. Predict on Test Set
    print("[INFO] Computing predictions on 10,000 test reviews...")
    y_pred_probs = model.predict(X_test, batch_size=128, verbose=1).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # 4. Compute Metrics
    loss, test_acc = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    class_names = ["Negative", "Positive"]
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)

    # 5. Display Console Summary
    print("\n" + "=" * 60)
    print("                    EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Loss:        {loss:.4f}")
    print(f"Test Accuracy:    {test_acc * 100:.2f}%")
    print(f"Precision:        {precision * 100:.2f}%")
    print(f"Recall:           {recall * 100:.2f}%")
    print(f"F1-Score:         {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(report)

    # 6. Save text metrics report
    metrics_txt_path = os.path.join(results_dir, "evaluation_metrics.txt")
    with open(metrics_txt_path, "w") as f:
        f.write("LSTM Sentiment Analysis - Model Evaluation Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Loss:     {loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Precision:     {precision * 100:.2f}%\n")
        f.write(f"Recall:        {recall * 100:.2f}%\n")
        f.write(f"F1-Score:      {f1 * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
    print(f"[INFO] Numerical report saved to: {metrics_txt_path}")

    # 7. Plot and Save Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("LSTM Sentiment Classifier - Confusion Matrix", fontsize=13, pad=12)
    plt.xlabel("Predicted Sentiment", fontsize=11)
    plt.ylabel("Actual Sentiment", fontsize=11)
    plt.tight_layout()
    cm_plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Confusion matrix figure saved to: {cm_plot_path}")

    # 8. Plot Training vs. Validation Loss and Accuracy curves
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

        epochs = range(1, len(history["accuracy"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy Subplot
        ax1.plot(epochs, [a * 100 for a in history["accuracy"]], "b-o", label="Training Accuracy", linewidth=2)
        ax1.plot(epochs, [a * 100 for a in history["val_accuracy"]], "g-s", label="Validation Accuracy", linewidth=2)
        ax1.set_title("Training vs. Validation Accuracy", fontsize=12)
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Accuracy (%)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="lower right")

        # Loss Subplot
        ax2.plot(epochs, history["loss"], "b-o", label="Training Loss", linewidth=2)
        ax2.plot(epochs, history["val_loss"], "r-s", label="Validation Loss", linewidth=2)
        ax2.set_title("Training vs. Validation Loss", fontsize=12)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right")

        plt.suptitle("LSTM Model Training Performance", fontsize=14, y=1.02)
        plt.tight_layout()
        curve_plot_path = os.path.join(results_dir, "accuracy_loss_curve.png")
        plt.savefig(curve_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Learning curves figure saved to: {curve_plot_path}")

    print("[SUCCESS] Evaluation and visualizations completed successfully!")


if __name__ == "__main__":
    evaluate_model()
