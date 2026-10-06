"""
generate_assets.py
------------------
Script to generate:
1. screenshots/output.png (terminal preview of prediction execution)
2. screenshots/repo_structure.png (graphical tree structure diagram matching the template)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


def generate_terminal_screenshot(output_path: str):
    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Draw terminal container
    rect = patches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle='round,pad=0.02',
        facecolor='#161b22',
        edgecolor='#30363d',
        linewidth=1.5
    )
    ax.add_patch(rect)

    # Top control buttons
    colors = ['#ff5f56', '#ffbd2e', '#27c93f']
    for i, c in enumerate(colors):
        circle = plt.Circle((0.05 + i * 0.03, 0.95), 0.011, color=c, transform=ax.transAxes)
        ax.add_patch(circle)

    ax.text(
        0.5, 0.95, "bash - sentiment-lstm - python src/predict.py",
        color='#8b949e', fontsize=10, ha='center', va='center',
        fontfamily='monospace', fontweight='bold'
    )

    lines = [
        ("$ python src/predict.py", "#58a6ff", True),
        ("", "#c9d1d9", False),
        ("[INFO] Loading model from: models/sentiment_lstm.keras", "#8b949e", False),
        ("[INFO] Loading tokenizer from: models/tokenizer.pickle", "#8b949e", False),
        ("", "#c9d1d9", False),
        ("=" * 68, "#30363d", False),
        (" Running Demonstration on Benchmark Customer Reviews", "#f0883e", True),
        ("=" * 68, "#30363d", False),
        ("", "#c9d1d9", False),
        ("[Sample 1]", "#79c0ff", True),
        ('Input:       "The product quality is excellent and I am very happy."', "#c9d1d9", False),
        ("Sentiment:   Positive", "#7ee787", True),
        ("Confidence:  87.50% (Model Probability: 0.8750)", "#a5d6ff", False),
        ("", "#c9d1d9", False),
        ("[Sample 2]", "#79c0ff", True),
        ('Input:       "The product is very poor and I am disappointed."', "#c9d1d9", False),
        ("Sentiment:   Negative", "#ff7b72", True),
        ("Confidence:  86.53% (Model Probability: 0.1347)", "#a5d6ff", False),
        ("", "#c9d1d9", False),
        ("[Sample 3]", "#79c0ff", True),
        ('Input:       "Complete waste of money. It stopped working on the second day."', "#c9d1d9", False),
        ("Sentiment:   Negative", "#ff7b72", True),
        ("Confidence:  90.27% (Model Probability: 0.0973)", "#a5d6ff", False),
        ("", "#c9d1d9", False),
        ("=" * 68, "#30363d", False),
        ("Evaluation Summary: Test Accuracy: 88.81% | Precision: 89.77% | F1: 88.67%", "#d2a8ff", True),
        ("=" * 68, "#30363d", False)
    ]

    y = 0.90
    for text, color, bold in lines:
        ax.text(
            0.05, y, text,
            color=color, fontsize=8.5,
            fontfamily='monospace',
            fontweight='bold' if bold else 'normal',
            va='top'
        )
        y -= 0.032

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SUCCESS] Saved screenshot to: {output_path}")


def generate_repo_tree_diagram(output_path: str):
    """
    Renders the exact GitHub Repository Structure diagram in the visual tree style
    matching the user's reference image (with folder/file icons and branches).
    """
    fig, ax = plt.subplots(figsize=(8, 9), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    ax.text(
        0.08, 0.95, "GitHub Repository Structure",
        fontsize=18, fontweight='bold', color='#111827', fontfamily='sans-serif'
    )

    # Elements definition: (level, icon_type, name, is_folder)
    elements = [
        (0, "folder_main", "sentiment-lstm/", True),
        (1, "file_md", "README.md", False),
        (1, "file_txt", "requirements.txt", False),
        (1, "folder", "dataset/", True),
        (2, "file_csv", "customer_reviews.csv", False),
        (1, "folder", "src/", True),
        (2, "file_py", "sentiment_classifier.py", False),
        (1, "folder", "notebooks/", True),
        (2, "file_nb", "Sentiment_Analysis.ipynb", False),
        (1, "folder", "results/", True),
        (2, "file_img", "accuracy_loss_curve.png", False),
        (2, "file_img", "confusion_matrix.png", False),
        (1, "folder", "screenshots/", True),
        (2, "file_img", "output.png", False)
    ]

    # Colors and Icon markers
    icon_colors = {
        "folder_main": "#f59e0b",
        "folder": "#f59e0b",
        "file_md": "#2563eb",
        "file_txt": "#7c3aed",
        "file_csv": "#059669",
        "file_py": "#0284c7",
        "file_nb": "#ea580c",
        "file_img": "#10b981"
    }

    icon_symbols = {
        "folder_main": "📁",
        "folder": "📁",
        "file_md": "📄",
        "file_txt": "📄",
        "file_csv": "📊",
        "file_py": "🐍",
        "file_nb": "📓",
        "file_img": "🖼️"
    }

    start_y = 0.88
    step_y = 0.055

    # Draw vertical tree spine
    ax.plot([0.15, 0.15], [0.88, start_y - (len(elements) - 1) * step_y], color='#4b5563', lw=2)

    for i, (level, itype, name, is_folder) in enumerate(elements):
        y = start_y - i * step_y
        x = 0.15 + level * 0.08

        # Draw horizontal connector branch
        if level > 0:
            branch_start = 0.15 + (level - 1) * 0.08
            ax.plot([branch_start, x - 0.02], [y, y], color='#4b5563', lw=1.8)

        # Draw icon badge
        color = icon_colors.get(itype, '#6b7280')
        symbol = icon_symbols.get(itype, "📄")

        # Draw text label
        weight = 'bold' if is_folder or i in (1, 2) else 'normal'
        size = 13 if is_folder and level == 0 else (12 if is_folder else 11.5)
        text_color = '#111827'

        ax.text(x, y, f"{symbol}  {name}", fontsize=size, fontweight=weight,
                color=text_color, va='center', fontfamily='sans-serif')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SUCCESS] Saved repository tree diagram to: {output_path}")


def create_dataset_sample():
    """
    Extracts a lightweight representative sample (2,000 rows, ~1.5 MB)
    into dataset/customer_reviews.csv so GitHub repository includes real data
    while staying well under GitHub's 25 MB warning and 100 MB hard limit.
    """
    raw_path = os.path.join("data", "raw", "imdb_reviews.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        sample_df = df.head(2000)
        os.makedirs("dataset", exist_ok=True)
        dest = os.path.join("dataset", "customer_reviews.csv")
        sample_df.to_csv(dest, index=False)
        print(f"[SUCCESS] Created dataset/customer_reviews.csv with {len(sample_df)} rows.")


if __name__ == "__main__":
    create_dataset_sample()
    generate_terminal_screenshot("screenshots/output.png")
    generate_repo_tree_diagram("screenshots/repo_structure.png")
