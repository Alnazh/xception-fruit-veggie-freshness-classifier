"""
Membuat ulang gambar confusion matrix dari model/training_metrics.json.
Dipakai kalau gambar lama tidak terbaca jelas, tanpa perlu training ulang model.
Jalankan: python generate_confusion_matrix.py
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_PATH = os.path.join("model", "training_metrics.json")
OUTPUT_PATH = os.path.join("static", "reports", "confusion_matrix.png")


def main():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    class_names = metrics["class_names"]
    cm = metrics["confusion_matrix"]
    num_classes = len(class_names)
    max_val = max(max(row) for row in cm)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.6), max(5, num_classes * 0.5)))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label Asli")
    ax.set_title("Confusion Matrix")

    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i][j]
            # Teks putih di kotak gelap, teks hitam di kotak terang, supaya angka selalu terbaca
            text_color = "white" if value > max_val * 0.5 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=7)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)

    print(f"Confusion matrix tersimpan ulang di {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
