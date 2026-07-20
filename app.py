"""
Aplikasi web Flask untuk klasifikasi kesegaran buah dan sayur memakai transfer learning Xception.
Jalankan: python app.py
"""

import os
import json
import uuid
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

MODEL_PATH = os.path.join(BASE_DIR, "model", "xception_fruit_freshness.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "model", "class_indices.json")
METRICS_PATH = os.path.join(BASE_DIR, "model", "training_metrics.json")
# Fallback: bobot float16 terkompresi (kecil, aman di-commit ke Git). Dipakai kalau file
# .keras penuh di atas tidak ada, misalnya di server deploy seperti Railway.
COMPACT_WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "xception_fruit_freshness_fp16.npz")
IMG_SIZE = (299, 299)

FRUIT_LABELS_ID = {
    "apple": "Apel", "apples": "Apel",
    "banana": "Pisang", "bananas": "Pisang",
    "bellpepper": "Paprika", "bellpeppers": "Paprika",
    "carrot": "Wortel", "carrots": "Wortel",
    "cucumber": "Timun", "cucumbers": "Timun",
    "mango": "Mangga", "mangoes": "Mangga", "mangos": "Mangga",
    "orange": "Jeruk", "oranges": "Jeruk",
    "potato": "Kentang", "potatoes": "Kentang",
    "strawberry": "Stroberi", "strawberries": "Stroberi",
    "tomato": "Tomat", "tomatoes": "Tomat",
}
CONDITION_LABELS_ID = {"fresh": "Segar", "rotten": "Busuk"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.secret_key = "fruit-freshness-secret-key"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
class_indices = None
index_to_label = {}
class_names = []
model_status_message = ""
metrics = None


def parse_class_name(class_name):
    # Memecah nama kelas seperti FreshApple menjadi kondisi dan jenis item, case insensitive
    name_lower = class_name.lower()

    if name_lower.startswith("fresh"):
        condition = "fresh"
        fruit_key = name_lower.replace("fresh", "", 1)
    elif name_lower.startswith("rotten"):
        condition = "rotten"
        fruit_key = name_lower.replace("rotten", "", 1)
    else:
        condition = "unknown"
        fruit_key = name_lower

    return {
        "condition": condition,
        "condition_display": CONDITION_LABELS_ID.get(condition, condition.title()),
        "fruit_key": fruit_key,
        "fruit_display": FRUIT_LABELS_ID.get(fruit_key, fruit_key.title()),
    }


def load_trained_model():
    # Memuat model dan mapping label dari folder model/.
    # Prioritas 1: file .keras penuh (dipakai untuk development lokal).
    # Prioritas 2: bobot float16 terkompresi (.npz) + rekonstruksi arsitektur, dipakai di
    #              server deploy seperti Railway karena file .keras penuh tidak di-commit ke Git.
    global model, class_indices, index_to_label, class_names, model_status_message

    if not os.path.exists(CLASS_INDEX_PATH):
        model_status_message = "Model belum ditemukan di folder model. Jalankan train_model.py terlebih dahulu."
        print(model_status_message)
        return

    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    index_to_label = {v: k for k, v in class_indices.items()}
    class_names = sorted(class_indices, key=class_indices.get)
    num_classes = len(class_names)

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        model_status_message = "Model berhasil dimuat dan siap digunakan."
        print(model_status_message)
    elif os.path.exists(COMPACT_WEIGHTS_PATH):
        # img_size dibaca dari training_metrics.json kalau ada, supaya arsitektur yang
        # dibangun ulang persis sama dengan resolusi saat training. Kalau belum ada, pakai
        # IMG_SIZE default modul ini (299, 299).
        img_size = IMG_SIZE
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                saved_metrics = json.load(f)
            if "img_size" in saved_metrics:
                img_size = tuple(saved_metrics["img_size"])

        from train_model import build_model
        model, _ = build_model(num_classes, img_size=img_size)

        with np.load(COMPACT_WEIGHTS_PATH) as data:
            weights_fp16 = [data[key] for key in sorted(data.files, key=lambda k: int(k.split("_")[1]))]
        model.set_weights(weights_fp16)

        model_status_message = "Model berhasil dimuat dari bobot ringkas (float16) dan siap digunakan."
        print(model_status_message)
    else:
        model_status_message = "Model belum ditemukan di folder model. Jalankan train_model.py terlebih dahulu."
        print(model_status_message)


def load_metrics():
    # Memuat training_metrics.json dan menyesuaikan IMG_SIZE mengikuti resolusi saat training
    global metrics, IMG_SIZE
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        if "img_size" in metrics:
            IMG_SIZE = tuple(metrics["img_size"])
        print(f"Metrik training dimuat, resolusi model {IMG_SIZE}.")
    else:
        metrics = None
        print("Belum ada training_metrics.json.")


load_trained_model()
load_metrics()


def allowed_file(filename):
    # Mengecek ekstensi file yang diizinkan
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_uploaded_image(filepath):
    # Mengubah file gambar jadi array siap pakai model
    img = keras_image.load_img(filepath, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_freshness(filepath):
    # Menjalankan klasifikasi satu gambar, mengembalikan kelas teratas dan semua probabilitas
    img_array = preprocess_uploaded_image(filepath)
    probabilities = model.predict(img_array, verbose=0)[0]

    sorted_indices = np.argsort(probabilities)[::-1]
    predicted_index = int(sorted_indices[0])
    predicted_class = index_to_label.get(predicted_index, "unknown")
    confidence = float(probabilities[predicted_index]) * 100

    parsed = parse_class_name(predicted_class)

    all_probs = []
    for idx, class_name in enumerate(class_names):
        info = parse_class_name(class_name)
        all_probs.append({
            "class_name": class_name,
            "display": f"{info['fruit_display']} {info['condition_display']}",
            "probability": round(float(probabilities[idx]) * 100, 2),
            "condition": info["condition"],
        })
    all_probs.sort(key=lambda item: item["probability"], reverse=True)

    # Selisih probabilitas kelas teratas dan kelas kedua, dipakai untuk narasi keyakinan model
    runner_up_gap = None
    if len(all_probs) > 1:
        runner_up_gap = round(all_probs[0]["probability"] - all_probs[1]["probability"], 2)

    return {
        "predicted_class": predicted_class,
        "condition": parsed["condition"],
        "condition_display": parsed["condition_display"],
        "fruit_display": parsed["fruit_display"],
        "confidence": round(confidence, 2),
        "all_probabilities": all_probs,
        "runner_up_gap": runner_up_gap,
        "runner_up_display": all_probs[1]["display"] if len(all_probs) > 1 else None,
    }


def build_class_badges():
    # Membangun daftar badge untuk semua kelas yang dikenali model
    badges = []
    for name in class_names:
        parsed = parse_class_name(name)
        badges.append({"display": f"{parsed['fruit_display']} {parsed['condition_display']}", "condition": parsed["condition"]})
    return badges


def build_unique_item_names():
    # Mengembalikan daftar nama item unik tanpa duplikasi fresh/rotten
    seen = []
    for name in class_names:
        parsed = parse_class_name(name)
        if parsed["fruit_display"] not in seen:
            seen.append(parsed["fruit_display"])
    return seen


def compute_top_confusions(top_n=5):
    # Mencari pasangan kelas yang paling sering tertukar dari confusion matrix
    if not metrics or not metrics.get("confusion_matrix"):
        return []

    cm = metrics["confusion_matrix"]
    names = metrics["class_names"]
    confusions = []

    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i][j] > 0:
                true_info = parse_class_name(names[i])
                pred_info = parse_class_name(names[j])
                confusions.append({
                    "true_class": f"{true_info['fruit_display']} {true_info['condition_display']}",
                    "predicted_class": f"{pred_info['fruit_display']} {pred_info['condition_display']}",
                    "count": cm[i][j],
                })

    confusions.sort(key=lambda item: item["count"], reverse=True)
    return confusions[:top_n]


@app.route("/")
def index():
    # Halaman dashboard utama
    total_train_images = 0
    total_test_images = 0
    if metrics:
        total_train_images = sum(metrics["dataset_counts"].get("train", {}).values())
        total_test_images = sum(metrics["dataset_counts"].get("test", {}).values())

    return render_template(
        "index.html",
        model_ready=(model is not None),
        status=model_status_message,
        metrics=metrics,
        total_train_images=total_train_images,
        total_test_images=total_test_images,
        num_classes=len(class_names) if class_names else 0,
        class_badges=build_class_badges(),
        unique_items=build_unique_item_names(),
    )


@app.route("/predict", methods=["GET"])
def predict_page():
    # Menampilkan halaman form unggah gambar
    return render_template("predict.html", model_ready=(model is not None), status=model_status_message, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    # Menerima file gambar, menjalankan klasifikasi, menampilkan hasil
    if model is None:
        flash("Model belum tersedia. Jalankan train_model.py terlebih dahulu.", "danger")
        return redirect(url_for("predict_page"))

    if "fruit_image" not in request.files:
        flash("Tidak ada file yang diunggah.", "warning")
        return redirect(url_for("predict_page"))

    file = request.files["fruit_image"]

    if file.filename == "":
        flash("Silakan pilih gambar terlebih dahulu.", "warning")
        return redirect(url_for("predict_page"))

    if not allowed_file(file.filename):
        flash("Format file tidak didukung. Gunakan PNG, JPG, JPEG, atau WEBP.", "warning")
        return redirect(url_for("predict_page"))

    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(filepath)

    try:
        prediction = predict_freshness(filepath)
    except Exception as error:
        flash(f"Terjadi kesalahan saat memproses gambar: {error}", "danger")
        return redirect(url_for("predict_page"))

    result_data = {
        "image_url": url_for("static", filename=f"uploads/{unique_filename}"),
        "condition": prediction["condition"],
        "condition_display": prediction["condition_display"],
        "fruit_display": prediction["fruit_display"],
        "confidence": prediction["confidence"],
        "all_probabilities": prediction["all_probabilities"],
        "runner_up_gap": prediction["runner_up_gap"],
        "runner_up_display": prediction["runner_up_display"],
        "timestamp": datetime.now().strftime("%d %B %Y, %H:%M"),
    }

    return render_template("predict.html", model_ready=True, status=model_status_message, result=result_data)


@app.route("/model")
def model_info():
    # Halaman detail arsitektur, proses training, dan evaluasi model
    confusion_max = 0
    if metrics and metrics.get("confusion_matrix"):
        confusion_max = max(max(row) for row in metrics["confusion_matrix"])

    confusion_image_path = os.path.join(BASE_DIR, "static", "reports", "confusion_matrix.png")
    confusion_image_exists = os.path.exists(confusion_image_path)

    return render_template(
        "model_info.html",
        model_ready=(model is not None),
        status=model_status_message,
        metrics=metrics,
        class_names=class_names,
        confusion_max=confusion_max,
        top_confusions=compute_top_confusions(),
        confusion_image_exists=confusion_image_exists,
    )


@app.route("/dataset")
def dataset_info():
    # Halaman informasi dan komposisi dataset
    return render_template(
        "dataset.html",
        metrics=metrics,
        model_ready=(model is not None),
        status=model_status_message,
        class_badges=build_class_badges(),
    )


@app.route("/about")
def about():
    # Halaman latar belakang dan metodologi proyek
    return render_template("about.html", model_ready=(model is not None), status=model_status_message)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
