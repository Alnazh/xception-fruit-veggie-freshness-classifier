"""
Training script transfer learning Xception untuk klasifikasi kesegaran buah dan sayur.
Mendukung dataset flat (folder kelas langsung) atau yang sudah dipisah train/test.
Jalankan: python train_model.py
"""

import os
import json
import random
import shutil
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report


# Mode cepat untuk training di CPU tanpa GPU: resolusi lebih kecil, epoch lebih sedikit
FAST_TRAINING = True

if FAST_TRAINING:
    IMG_SIZE = (224, 224)
    EPOCHS_FEATURE_EXTRACTION = 40
    EPOCHS_FINE_TUNING = 4
    FINE_TUNE_LAYERS = 30
else:
    IMG_SIZE = (299, 299)
    EPOCHS_FEATURE_EXTRACTION = 12
    EPOCHS_FINE_TUNING = 8
    FINE_TUNE_LAYERS = 30

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT_RATIO = 0.1
RANDOM_SEED = 42

DATA_LOADING_WORKERS = 4
USE_MULTIPROCESSING = False

RAW_DATASET_DIR = "dataset"
SPLIT_DATASET_DIR = "dataset_split"

MODEL_DIR = "model"
REPORT_DIR = os.path.join("static", "reports")
MODEL_PATH = os.path.join(MODEL_DIR, "xception_fruit_freshness.keras")
CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_indices.json")
METRICS_PATH = os.path.join(MODEL_DIR, "training_metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def list_class_folders(root_dir):
    # Mengembalikan daftar nama sub folder kelas di dalam root_dir
    if not os.path.isdir(root_dir):
        return []
    return sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])


def split_flat_dataset_if_needed():
    # Mendeteksi struktur dataset dan mengembalikan path train_dir, test_dir yang siap dipakai
    train_sub = os.path.join(RAW_DATASET_DIR, "train")
    test_sub = os.path.join(RAW_DATASET_DIR, "test")

    if os.path.isdir(train_sub) and os.path.isdir(test_sub):
        print(f"Dataset sudah terpisah train/test: {train_sub}, {test_sub}")
        return train_sub, test_sub

    split_train = os.path.join(SPLIT_DATASET_DIR, "train")
    split_test = os.path.join(SPLIT_DATASET_DIR, "test")

    if os.path.isdir(split_train) and os.path.isdir(split_test) and os.listdir(split_train):
        print(f"Memakai hasil split otomatis sebelumnya: {split_train}, {split_test}")
        return split_train, split_test

    print(f"Dataset flat terdeteksi, split otomatis {int((1 - TEST_SPLIT_RATIO) * 100)}:{int(TEST_SPLIT_RATIO * 100)} ke {SPLIT_DATASET_DIR}/")

    class_folders = list_class_folders(RAW_DATASET_DIR)
    if not class_folders:
        raise FileNotFoundError(f"Tidak ada folder kelas di dalam {RAW_DATASET_DIR}/")

    rng = random.Random(RANDOM_SEED)

    for class_name in class_folders:
        src_dir = os.path.join(RAW_DATASET_DIR, class_name)
        images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        rng.shuffle(images)

        split_point = int(len(images) * (1 - TEST_SPLIT_RATIO))
        train_images = images[:split_point]
        test_images = images[split_point:]

        train_dest = os.path.join(split_train, class_name)
        test_dest = os.path.join(split_test, class_name)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(test_dest, exist_ok=True)

        for filename in train_images:
            shutil.copy2(os.path.join(src_dir, filename), os.path.join(train_dest, filename))
        for filename in test_images:
            shutil.copy2(os.path.join(src_dir, filename), os.path.join(test_dest, filename))

        print(f"  {class_name}: {len(train_images)} train, {len(test_images)} test")

    return split_train, split_test


def count_images_per_class(root_dir):
    # Menghitung jumlah gambar per folder kelas
    counts = {}
    if not os.path.exists(root_dir):
        return counts
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            counts[class_name] = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
    return counts


def build_data_generators(train_dir, test_dir):
    # Membuat generator training (dengan augmentasi), validasi, dan test
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        validation_split=VALIDATION_SPLIT,
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", shuffle=True, seed=RANDOM_SEED,
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", shuffle=False, seed=RANDOM_SEED,
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )

    return train_generator, validation_generator, test_generator


def build_model(num_classes):
    # Membangun model Xception dengan classifier head baru untuk num_classes kelas
    base_model = Xception(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    return model, base_model


def plot_history(history, filename, title_suffix):
    # Menyimpan grafik akurasi dan loss ke file PNG
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title(f"Akurasi Model {title_suffix}")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title(f"Loss Model {title_suffix}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, filename))
    plt.close(fig)


def history_to_dict(history):
    # Mengubah objek History Keras jadi dict biasa agar bisa disimpan ke JSON
    return {
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }


def main():
    print("Training model: Transfer Learning Xception, klasifikasi kesegaran buah dan sayur")

    train_dir, test_dir = split_flat_dataset_if_needed()

    train_counts = count_images_per_class(train_dir)
    test_counts = count_images_per_class(test_dir)
    print("Jumlah data train:", train_counts)
    print("Jumlah data test:", test_counts)

    train_gen, val_gen, test_gen = build_data_generators(train_dir, test_dir)
    class_indices = train_gen.class_indices
    class_names = sorted(class_indices, key=class_indices.get)
    num_classes = len(class_names)
    print(f"Jumlah kelas terdeteksi: {num_classes}")

    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump(class_indices, f)

    model, base_model = build_model(num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]

    # Tahap 1: melatih classifier head baru, base model Xception dibekukan
    print("Tahap 1: feature extraction")
    history_stage1 = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS_FEATURE_EXTRACTION, callbacks=callbacks,
        workers=DATA_LOADING_WORKERS, use_multiprocessing=USE_MULTIPROCESSING, max_queue_size=20,
    )
    plot_history(history_stage1, "training_stage1_feature_extraction.png", "Tahap 1")

    # Tahap 2: fine tuning, membuka beberapa layer terakhir Xception
    print(f"Tahap 2: fine tuning {FINE_TUNE_LAYERS} layer terakhir")
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

    history_stage2 = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS_FINE_TUNING, callbacks=callbacks,
        workers=DATA_LOADING_WORKERS, use_multiprocessing=USE_MULTIPROCESSING, max_queue_size=20,
    )
    plot_history(history_stage2, "training_stage2_fine_tuning.png", "Tahap 2 Fine Tuning")

    # Evaluasi akhir memakai data test yang belum pernah dilihat model
    print("Evaluasi pada data test")
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Akurasi test: {test_acc:.4f}, Loss test: {test_loss:.4f}")

    test_gen.reset()
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Menyimpan gambar confusion matrix untuk ditampilkan di halaman web
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.6), max(5, num_classes * 0.5)))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label Asli")
    ax.set_title("Confusion Matrix")
    max_val = cm.max()
    for i in range(num_classes):
        for j in range(num_classes):
            # Teks putih di kotak gelap, teks hitam di kotak terang, supaya angka selalu terbaca
            text_color = "white" if cm[i, j] > max_val * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
    plt.close(fig)

    model.save(MODEL_PATH)
    print(f"Model tersimpan di {MODEL_PATH}")

    metrics = {
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "img_size": list(IMG_SIZE),
        "class_names": class_names,
        "num_classes": num_classes,
        "dataset_counts": {"train": train_counts, "test": test_counts},
        "history_stage1": history_to_dict(history_stage1),
        "history_stage2": history_to_dict(history_stage2),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrik tersimpan di {METRICS_PATH}")


if __name__ == "__main__":
    main()