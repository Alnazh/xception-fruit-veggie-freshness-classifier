"""
Script opsional untuk mengonversi dataset multi kelas menjadi struktur biner fresh/rotten.
Tidak wajib dipakai karena train_model.py sudah bisa langsung memakai dataset multi kelas.
Jalankan: python dataset_tools/prepare_dataset.py --source dataset --target data
"""

import os
import shutil
import argparse
from pathlib import Path

FRESH_KEYWORD = "fresh"
ROTTEN_KEYWORD = "rotten"


def classify_folder_name(folder_name):
    # Menentukan apakah nama folder kelas termasuk fresh atau rotten
    name_lower = folder_name.lower()
    if ROTTEN_KEYWORD in name_lower:
        return "rotten"
    if FRESH_KEYWORD in name_lower:
        return "fresh"
    return None


def copy_split(source_split_dir, target_split_dir):
    # Menyalin gambar dari satu split ke struktur biner fresh/rotten
    (target_split_dir / "fresh").mkdir(parents=True, exist_ok=True)
    (target_split_dir / "rotten").mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for class_folder in sorted(source_split_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        label = classify_folder_name(class_folder.name)
        if label is None:
            print(f"Folder tidak dikenali, dilewati: {class_folder.name}")
            continue

        destination = target_split_dir / label
        count = 0
        for img_file in class_folder.glob("*"):
            if img_file.is_file():
                new_name = f"{class_folder.name}_{img_file.name}"
                shutil.copy2(img_file, destination / new_name)
                count += 1
                total_copied += 1

        print(f"{class_folder.name}: {count} gambar disalin ke {label}")

    print(f"Total gambar disalin ke {target_split_dir}: {total_copied}")


def main():
    parser = argparse.ArgumentParser(description="Konversi dataset multi kelas ke struktur biner")
    parser.add_argument("--source", type=str, default="dataset")
    parser.add_argument("--target", type=str, default="data")
    args = parser.parse_args()

    source_root = Path(args.source)
    target_root = Path(args.target)

    if not source_root.exists():
        raise FileNotFoundError(f"Folder sumber {source_root} tidak ditemukan.")

    for split_name in ["train", "test"]:
        source_split = source_root / split_name
        if not source_split.exists():
            print(f"Split {split_name} tidak ditemukan, dilewati.")
            continue

        print(f"Memproses split: {split_name}")
        copy_split(source_split, target_root / split_name)

    print("Selesai, struktur dataset biner ada di folder:", target_root.resolve())


if __name__ == "__main__":
    main()
