# FreshScan, Klasifikasi Kesegaran Buah & Sayur dengan Transfer Learning Xception

Aplikasi web berbasis **Flask** yang menerapkan **Transfer Learning** dengan arsitektur
**Xception** untuk mengklasifikasikan gambar buah & sayur ke dalam **beberapa kelas
sekaligus**: jenis item (apel, pisang, wortel, dst, mengikuti dataset yang dipakai)
**dan** kondisinya (segar/busuk). Jumlah & nama kelas **otomatis mengikuti struktur
folder dataset** yang kamu pakai untuk training, tidak di-hardcode di kode.

Dibuat untuk **Tugas 12** mata kuliah Artificial Intelligence:
*"Penerapan Transfer Learning Xception untuk Klasifikasi Kesegaran Buah (Fresh vs
Rotten) Berbasis Web Menggunakan Flask"*.

---

## 1. Halaman Aplikasi

| Route       | Isi                                                                 |
|-------------|----------------------------------------------------------------------|
| `/`         | Dashboard: statistik ringkas, cara kerja, kenapa deteksi ini penting  |
| `/predict`  | Form unggah gambar + hasil klasifikasi lengkap dengan grafik probabilitas |
| `/model`    | Arsitektur model, diagram, grafik training, confusion matrix, classification report |
| `/dataset`  | Informasi & komposisi dataset (grafik + tabel jumlah gambar per kelas) |
| `/about`    | Latar belakang masalah, metodologi, batasan, potensi pengembangan     |

---

## 2. Struktur Proyek

```
fruit-freshness-app/
├── app.py                       # Backend Flask (routing, load model, klasifikasi)
├── train_model.py               # Script training transfer learning Xception (multi-kelas)
├── requirements.txt
├── dataset_tools/
│   └── prepare_dataset.py       # OPSIONAL/legacy (lihat catatan di dalam file)
├── model/                       # (dibuat otomatis) model .keras + label mapping + metrik
├── static/reports/              # (dibuat otomatis) grafik akurasi/loss dan confusion matrix (PNG)
├── templates/                   # HTML (Jinja2 + Bootstrap 5)
│   ├── base.html
│   ├── index.html               # Dashboard
│   ├── predict.html             # Upload & hasil
│   ├── model_info.html          # Detail model & evaluasi
│   ├── dataset.html             # Info dataset
│   └── about.html               # Metodologi
└── static/
    ├── css/style.css            # Styling kustom
    ├── js/script.js             # Interaksi drag & drop, preview gambar
    ├── js/charts.js             # Rendering semua grafik Chart.js
    └── uploads/                 # Gambar yang diunggah pengguna
```

Prinsip **"1 file 1 bahasa pemrograman"** diterapkan ketat: seluruh logika Python
(Flask + model AI) ada di file `.py`, struktur halaman di file `.html`, styling di
`.css`, dan seluruh interaksi/visualisasi sisi klien di file `.js`, tidak dicampur
dalam satu file.

---

## 3. Kenapa Multi-Kelas (Bukan Cuma Fresh/Rotten Biner)?

Dataset yang dipakai sudah menyediakan label per jenis item (bukan cuma fresh/rotten
generik), jadi model dilatih untuk mengenali kelas gabungan jenis x kondisi langsung,
contoh:

`FreshApple`, `FreshBanana`, `FreshCarrot`, `RottenApple`, `RottenBanana`, `RottenCarrot`, dst.

Jumlah kelas otomatis mengikuti berapa banyak sub-folder yang ada di dataset kamu,
bisa 6 kelas (dataset 3 buah) atau 20 kelas (dataset 10 buah & sayur), tanpa perlu ubah
kode. Ini membuat aplikasi bisa menampilkan info lebih kaya, misalnya *"Terdeteksi:
Wortel, kondisi Busuk (96%)"*, bukan sekadar *"Rotten (96%)"*. Label fresh/rotten +
jenis item diturunkan otomatis dari nama kelas di `app.py` (lihat fungsi
`parse_class_name`), dan sudah case-insensitive (mendukung `FreshApple` maupun
`freshapples`).

> **Catatan istilah:** tugas ini bertipe **klasifikasi** (mengelompokkan gambar ke
> kategori tertentu). Namun tindakan model saat memproses satu gambar tetap disebut
> **"memprediksi"** (`model.predict()`), ini istilah standar di ML/TensorFlow,
> bukan kesalahan penulisan. Karena itu di kode kamu akan menemukan fungsi seperti
> `predict_freshness()`, sementara di teks pengguna kami konsisten memakai
> "klasifikasi" untuk merujuk ke tugas/metodenya.

---

## 4. Instalasi

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 5. Menyiapkan Dataset & Melatih Model

1. Download dataset dari Kaggle:
   [muhriddinmuxiddinov/fruits-and-vegetables-dataset](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)
2. Extract ke folder `dataset/` di root proyek. **Dua jenis struktur folder didukung
   secara otomatis:**

   **A) Sudah terpisah train/test:**
   ```
   dataset/
   ├── train/
   │   ├── FreshApple/
   │   ├── RottenApple/
   │   └── ... (kelas lain)
   └── test/
       └── (folder yang sama)
   ```

   **B) Flat, semua kelas langsung di bawah `dataset/` (tanpa folder train/test):**
   ```
   dataset/
   ├── FreshApple/
   ├── RottenApple/
   ├── FreshCarrot/
   ├── RottenCarrot/
   └── ... (kelas lain)
   ```
   Untuk struktur ini, `train_model.py` **otomatis membuat split 90:10** (train:test)
   ke folder `dataset_split/` saat pertama kali dijalankan (dataset asli tidak diubah).

   Tidak perlu jalankan script konversi manual apapun, `train_model.py` mendeteksi
   struktur mana yang kamu punya secara otomatis.
3. Jalankan training:
   ```bash
   python train_model.py
   ```

> ⚠️ **PENTING sebelum training ulang:** hapus dulu isi folder `model/` (file
> `.h5` maupun `.keras` yang lama). Kalau ada 2 file model berbeda tertinggal di
> sana (misal sisa training lama + training baru), `app.py` bisa salah memuat
> file yang **bukan hasil training terbarumu**, inilah penyebab paling umum
> error `quantization_config` / hasil klasifikasi yang aneh meskipun training
> barusan sukses dengan akurasi bagus.

### Training Lebih Cepat di Laptop CPU (tanpa GPU)

Kalau training 8 jam+ terasa terlalu lama, buka `train_model.py` dan cek variabel
`FAST_TRAINING` di bagian atas file:

```python
FAST_TRAINING = True   # default: mode cepat aktif
```

Saat `True`, script otomatis memakai:
- Resolusi gambar 224&times;224 (bukan 299&times;299) &rarr; komputasi ~44% lebih ringan
- Epoch lebih sedikit (8 + 5, dibanding 12 + 8). `EarlyStopping` tetap aktif jadi
  training otomatis berhenti lebih awal kalau sudah konvergen
- Fine-tuning hanya 15 layer terakhir (bukan 30) &rarr; backpropagation tahap 2 lebih ringan
- Data loading pakai beberapa thread (`DATA_LOADING_WORKERS = 4`) supaya CPU tidak nganggur
  menunggu gambar berikutnya selesai dibaca/diolah

**Trade-off:** akurasi biasanya turun sedikit (1-3%) dibanding mode penuh, tapi waktu
training bisa berkurang signifikan. Set `FAST_TRAINING = False` kalau mau kualitas
maksimal dan tidak keberatan menunggu lebih lama.

> Resolusi gambar (`IMG_SIZE`) otomatis disimpan ke `training_metrics.json` dan dibaca
> ulang oleh `app.py` saat inferensi, jadi kamu tidak perlu mengubah apapun di `app.py`
>, apapun resolusi yang dipakai saat training akan otomatis dipakai juga saat prediksi.

Setelah selesai, akan terbentuk di folder `model/`:
- `xception_fruit_freshness.keras`, model terlatih
- `class_indices.json`, mapping label ke index
- `training_metrics.json`, **semua metrik** (riwayat akurasi/loss, confusion matrix,
  classification report, komposisi dataset) yang otomatis dipakai halaman `/model`
  dan `/dataset` untuk menggambar grafik

Serta di folder `static/reports/`: grafik PNG (akurasi/loss tiap tahap + confusion matrix)
untuk dilampirkan ke laporan tugas.

**Ingin training lebih cepat dengan GPU gratis?** Gunakan notebook Google Colab yang
sudah disediakan terpisah (`FreshScan_Training_Colab.ipynb`), tinggal upload ke
[colab.research.google.com](https://colab.research.google.com), jalankan semua cell,
lalu download 3 file (`xception_fruit_freshness.keras`, `class_indices.json`,
`training_metrics.json`) ke folder `model/` lokal.

---

## 6. Menjalankan Aplikasi Web

```bash
python app.py
```

Buka `http://127.0.0.1:5000`. Jika model belum dilatih, aplikasi tetap bisa dibuka
tapi halaman `/model` dan `/dataset` akan menampilkan pesan "belum ada data" alih-alih
grafik.

---

## 7. Fitur Aplikasi

- ✅ Dashboard dengan statistik ringkas (total gambar, jumlah kelas, akurasi model)
- ✅ Upload gambar dengan **drag & drop** + preview
- ✅ Klasifikasi multi-kelas: jenis buah + kondisi kesegaran + tingkat keyakinan
- ✅ Grafik bar probabilitas untuk semua 6 kelas di setiap hasil klasifikasi
- ✅ Halaman **Model & Akurasi**: diagram arsitektur, grafik training 2 tahap,
  confusion matrix (heatmap tabel), classification report
- ✅ Halaman **Dataset**: grafik komposisi data per kelas, tabel rincian jumlah gambar
- ✅ Halaman **Tentang**: latar belakang masalah, metodologi, batasan, pengembangan
- ✅ UI Bootstrap 5 dengan tema visual khas (hijau segar vs rust busuk)

---

## 8. Deployment ke Hosting Gratis

- **Render.com** (free web service tier)
- **Railway.app**
- **Hugging Face Spaces** (Docker template)

Langkah umum:
1. Push project (kecuali `dataset/`, `data/`, lihat `.gitignore`) ke GitHub
2. Hubungkan repository ke platform hosting
3. Start command: `gunicorn app:app`
4. Pastikan folder `model/` (termasuk file `.keras` & `.json`) ikut ter-deploy
   (pakai Git LFS jika ukurannya besar)

---

## 9. Troubleshooting Versi TensorFlow/Keras

Jika muncul error seperti `Unrecognized keyword arguments` atau
`quantization_config` saat `load_model()`:
- Ini artinya versi TensorFlow/Keras yang dipakai untuk **training** (misal di
  Colab) berbeda dengan versi yang dipakai untuk **menjalankan aplikasi** (lokal).
- Solusi: pastikan `requirements.txt` (`tensorflow==2.20.0`) dipakai konsisten di
  kedua tempat. Jika training ulang di Colab, tambahkan cell
  `!pip install -q tensorflow==2.20.0` di awal notebook sebelum training, lalu
  **restart runtime** sebelum lanjut.

---

## 10. Catatan untuk Laporan/Dokumentasi Tugas

- **Studi kasus**: Klasifikasi kesegaran buah multi-kelas (jenis buah x kondisi)
  menggunakan Transfer Learning Xception.
- **Dataset**: Fruits and Vegetables Dataset (Kaggle, oleh muhriddinmuxiddinov).
- **Arsitektur**: Xception (pretrained ImageNet) + GAP + Dense(256) + Dropout(0.5)
  + Dense(64) + Dense(6, softmax).
- **Strategi training**: Feature Extraction lalu Fine-Tuning 30 layer terakhir.
- **Evaluasi**: Akurasi, precision, recall, F1-score, dan confusion matrix pada
  data test terpisah, semuanya bisa dilihat langsung di halaman `/model` setelah
  training, atau di folder `static/reports/` sebagai gambar untuk lampiran laporan.
