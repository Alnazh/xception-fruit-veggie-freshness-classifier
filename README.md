<div align="center">

# FreshScan

### Klasifikasi Kesegaran Buah dan Sayur Berbasis Transfer Learning Xception

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?logo=bootstrap&logoColor=white)
![ChartJS](https://img.shields.io/badge/Chart.js-visualisasi-F5788D?logo=chartdotjs&logoColor=white)

Unggah foto buah atau sayur, sistem akan mengenali jenisnya sekaligus menilai kondisinya, segar atau busuk, ditenagai Xception hasil transfer learning.

[Fitur](#fitur-utama) &middot; [Tangkapan Layar](#tangkapan-layar) &middot; [Instalasi](#instalasi--menjalankan-lokal) &middot; [Melatih Model](#melatih-model-dengan-dataset-sendiri) &middot; [Struktur Proyek](#struktur-proyek)

</div>

<br>

<div align="center">
  <img src="docs/screenshots/dashboard.png" alt="Dashboard FreshScan" width="850">
</div>

<br>

## Tentang Proyek

Penyortiran buah dan sayur berdasarkan kesegaran di rantai pasok pangan masih sering dilakukan manual, lambat, dan rawan penilaian berbeda antar orang. FreshScan dibangun untuk membantu mengotomasi proses tersebut menggunakan pendekatan **transfer learning** dengan arsitektur **Xception** yang sudah dilatih pada jutaan gambar ImageNet, kemudian dilatih ulang secara khusus dalam dua tahap, feature extraction dan fine tuning, untuk mengenali kombinasi jenis item dan kondisi kesegarannya sekaligus dalam satu model.

Jumlah dan nama kelas yang dikenali **otomatis mengikuti struktur folder dataset** yang dipakai untuk training, tidak pernah di-hardcode di dalam kode. Aplikasi ini juga transparan soal performa modelnya sendiri, lengkap dengan grafik training, confusion matrix, dan statistik dataset yang bisa dilihat langsung di halaman web.

## Fitur Utama

- **Klasifikasi real time**, unggah foto lewat drag and drop, hasil keluar dalam hitungan detik
- **Klasifikasi multi kelas**, mengenali jenis item dan kondisi kesegarannya sekaligus, bukan sekadar segar atau busuk generik
- **Top 5 probabilitas tertinggi**, ditampilkan dengan grafik batang beserta selisih keyakinan antar kelas
- **Dashboard informatif**, statistik dataset, alur cara kerja sistem, dan daftar kelas yang dikenali
- **Evaluasi model transparan**, grafik akurasi dan loss dua tahap training, confusion matrix, serta arsitektur model, semuanya bisa dilihat publik
- **Halaman dataset**, sumber data, rincian jumlah gambar, dan grafik komposisi per kelas
- **Mode aman tanpa model**, aplikasi tetap bisa dibuka walau model belum dilatih, halaman terkait menampilkan pesan informatif alih-alih error

## Tangkapan Layar

<table>
<tr>
<td width="50%">

**Dashboard**
<img src="docs/screenshots/dashboard.png" alt="Dashboard">

</td>
<td width="50%">

**Klasifikasi**
<img src="docs/screenshots/klasifikasi.png" alt="Halaman Klasifikasi">

</td>
</tr>
<tr>
<td width="50%">

**Hasil Klasifikasi**
<img src="docs/screenshots/hasil.png" alt="Hasil Klasifikasi">

</td>
<td width="50%">

**Model dan Akurasi**
<img src="docs/screenshots/model.png" alt="Halaman Model dan Akurasi">

</td>
</tr>
<tr>
<td width="50%">

**Dataset**
<img src="docs/screenshots/dataset.png" alt="Halaman Dataset">

</td>
<td width="50%">

**Tentang**
<img src="docs/screenshots/tentang.png" alt="Halaman Tentang">

</td>
</tr>
</table>

## Kelas yang Dikenali

Dataset menyediakan label per jenis item, bukan cuma fresh atau rotten generik, sehingga model dilatih untuk mengenali kelas gabungan jenis dan kondisi secara langsung, contoh `FreshApple`, `FreshBanana`, `RottenApple`, `RottenBanana`, dan seterusnya. Jumlah kelas mengikuti berapa banyak sub folder yang tersedia di dataset yang dipakai, bisa 6 kelas untuk dataset kecil atau lebih dari 20 kelas untuk dataset yang lebih lengkap.

Label jenis item dan kondisi diturunkan otomatis dari nama folder kelas melalui fungsi `parse_class_name` di `app.py`, mendukung penulisan besar kecil huruf yang bervariasi.

## Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| Backend | Flask 3.0 |
| Machine Learning | TensorFlow / Keras, Xception (transfer learning) |
| Pemrosesan Gambar | Pillow, NumPy |
| Frontend | Bootstrap 5, Bootstrap Icons, Chart.js |
| Evaluasi Model | scikit-learn, Matplotlib |
| Server Produksi | Gunicorn |

## Struktur Proyek

```
freshscan-app/
├── app.py                        # Aplikasi Flask (routing, load model, klasifikasi)
├── train_model.py                # Script training transfer learning Xception
├── generate_confusion_matrix.py  # Script pembuat confusion matrix dari model terlatih
├── requirements.txt
├── dataset_tools/
│   └── prepare_dataset.py        # Alat bantu penyiapan dataset
├── model/                        # Output training, model terlatih, mapping kelas, metrik
├── static/
│   ├── css/style.css             # Styling kustom
│   ├── js/script.js              # Interaksi drag and drop dan preview gambar
│   ├── js/charts.js              # Rendering seluruh grafik Chart.js
│   ├── reports/                  # Grafik akurasi/loss dan confusion matrix (PNG)
│   └── uploads/                  # Gambar yang diunggah pengguna
├── templates/                    # dashboard, klasifikasi, hasil, model, dataset, tentang
└── docs/screenshots/             # Tangkapan layar untuk README ini
```

## Instalasi & Menjalankan Lokal

```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt

# Jalankan aplikasi
python app.py
```

Buka **http://127.0.0.1:5000** di browser. Jika model belum dilatih, aplikasi tetap bisa dibuka, halaman Model dan Dataset akan menampilkan pesan "belum ada data" alih-alih grafik.

## Melatih Model dengan Dataset Sendiri

1. Unduh dataset dari Kaggle: [Fruits and Vegetables Dataset](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset).
2. Ekstrak ke folder `dataset/` di root proyek. Dua struktur folder didukung otomatis:

   **Sudah terpisah train/test:**
   ```
   dataset/
   ├── train/
   │   ├── FreshApple/
   │   ├── RottenApple/
   │   └── ...
   └── test/
       └── (struktur yang sama)
   ```

   **Flat, semua kelas langsung di bawah `dataset/`:**
   ```
   dataset/
   ├── FreshApple/
   ├── RottenApple/
   └── ...
   ```
   Untuk struktur flat, `train_model.py` otomatis membuat split 90:10 ke folder `dataset_split/` saat pertama kali dijalankan, dataset asli tidak diubah.

3. Jalankan training:
   ```bash
   python train_model.py
   ```

Sebelum melatih ulang, kosongkan dulu folder `model/` dari file `.h5` atau `.keras` lama, supaya aplikasi tidak keliru memuat model hasil training sebelumnya.

Untuk training lebih ringan di laptop tanpa GPU, atur `FAST_TRAINING = True` di bagian atas `train_model.py`. Mode ini memakai resolusi gambar 224x224, jumlah epoch lebih sedikit, dan fine tuning pada lapisan yang lebih sedikit, dengan trade off akurasi yang sedikit lebih rendah dibanding mode penuh (299x299).

Setelah training selesai, folder `model/` akan berisi model terlatih, mapping label, dan seluruh metrik yang dipakai halaman Model dan Dataset untuk menggambar grafik. Grafik PNG untuk lampiran laporan juga otomatis tersimpan di `static/reports/`.

Untuk training dengan GPU gratis, tersedia notebook `FreshScan_Training_Colab.ipynb` yang tinggal diunggah ke [Google Colab](https://colab.research.google.com).

## Arsitektur Model

Xception (pretrained ImageNet, dibekukan) &rarr; Global Average Pooling &rarr; Dense 256 (ReLU) &rarr; Dropout 0.5 &rarr; Dense 64 (ReLU) &rarr; Dense sejumlah kelas (Softmax).

Strategi training dilakukan dalam dua tahap, feature extraction dengan base model dibekukan, dilanjutkan fine tuning dengan sebagian lapisan terakhir dibuka. Evaluasi akhir memakai akurasi, precision, recall, F1-score, dan confusion matrix pada data uji yang terpisah dari data training.

## Deployment

Aplikasi ini kompatibel dengan platform hosting yang mendukung aplikasi Flask berbasis Python, seperti Render atau Railway, dengan start command:

```bash
gunicorn app:app
```

Pastikan folder `model/` beserta file `.keras` dan `.json` di dalamnya ikut ter-deploy, gunakan Git LFS bila ukurannya besar.

## Catatan Kompatibilitas Versi

Jika muncul error seperti `Unrecognized keyword arguments` atau `quantization_config` saat model dimuat, artinya versi TensorFlow/Keras yang dipakai saat training berbeda dengan versi yang dipakai saat menjalankan aplikasi. Pastikan `requirements.txt` (`tensorflow==2.20.0`) dipakai konsisten di kedua tempat.
