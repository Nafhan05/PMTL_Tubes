# 📡 Deteksi Wireless Jamming Berbasis Deep Learning

Sistem deteksi dan klasifikasi serangan *Wireless Jamming* pada *Physical Layer* (L1) menggunakan model **1D-CNN**, **LSTM**, dan **2D-CNN** dengan data sinyal I/Q dari DeepSig, dilengkapi **Bayesian Hyperparameter Optimization**.

---

## 🎯 Deskripsi Proyek

### Latar Belakang

Serangan *Wireless Jamming* merupakan ancaman serius pada sistem komunikasi nirkabel modern. Teknik deteksi konvensional berbasis *energy threshold* sering gagal membedakan antara interferensi sengaja dengan degradasi sinyal alami, terutama pada *Signal-to-Jamming Ratio* (SJR) rendah.

### Tujuan

Mengembangkan sistem *Artificial Intelligence* yang mampu mendeteksi dan mengklasifikasikan serangan *Wireless Jamming* secara langsung pada sinyal I/Q mentah menggunakan *Deep Learning*, serta membandingkan performa pendekatan domain waktu (1D-CNN, LSTM) vs domain frekuensi (2D-CNN).

### Kelas Target

| Label | Kelas               | Deskripsi                                                             |
| ----- | ------------------- | --------------------------------------------------------------------- |
| 0     | `Normal`          | Sinyal komunikasi bersih dengan noise alam (AWGN)                     |
| 1     | `CW_Jamming`      | Terinfeksi serangan *Continuous Wave* (Gelombang Sinusoidal Konstan) |
| 2     | `Barrage_Jamming` | Terinfeksi serangan *Gaussian Noise* berskala lebar                  |

---

## 📊 Hasil Evaluasi

### Ringkasan Performa Akhir — Semua Model

| Rank | Model              | Accuracy | Precision | Recall | F1-Score | vs Baseline      | Status           |
|:----:|--------------------|---------:|----------:|-------:|---------:|------------------|------------------|
| #1   | **LSTM**           | 97.74%   | 97.77%    | 97.76% | 0.977    | - (tanpa HPO)    | ✅ Terbaik       |
| #2   | **1D-CNN (HPO)**   | 96.31%   | 96.51%    | 96.35% | 0.963    | **+20.8%**       | ✅ Sukses HPO    |
| #3   | 2D-CNN (Baseline)  | 83.27%   | 88.74%    | 83.01% | 0.822    | - (sebelum HPO)  | ⚠️ Saturasi      |
| #4   | 2D-CNN (HPO)       | 83.36%   | 88.83%    | 83.10% | 0.823    | +0.1% (tetap)    | ❌ Bottleneck input |
| #5   | 1D-CNN (Baseline)  | 75.49%   | 85.59%    | 75.90% | 0.756    | - (sebelum HPO)  | ❌ Overfitting   |

### Computational Cost

| Model             | Parameters | Trainable | Epochs | Inference Latency | Accuracy |
|-------------------|-----------:|----------:|-------:|-------------------:|---------:|
| LSTM              | 120,835    | 120,835   | 21     | 61.5 ms            | 97.74%   |
| 1D-CNN (Baseline) | 150,851    | 149,955   | 6*     | 36.9 ms            | 75.49%   |
| 1D-CNN (HPO)      | 871,139    | 869,667   | 30     | 61.8 ms            | 96.31%   |
| 2D-CNN (Baseline) | 102,019    | 101,571   | 30     | 34.3 ms            | 83.27%   |
| 2D-CNN (HPO)      | 164,579    | 164,195   | 40     | 66.4 ms            | 83.36%   |

> *1D-CNN Baseline: early stopping pada epoch 6 karena overfitting parah (val_loss meningkat terus).

### Temuan Utama

- 🏆 **LSTM** mencapai akurasi tertinggi (97.7%) dengan parameter paling sedikit (121K) — paling efisien
- 🚀 **1D-CNN HPO** berhasil meningkatkan akurasi dari 75.5% → 96.3% (+20.8%) — HPO sangat efektif mengatasi overfitting
- ⚠️ **2D-CNN HPO** hanya naik 0.1% — bottleneck ada di representasi data (STFT spektrogram), bukan arsitektur model
- ⚡ Semua model memiliki inference latency <70ms — layak untuk real-time detection

---

## 🔧 Hyperparameter Optimization (HPO)

### Metodologi

Menggunakan **Bayesian Optimization** via Keras Tuner karena efisien dalam mencari 14 hyperparameter secara simultan.

**Alur Bayesian Optimization:**
1. **Inisialisasi Random** — Coba 5 konfigurasi acak sebagai titik awal
2. **Surrogate Model** — Bangun model prediksi (Gaussian Process) dari hasil trial sebelumnya
3. **Acquisition Function** — Pilih konfigurasi paling menjanjikan berdasarkan surrogate
4. **Evaluasi Model** — Training dengan HP terpilih, ukur `val_accuracy`
5. **Update & Ulangi** — Perbarui surrogate, ulangi hingga 30 trial

### Perbandingan Metode HPO

| Aspek         | Grid Search              | Random Search          | Bayesian Optimization ✓   |
|---------------|--------------------------|------------------------|----------------------------|
| Cara kerja    | Coba SEMUA kombinasi     | Coba kombinasi acak    | Belajar dari trial sebelum |
| Efisiensi     | Sangat lambat            | Sedang (untung2an)     | Cerdas & efisien (30 trial)|
| Kualitas      | Pasti optimal (tapi mahal)| Bisa miss optimal     | Mendekati optimal dg cepat |
| Cocok untuk   | Search space kecil (<4)  | Eksplorasi awal        | Search space besar (14 HP) |

---

## 🗂️ Dataset

Menggunakan dataset **DeepSig GOLD_XYZ_OSC.0001_1024.hdf5** (2,555,904 frame, dimensi `1024×2`).

> **Sumber:** T. J. O'Shea, T. Roy, T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," *IEEE JSTSP*, 2018.

Sinyal jamming di-injeksikan secara **sintetis** saat training menggunakan *Data Generator*:

- **CW Jamming:** Gelombang sinus pada frekuensi acak
- **Barrage Jamming:** Gaussian noise wideband
- **SJR Range:** -10 dB hingga +10 dB
- **Split:** Train 70% | Validation 15% | Test 15% (seed=42)

---

## 🧠 Arsitektur Model

### 1D-CNN (Baseline → HPO)

```
Input (1024, 2)
→ Conv1D(64, k=7) + BN + ReLU + SpatialDropout1D + MaxPool
→ Conv1D(128, k=5) + BN + ReLU + SpatialDropout1D + MaxPool
→ Conv1D(256, k=3) + BN + ReLU + SpatialDropout1D + MaxPool
→ Conv1D(256, k=3) + BN + ReLU + SpatialDropout1D + MaxPool
→ GlobalAveragePooling1D
→ Dense(128, ReLU) + Dropout(0.5)
→ Dense(3, Softmax)
```

### LSTM

```
Input (256, 2) — downsampled dari 1024
→ LSTM(128, return_sequences=True) + Dropout(0.4)
→ LSTM(64) + Dropout(0.4)
→ Dense(64, ReLU) + Dropout(0.3)
→ Dense(3, Softmax)
```

### 2D-CNN

```
Input: STFT Spectrogram dari sinyal I/Q
→ Conv2D(32, 3x3) + BN + ReLU + MaxPool
→ Conv2D(64, 3x3) + BN + ReLU + MaxPool
→ Conv2D(128, 3x3) + BN + ReLU + MaxPool
→ Flatten
→ Dense(64, ReLU) + Dropout(0.5)
→ Dense(3, Softmax)
```

---

## 🛠️ Struktur Direktori

```text
PMTL_Tubes/
├── data/
│   └── GOLD_XYZ_OSC.0001_1024.hdf5    # Dataset (download manual, ~6.4 GB)
├── notebooks/
│   ├── EDA_and_Visualization.ipynb     # Eksplorasi data
│   ├── visualize_hpo_curves.py         # Generate training curves HPO
│   ├── visualize_ppt_hpo.py            # Generate tabel & chart untuk PPT
│   └── visualize_ppt_updated.py        # Generate summary & computational cost
├── src/
│   ├── 1_data_generator.py             # Pipeline injeksi jamming sintetis
│   ├── 2_train_1dcnn.py                # Training model 1D-CNN
│   ├── 2_train_lstm.py                 # Training model LSTM
│   ├── 2_train_2dcnn.py                # Training model 2D-CNN
│   ├── 3_evaluate_models.py            # Evaluasi & benchmarking
│   ├── 4_demo_app.py                   # Demo interaktif (Streamlit)
│   ├── hpo_1dcnn.py                    # HPO Bayesian Optimization — 1D-CNN
│   ├── hpo_2dcnn.py                    # HPO Bayesian Optimization — 2D-CNN
│   ├── demo_components.py              # Komponen efek dunia nyata + animasi
│   ├── data_loader.py                  # Bridge module import
│   ├── gpu_setup.py                    # Setup GPU NVIDIA DLL
│   └── utils.py                        # Fungsi pembantu (visualisasi, metrik)
├── models/                             # Bobot model (.keras) — git-ignored
├── results/                            # Hasil evaluasi baseline
│   └── hpo/                            # Hasil evaluasi HPO + visualisasi PPT
├── requirements.txt
├── run_gpu.bat                         # Helper script untuk GPU support
├── .gitignore
└── README.md
```

---

## 💻 Persyaratan & Instalasi

### Spesifikasi Minimum

- **Python:** 3.10+
- **GPU:** NVIDIA dengan CUDA support (diuji pada GTX 1650)
- **RAM:** Minimal 16 GB
- **Storage:** ~10 GB (dataset + model)

### Instalasi

```bash
# 1. Clone repository
git clone https://github.com/Nafhan05/PMTL_Tubes.git
cd PMTL_Tubes

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktivasi (Windows PowerShell)
.venv\Scripts\activate

# 4. Install dependensi
pip install -r requirements.txt

# 5. Download dataset ke folder data/
#    Sumber: https://www.deepsig.ai/datasets
#    File: GOLD_XYZ_OSC.0001_1024.hdf5
#    Letakkan di: data/GOLD_XYZ_OSC.0001_1024.hdf5
```

### Setup GPU (NVIDIA)

Proyek ini menggunakan **TensorFlow 2.10** yang memerlukan CUDA runtime dari pip. Library CUDA sudah otomatis terinstall dari `requirements.txt` sebagai dependensi TensorFlow.

**Cara kerja GPU pada proyek ini:**

1. `run_gpu.bat` — menambahkan path DLL NVIDIA (CUDA, cuDNN, cuBLAS) ke `PATH` sebelum menjalankan script
2. `src/gpu_setup.py` — otomatis mencari dan memuat semua DLL NVIDIA dari `.venv/Lib/site-packages/nvidia/`

**Verifikasi GPU terdeteksi:**

```bash
.\run_gpu.bat python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

> ⚠️ **Troubleshooting:**
>
> - Pastikan **NVIDIA driver** terbaru sudah terinstall (download dari [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx))
> - Warning `ptxas.exe not found` saat training adalah **normal** — TensorFlow tetap berjalan menggunakan driver GPU
> - Diuji pada **GTX 1650 Mobile** dan **GCP VM** (untuk retrain HPO)

---

## 🚀 Cara Menjalankan

### 1. Eksplorasi Data

```bash
jupyter notebook notebooks/EDA_and_Visualization.ipynb
```

### 2. Training Model

```bash
# Training 1D-CNN
.\run_gpu.bat python src/2_train_1dcnn.py

# Training LSTM
.\run_gpu.bat python src/2_train_lstm.py

# Training 2D-CNN
.\run_gpu.bat python src/2_train_2dcnn.py

# Quick test (subset data)
.\run_gpu.bat python src/2_train_1dcnn.py --dry-run
```

> Model terbaik otomatis tersimpan di `models/best_*.keras`.

### 3. Hyperparameter Optimization

```bash
# HPO 1D-CNN (30 trials Bayesian Optimization)
.\run_gpu.bat python src/hpo_1dcnn.py --max-trials 30

# HPO 2D-CNN
.\run_gpu.bat python src/hpo_2dcnn.py --max-trials 30

# Retrain dengan best hyperparameters
.\run_gpu.bat python src/hpo_1dcnn.py --max-trials 0 --retrain --retrain-epochs 40
```

### 4. Evaluasi Model

```bash
.\run_gpu.bat python src/3_evaluate_models.py
```

Hasil disimpan di `results/` (baseline) dan `results/hpo/` (setelah HPO).

### 5. Generate Visualisasi PPT

```bash
# Training curves HPO
python notebooks/visualize_hpo_curves.py

# Tabel, chart, dan diagram untuk PPT
python notebooks/visualize_ppt_hpo.py

# Summary table & computational cost
python notebooks/visualize_ppt_updated.py
```

### 6. Demo Interaktif

```bash
.\run_gpu.bat streamlit run src/4_demo_app.py
```

Buka **http://localhost:8501** di browser.

---

## 📈 Visualisasi Hasil

Semua visualisasi tersimpan di `results/` dan `results/hpo/`:

### Baseline
- Training curves: `*_training_curves.png`
- Confusion matrix: `confusion_matrix_*.png`
- Classification report: `classification_report_*.png`

### HPO & PPT
- Training curves HPO: `hpo/*_hpo_training_curves.png`
- Confusion matrix HPO: `hpo/confusion_matrix_*_hpo.png`
- Perbandingan metode HPO: `hpo/ppt_hpo_method_comparison.png`
- Search space & best HP: `hpo/ppt_search_space_*.png`
- Alur Bayesian Optimization: `hpo/ppt_bayesian_flow.png`
- Bar chart perbandingan: `hpo/ppt_comparison_bar.png`
- Tabel ringkasan akhir: `hpo/ppt_final_summary.png`
- Computational cost: `hpo/ppt_computational_cost.png`, `hpo/ppt_computational_cost_bars.png`

---

## 📚 Referensi

1. T. J. O'Shea, T. Roy, T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," *IEEE JSTSP*, 2018.
2. K. Grover, A. Lim, Q. Yang, "Jamming and Anti-jamming Techniques in Wireless Networks: A Survey," *IJAHUC*, 2014.
3. M. Lichtman et al., "Antifragile Communications," *IEEE Systems Journal*, 2018.
4. DeepSig Inc., "RF Datasets for Machine Learning," https://www.deepsig.ai/datasets

---

## 👥 Kontributor

| Nama                       | NIM       | Peran       |
| -------------------------- | --------- | ----------- |
| Nafhan Hadiyan Shafwatudin | 18123029  | Developer   |
| *(Isi nama anggota 2)*   | *(NIM)* | *(Peran)* |
| *(Isi nama anggota 3)*   | *(NIM)* | *(Peran)* |

---

> **Catatan:** Dataset `GOLD_XYZ_OSC.0001_1024.hdf5` (~6.4 GB) dan model weights (`.keras`) tidak di-commit ke repository karena ukurannya. Silakan download dataset manual dari sumber DeepSig, lalu jalankan training sendiri.
