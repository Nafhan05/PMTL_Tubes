# 📡 Deteksi Wireless Jamming Berbasis Deep Learning

Sistem deteksi dan klasifikasi serangan *Wireless Jamming* pada *Physical Layer* (L1) menggunakan model **1D-CNN** dan **LSTM** dengan data sinyal I/Q dari DeepSig.

---

## 🎯 Deskripsi Proyek

### Latar Belakang

Serangan *Wireless Jamming* merupakan ancaman serius pada sistem komunikasi nirkabel modern. Teknik deteksi konvensional berbasis *energy threshold* sering gagal membedakan antara interferensi sengaja dengan degradasi sinyal alami, terutama pada *Signal-to-Jamming Ratio* (SJR) rendah.

### Tujuan

Mengembangkan sistem *Artificial Intelligence* yang mampu mendeteksi dan mengklasifikasikan serangan *Wireless Jamming* secara langsung pada sinyal I/Q mentah menggunakan *Deep Learning*.

### Kelas Target

| Label | Kelas | Deskripsi |
|-------|-------|-----------|
| 0 | `Normal` | Sinyal komunikasi bersih dengan noise alam (AWGN) |
| 1 | `CW_Jamming` | Terinfeksi serangan *Continuous Wave* (Gelombang Sinusoidal Konstan) |
| 2 | `Barrage_Jamming` | Terinfeksi serangan *Gaussian Noise* berskala lebar |

---

## 📊 Hasil Evaluasi

### Performa Model pada Test Set (383.387 sampel)

| Metrik | 1D-CNN | LSTM |
|--------|--------|------|
| **Accuracy** | 97.0% | **98.2%** |
| **Precision (macro)** | 97.2% | **98.2%** |
| **Recall (macro)** | 97.1% | **98.2%** |
| **F1-Score (macro)** | 97.0% | **98.2%** |
| **Avg Latency** | **36 ms** | 47 ms |

### Performa Per-Kelas

| Kelas | 1D-CNN (F1) | LSTM (F1) |
|-------|-------------|-----------|
| Normal | 95.9% | **97.5%** |
| CW Jamming | **98.9%** | 99.3% |
| Barrage Jamming | 96.3% | **97.7%** |

### Temuan Utama dari Demo Interaktif

- ✅ **SJR ≤ 0 dB:** Kedua model mendeteksi jamming dengan sangat baik (>97%)
- ⚠️ **SJR +10 dB:** Batas deteksi — CW masih kadang terdeteksi, Barrage sering gagal
- ❌ **Doppler tinggi (200 km/h):** False positive CW Jamming karena pergeseran frekuensi mirip CW
- 🏆 **LSTM lebih sensitif** terhadap jamming lemah, **1D-CNN lebih cepat dan stabil**

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

### 1D-CNN

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

---

## 🛠️ Struktur Direktori

```text
PMTL_Tubes/
├── data/
│   └── GOLD_XYZ_OSC.0001_1024.hdf5    # Dataset (download manual, ~6.4 GB)
├── notebooks/
│   └── EDA_and_Visualization.ipynb     # Eksplorasi data
├── src/
│   ├── 1_data_generator.py             # Pipeline injeksi jamming sintetis
│   ├── 2_train_1dcnn.py                # Training model 1D-CNN
│   ├── 2_train_lstm.py                 # Training model LSTM
│   ├── 3_evaluate_models.py            # Evaluasi & benchmarking
│   ├── 4_demo_app.py                   # 🆕 Demo interaktif (Streamlit)
│   ├── demo_components.py              # 🆕 Komponen efek dunia nyata + animasi
│   ├── data_loader.py                  # Bridge module import
│   ├── gpu_setup.py                    # Setup GPU NVIDIA DLL
│   └── utils.py                        # Fungsi pembantu (visualisasi, metrik)
├── models/                             # Bobot model (.keras) — git-ignored
├── results/                            # Confusion matrix, evaluation JSON
├── webui_test/                         # Screenshot hasil pengujian demo
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

> ⚠️ **TensorFlow GPU:** Proyek ini menggunakan TensorFlow 2.10 yang memerlukan konfigurasi CUDA manual. File `run_gpu.bat` sudah menangani path NVIDIA DLL secara otomatis.

---

## 🚀 Cara Menjalankan

### 1. Eksplorasi Data

```bash
jupyter notebook notebooks/EDA_and_Visualization.ipynb
```

### 2. Training Model

```bash
# Training 1D-CNN (full)
.\run_gpu.bat python src/2_train_1dcnn.py

# Training 1D-CNN (quick test)
.\run_gpu.bat python src/2_train_1dcnn.py --dry-run

# Training LSTM (full)
.\run_gpu.bat python src/2_train_lstm.py

# Custom hyperparameters
.\run_gpu.bat python src/2_train_1dcnn.py --epochs 50 --lr 0.0005
.\run_gpu.bat python src/2_train_lstm.py --seq-len 512
```

> Model terbaik otomatis tersimpan di `models/best_1dcnn.keras` dan `models/best_lstm.keras`.

### 3. Evaluasi Model

```bash
.\run_gpu.bat python src/3_evaluate_models.py
```

Hasil disimpan di `results/evaluation_results.json` dan `results/confusion_matrix_*.png`.

### 4. Demo Interaktif 🆕

```bash
.\run_gpu.bat streamlit run src/4_demo_app.py
```

Buka **http://localhost:8501** di browser.

#### Fitur Demo

| Fitur | Deskripsi |
|-------|-----------|
| **Model Selector** | Pilih 1D-CNN atau LSTM |
| **Jenis Jamming** | None / CW / Barrage dengan slider SJR |
| **Efek Dunia Nyata** | Doppler Shift & Multipath Fading |
| **Network Diagram** | Animasi TX → RX → AI Model dengan gelombang bergerak |
| **Visualisasi Sinyal** | Time Domain, Konstelasi, Power Spectrum (Plotly) |
| **Hasil Deteksi** | Prediksi + Confidence + Latency real-time |

---

## 📚 Referensi

1. T. J. O'Shea, T. Roy, T. C. Clancy, "Over-the-Air Deep Learning Based Radio Signal Classification," *IEEE JSTSP*, 2018.
2. K. Grover, A. Lim, Q. Yang, "Jamming and Anti-jamming Techniques in Wireless Networks: A Survey," *IJAHUC*, 2014.
3. M. Lichtman et al., "Antifragile Communications," *IEEE Systems Journal*, 2018.
4. DeepSig Inc., "RF Datasets for Machine Learning," https://www.deepsig.ai/datasets

---

## 👥 Kontributor

| Nama | NIM | Peran |
|------|-----|-------|
| *(Isi nama anggota 1)* | *(NIM)* | *(Peran)* |
| *(Isi nama anggota 2)* | *(NIM)* | *(Peran)* |

---

> **Catatan:** Dataset `GOLD_XYZ_OSC.0001_1024.hdf5` (~6.4 GB) dan model weights (`.keras`) tidak di-commit ke repository karena ukurannya. Silakan download dataset manual dari sumber DeepSig, lalu jalankan training sendiri.
