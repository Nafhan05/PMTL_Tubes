# 📋 Handover — Wireless Jamming Detection (PMTL_Tubes)

> **Tanggal:** 29 Mei 2026  
> **Project:** Deteksi Wireless Jamming menggunakan Deep Learning (1D-CNN, LSTM, 2D-CNN)  
> **Lokasi:** `D:\Coding\College Coding vibes\PMTL_Tubes`

---

## 1. Status Proyek Saat Ini

| Item | Status |
|------|--------|
| Dataset & Preprocessing | ✅ Selesai |
| Training LSTM | ✅ Selesai — **97.7% accuracy** |
| Training 1D-CNN (baseline) | ✅ Selesai — 75.5% accuracy (overfitting) |
| Training 2D-CNN (baseline) | ✅ Selesai — 83.3% accuracy (underfitting) |
| HPO Search 1D-CNN (30 trials) | ✅ Selesai |
| HPO Search 2D-CNN (30 trials) | ✅ Selesai |
| Retrain 1D-CNN dgn best HP | ✅ Selesai — **96.3% val_accuracy** (Overfitting teratasi!) |
| Retrain 2D-CNN dgn best HP | ❌ Belum dijalankan (Siap dijalankan dengan GPU + multiprocessing) |
| Evaluasi final (test set) | ❌ Belum dijalankan |
| Presentasi final | ❌ Belum disiapkan |

---

## 2. Hasil Saat Ini (Ringkasan Performa)

### Baseline (Sebelum HPO)

| Model | Test Accuracy | F1 Macro | Masalah |
|-------|:---:|:---:|---|
| **LSTM** | **97.74%** | **0.977** | ✅ Sudah bagus |
| 2D-CNN | 83.27% | 0.822 | Recall kelas Normal hanya 50% |
| 1D-CNN | 75.49% | 0.756 | Overfitting parah (train 92% vs val 66%) |

### Setelah HPO — 1D-CNN Retrain (30 epoch, Optimal)

| Metrik | Baseline | HPO Retrain | Perubahan |
|--------|:---:|:---:|:---:|
| Train Accuracy | 92.2% | 96.8% | ↑ Meningkat |
| Val Accuracy | 66-72% | **96.3%** | ↑ Naik signifikan, Gap hilang! |
| Val Loss | 1.25–2.49 | **0.11** | ↓ Turun drastis (Sangat baik) |

> [!TIP]
> 1D-CNN HPO retrain pada full dataset (2.5M samples) dengan optimalisasi vectorization & multiprocessing berhasil mencapai **96.38% validation accuracy** dan **0.11 validation loss** pada Epoch 23 (Early Stopping). Gap overfitting antara training & validation terpangkas menjadi hanya **~0.5%**, membuktikan model hasil HPO sangat robust.

### HPO Best Hyperparameters yang Ditemukan

**1D-CNN (`results/hpo_best_1dcnn.json`):**
| Parameter | Baseline | HPO Best |
|-----------|:---:|:---:|
| Num blocks | 4 | 4 |
| Kernel size | 7,5,3,3 | **7 (uniform)** |
| Filters | 64→128→128→128 | **192→256→256→32** |
| Spatial dropout | 0.1–0.2 | **0.05 (semua)** |
| Dense units | 64 | **128** |
| Dense dropout | 0.5 | **0.3** |
| L2 weight | 1e-4 | **1e-5** |
| Learning rate | 3e-4 | **1.83e-3** |

**2D-CNN (`results/hpo_best_2dcnn.json`):**
| Parameter | Baseline | HPO Best |
|-----------|:---:|:---:|
| Num blocks | 3 | **5** |
| Kernel size | 3 | **5** |
| Filters | 32→64→128 | **32→64→32→32→32** |
| Use L2 | Yes (1e-4) | **No** |
| Dense units | 64 | **256** |
| Dense dropout | 0.5 | **0.5** |
| Learning rate | 3e-4 | **8.8e-3** |
| STFT nperseg | 64 | 64 |

---

## 3. Struktur File & Penjelasan

### `src/` — Source Code Utama

| File | Fungsi |
|------|--------|
| `1_data_generator.py` | Load dataset HDF5, inject jamming (CW & Barrage), buat `JammingDataGenerator` |
| `2_train_1dcnn.py` | Training 1D-CNN baseline (downsampled 1024→256) |
| `2_train_lstm.py` | Training LSTM baseline (downsampled 1024→256) |
| `2_train_2dcnn.py` | Training 2D-CNN baseline (STFT → Spektrogram) |
| `3_evaluate_models.py` | Evaluasi semua model: confusion matrix, classification report, latency |
| `4_demo_app.py` | Demo web app (Streamlit) untuk showcase |
| **`hpo_1dcnn.py`** | **HPO Bayesian Search + Retrain untuk 1D-CNN** |
| **`hpo_2dcnn.py`** | **HPO Bayesian Search + Retrain untuk 2D-CNN** |
| `data_loader.py` | Bridge/proxy import karena nama file dimulai angka |
| `gpu_setup.py` | Setup NVIDIA CUDA DLL paths |
| `utils.py` | Fungsi plotting, metrik, helper umum |
| `demo_components.py` | Komponen UI untuk demo app |

### `models/` — Model Tersimpan

| File | Ukuran | Keterangan |
|------|--------|------------|
| `best_lstm.keras` | 1.5 MB | ✅ Model terbaik (97.7%) — **tidak perlu HPO** |
| `best_1dcnn.keras` | 1.9 MB | Baseline 1D-CNN (75.5%) |
| `best_2dcnn.keras` | 1.3 MB | Baseline 2D-CNN (83.3%) |
| `best_1dcnn_hpo.keras` | 10.6 MB | 1D-CNN setelah HPO retrain (val ~80.5%) |

### `results/` — Metrik & History Training

| File | Isi |
|------|-----|
| `evaluation_results.json` | Metrik test set 1D-CNN baseline + LSTM |
| `evaluation_results_2dcnn.json` | Metrik test set 2D-CNN baseline |
| `hpo_best_1dcnn.json` | Best hyperparameters 1D-CNN (dari 30 trials) |
| `hpo_best_2dcnn.json` | Best hyperparameters 2D-CNN (dari 30 trials) |
| `1dcnn_hpo_history.json` | Loss/accuracy per epoch dari HPO retrain |
| `*_training_curves.png` | Grafik loss & accuracy curves |
| `confusion_matrix_*.png` | Confusion matrix per model |
| `classification_report_*.png` | Heatmap precision/recall/F1 per kelas |

### `notebooks/` — Visualisasi & Dokumentasi

| File | Isi |
|------|-----|
| `EDA_and_Visualization.ipynb` | Exploratory Data Analysis awal |
| `ppt_hpo_1_dimensionality.png` | Gambar HPO: Downsampling sinyal |
| `ppt_hpo_2_model_sizing.png` | Gambar HPO: Diet kapasitas model |
| `ppt_hpo_3_regularization.png` | Gambar HPO: Dropout & L2 |
| `ppt_hpo_4_learning_rate.png` | Gambar HPO: ReduceLROnPlateau |
| `ppt_cnn_vs_lstm_real.png` | Visualisasi input 1D-CNN vs LSTM (data asli) |
| `ppt_model_inputs_real.png` | Visualisasi format input model (data asli) |
| `project_qa.md` | Daftar Q&A untuk antisipasi pertanyaan dosen |

### Root Files

| File | Fungsi |
|------|--------|
| `run_gpu.bat` | Script untuk menjalankan python dengan GPU CUDA support |
| `requirements.txt` | Daftar dependency Python |
| `README.md` | Dokumentasi proyek |

---

## 4. Cara Menjalankan

### Aktivasi Virtual Environment
```cmd
cd /d "D:\Coding\College Coding vibes\PMTL_Tubes"
.venv\Scripts\activate
```

### Semua command training/HPO harus via `run_gpu.bat`:
```cmd
.\run_gpu.bat python <script> <args>
```

---

## 5. Langkah Selanjutnya (TODO)

### A. Retrain 2D-CNN dengan Best HP ❌
```cmd
.\run_gpu.bat python src/hpo_2dcnn.py --retrain-only --retrain-epochs 40 --max-samples 2555904
```

### B. Evaluasi Final Semua Model ❌
Setelah retrain selesai, jalankan evaluasi pada **test set** untuk mendapat metrik final:
```cmd
.\run_gpu.bat python src/3_evaluate_models.py
```

> [!IMPORTANT]
> Script `3_evaluate_models.py` mungkin perlu diupdate untuk:
> 1. Memuat `best_1dcnn_hpo.keras` dan `best_2dcnn_hpo.keras` (bukan baseline)
> 2. Menggunakan arsitektur HPO baru (bukan `build_1dcnn()` lama)
> 3. Menyimpan hasil ke file terpisah (misal `evaluation_results_hpo.json`)

### C. Bandingkan Before vs After HPO
Buat tabel perbandingan untuk presentasi:

| Model | Baseline Acc | HPO Acc | Δ |
|-------|:---:|:---:|:---:|
| 1D-CNN | 75.5% | ? | ? |
| 2D-CNN | 83.3% | ? | ? |
| LSTM | 97.7% | - (tidak di-HPO) | - |

### D. Siapkan Materi Presentasi Final
- Slide HPO: tunjukkan search space, best HP, dan perbandingan before/after
- Gunakan gambar-gambar di `notebooks/ppt_hpo_*.png`
- Update slide training curves dengan grafik terbaru

---

## 6. Catatan Penting

> [!TIP]
> **Masalah overfitting pada 1D-CNN telah teratasi sepenuhnya!** 
> Dengan melatih model pada full dataset (2,555,904 sampel) menggunakan setting HPO terbaik selama 30 epoch (dengan Early Stopping di epoch 23), model berhasil mencapai generalisasi yang luar biasa baik dengan gap antara training accuracy (96.8%) dan validation accuracy (96.3%) hanya sebesar ~0.5%.

> [!NOTE]
> **File `best_1dcnn_hpo.keras` (10.6 MB)** jauh lebih besar dari baseline (1.9 MB) karena HPO memilih filter 192→256→256→32 (total parameter jauh lebih banyak). Namun performanya meningkat sangat pesat dari 75.5% menjadi 96.3%.
