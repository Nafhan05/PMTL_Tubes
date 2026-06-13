# 📊 Analisis Hasil HPO — Wireless Jamming Detection

## 1. Ringkasan Hasil: Before vs After HPO

### Tabel Perbandingan Utama

| Model | Baseline Acc | **HPO Acc** | **Δ Perubahan** | F1 Baseline | **F1 HPO** |
|-------|:---:|:---:|:---:|:---:|:---:|
| **1D-CNN** | 75.5% | **96.3%** | **🟢 +20.8%** | 0.756 | **0.963** |
| **2D-CNN** | 83.3% | **83.4%** | **🔴 +0.1% (tidak berubah)** | 0.822 | **0.823** |
| LSTM | 97.7% | — (tidak di-HPO) | — | 0.977 | — |

---

## 2. Analisis Detail per Model

### ✅ 1D-CNN: SUKSES BESAR (75.5% → 96.3%)

**Ini hasil yang sangat bagus!** HPO berhasil mengatasi masalah overfitting yang sebelumnya sangat parah.

#### Apa yang berubah?

| Aspek | Baseline (Dulu) | HPO (Sekarang) |
|-------|:---:|:---:|
| Train Accuracy | 92% | 96.8% |
| Val Accuracy | 66-72% | **96.0%** |
| Gap Train-Val | **~24% (overfitting parah)** | **~0.8% (hampir tidak ada)** |
| Test Accuracy | 75.5% | **96.3%** |

#### Performa per Kelas (1D-CNN HPO)

| Kelas | Precision | Recall | F1-Score |
|-------|:---:|:---:|:---:|
| Normal | 0.913 | **0.996** | 0.953 |
| CW Jamming | **0.997** | 0.956 | 0.976 |
| Barrage Jamming | 0.985 | 0.939 | 0.961 |

**Interpretasi:** Model sekarang bisa membedakan ketiga kelas dengan sangat baik. Recall Normal 99.6% artinya hampir semua sinyal normal berhasil dikenali. CW Jamming punya precision 99.7% artinya kalau model bilang "ini CW Jamming", hampir pasti benar.

#### Training History — Bukti Overfitting Teratasi

```
Epoch  | Train Acc | Val Acc  | Gap    | Keterangan
-------|-----------|----------|--------|------------------
1      | 91.5%     | 94.5%    | -3.0%  | Val lebih tinggi (masih belajar)
5      | 95.6%     | 94.8%    | 0.8%   | Seimbang
10     | 96.0%     | 94.8%    | 1.2%   | Gap kecil
18     | 96.4%     | 96.4%    | 0.0%   | Sempurna seimbang!
30     | 96.8%     | 96.0%    | 0.8%   | Akhir training, gap sangat kecil
```

Learning rate secara otomatis turun dari `0.00183` → `0.000057` selama 30 epoch, membantu model konvergen dengan halus.

#### Kenapa bisa bagus?

1. **Kernel size 7 (besar)** — Sinyal I/Q punya pola temporal yang panjang. Filter besar bisa "melihat" pola lebih luas daripada filter kecil (3), sehingga menangkap fitur jamming yang tersebar di sepanjang sinyal.

2. **Filter pyramid 192→256→256→32** — Berbeda dari baseline yang naik bertahap (64→128), arsitektur ini langsung agresif di awal (menangkap banyak fitur dasar) lalu menyempit di akhir (memaksa model meringkas informasi).

3. **Spatial Dropout rendah (0.05)** — Ternyata data ini tidak butuh regularisasi berat. Dropout terlalu tinggi di baseline justru "membunuh" informasi penting.

4. **Learning rate lebih tinggi (0.00183 vs 0.0003)** — Model baseline terlalu pelan belajar sehingga stuck di local minimum. Learning rate lebih tinggi membantu model "melompat" keluar dari solusi yang buruk.

---

### 🔴 2D-CNN: TIDAK BERUBAH SIGNIFIKAN (83.3% → 83.4%)

**HPO praktis tidak mengubah performa 2D-CNN.** Ini bukan berarti HPO gagal — ini menunjukkan masalahnya bukan di hyperparameter.

#### Performa per Kelas (2D-CNN HPO)

| Kelas | Precision | Recall | F1-Score |
|-------|:---:|:---:|:---:|
| Normal | **0.989** | **0.501** ⚠️ | 0.665 |
| CW Jamming | 0.999 | 0.994 | 0.996 |
| Barrage Jamming | 0.676 | **0.998** | 0.806 |

**Masalah utama tetap sama:** Kelas **Normal** punya recall hanya 50.1%. Artinya **setengah dari sinyal normal salah diklasifikasi** sebagai Barrage Jamming (62.716 dari 125.799 sample normal salah prediksi). Ini terlihat jelas di confusion matrix.

#### Training History — Bukti Model Stuck

```
Epoch  | Train Acc | Val Acc  | Keterangan
-------|-----------|----------|------------------
1      | 80.9%     | 81.9%    | Awal
10     | 82.9%     | 82.9%    | Sudah plateau
20     | 83.0%     | 83.0%    | Tidak naik lagi
40     | 83.1%     | 83.1%    | 40 epoch, masih sama
```

Model sudah **saturated** (jenuh) — tidak peduli berapa lama training, accuracy tidak naik lagi.

#### Kenapa tidak berubah?

1. **Masalahnya di representasi data, bukan hyperparameter.** Spektrogram dari sinyal Normal dan Barrage Jamming terlihat sangat mirip secara visual. Konversi STFT kehilangan informasi temporal yang justru penting untuk membedakan keduanya.

2. **HPO hanya mengubah "resep masak", bukan "bahan masak".**  HPO mengoptimalkan arsitektur model (jumlah layer, filter, dropout, dll). Tapi kalau input data-nya sendiri (spektrogram) tidak cukup informatif, model terbaik sekalipun tidak akan bisa.

3. **Buktinya:** Semua 30 trial HPO menghasilkan accuracy di kisaran 82-84% — tidak ada satupun yang bisa tembus 85%. Ini menunjukkan ada "langit-langit" (ceiling) yang bukan disebabkan oleh arsitektur model.

4. **Bandingkan:** 1D-CNN dan LSTM yang langsung mengolah sinyal mentah (time-domain) bisa mencapai 96-97%. Ini membuktikan bahwa representasi time-domain lebih cocok untuk tugas ini.

---

## 3. Penjelasan Metode HPO (untuk dijelaskan ke dosen)

### Apa itu HPO?

> **Hyperparameter Optimization (HPO)** adalah proses mencari konfigurasi terbaik untuk model deep learning secara **otomatis**, menggantikan proses trial-and-error manual.

Bayangkan Anda memasak. **Parameter** itu seperti rasa masakan yang otomatis berubah selama memasak (model belajar sendiri). Tapi **hyperparameter** itu seperti **suhu oven, lama masak, ukuran panci** — hal-hal yang harus Anda tentukan **sebelum** mulai memasak. HPO membantu mencari kombinasi terbaik dari pengaturan-pengaturan ini.

### Metode: Bayesian Optimization

Kita menggunakan **Bayesian Optimization** via library `keras-tuner`. Ini berbeda dari pendekatan lain:

| Metode | Cara Kerja | Efisiensi |
|--------|-----------|-----------|
| **Grid Search** | Coba SEMUA kombinasi | ❌ Sangat lambat (ribuan percobaan) |
| **Random Search** | Coba kombinasi acak | ⚠️ Bisa miss konfigurasi bagus |
| **Bayesian Optimization** ✅ | Belajar dari percobaan sebelumnya | ✅ Cerdas & efisien |

#### Cara kerja Bayesian Optimization (analogi):

1. **Trial 1-5 (Eksplorasi awal):** Model mencoba konfigurasi acak, seperti "lempar umpan ke berbagai tempat di kolam untuk cari ikan."
2. **Trial 6-30 (Eksploitasi cerdas):** Berdasarkan hasil trial sebelumnya, algoritma **memprediksi** di mana kemungkinan konfigurasi bagus berada, lalu fokus mencari di area tersebut. Seperti "ikan banyak di pojok kiri, mari fokus pancing di situ."

#### Apa saja yang di-optimize?

**Search Space 1D-CNN:**
| Hyperparameter | Range yang Dicari | Best Value |
|----------------|:-----------------:|:----------:|
| Jumlah blok konvolusi | 2 – 4 | **4** |
| Ukuran kernel | 3, 5, 7 | **7** |
| Filter per blok | 32 – 256 | **192, 256, 256, 32** |
| Spatial Dropout | 0.05 – 0.4 | **0.05** |
| Dense units | 32, 64, 128 | **128** |
| Dense Dropout | 0.3 – 0.6 | **0.3** |
| L2 regularization | 1e-5, 1e-4, 1e-3 | **1e-5** |
| Learning rate | 0.0001 – 0.01 | **0.00183** |

**Search Space 2D-CNN:** (serupa tapi untuk arsitektur 2D + parameter STFT)

#### Setup Teknis:
- **30 trials** per model (total 60 percobaan)
- **15 epoch per trial** (dengan Early Stopping supaya trial buruk cepat berhenti)
- **500K sample subset** untuk search (supaya cepat), lalu **retrain di full 2.5M sample** dengan konfigurasi terbaik
- **40 epoch** untuk retrain final
- **ReduceLROnPlateau**: Learning rate otomatis diturunkan 50% jika val_loss tidak membaik selama 3 epoch berturut-turut

---

## 4. Kesimpulan untuk Presentasi

### Narasi yang bisa disampaikan ke dosen:

> "Kami melakukan Hyperparameter Optimization menggunakan Bayesian Optimization (30 trials) pada model 1D-CNN dan 2D-CNN. Hasilnya:
> 
> **1D-CNN mengalami peningkatan sangat signifikan** dari 75.5% menjadi 96.3% accuracy. Masalah overfitting yang sebelumnya terjadi (gap 24% antara train dan validation) berhasil diatasi sepenuhnya (gap turun ke <1%). HPO menemukan bahwa konfigurasi optimal memerlukan kernel size besar (7) dan learning rate lebih tinggi — dua hal yang tidak intuitif dan sulit ditemukan secara manual.
> 
> **2D-CNN tidak mengalami peningkatan** (83.3% → 83.4%). Dari 30 trial, tidak ada konfigurasi yang berhasil menembus 85%. Ini menunjukkan bahwa bottleneck 2D-CNN bukan pada arsitektur model, melainkan pada **representasi data** — konversi STFT kehilangan informasi temporal kritis yang dibutuhkan untuk membedakan sinyal Normal dan Barrage Jamming. Hal ini konsisten dengan performa LSTM (97.7%) dan 1D-CNN HPO (96.3%) yang keduanya bekerja langsung di domain waktu."

### Ranking Model Final:

| Rank | Model | Accuracy | F1 Macro |
|:---:|-------|:---:|:---:|
| 🥇 | LSTM | 97.7% | 0.977 |
| 🥈 | **1D-CNN (HPO)** | **96.3%** | **0.963** |
| 🥉 | 2D-CNN (HPO) | 83.4% | 0.823 |
