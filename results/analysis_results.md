# 📊 Analisis Hasil Pengujian Demo — Wireless Jamming Detection

## Rangkuman Singkat

Kedua model (1D-CNN & LSTM) **sangat baik** pada SJR ≤ 0 dB (jamming kuat), tetapi **mulai gagal di SJR +10 dB** (jamming lemah). Efek dunia nyata (Doppler + Multipath) menyebabkan **false positive CW Jamming** pada sinyal normal.

---

## 1. Analisis Per-Temuan

### Temuan #1: 1D-CNN + CW + SJR +10 → Tidak Konsisten

![1D-CNN CW +10 dB](file:///C:/Users/Acer/.gemini/antigravity/brain/16a951e0-5913-41db-a5c7-3523fccfa3a2/1d_cw_plus10.png)

**Penjelasan:**
SJR +10 dB artinya **sinyal asli 10× lebih kuat dari jamming**. Ini seperti ada orang bisik-bisik di ruangan yang berisik — sangat susah didengar. Jamming CW pada SJR +10 hanya menambahkan **gelombang sinus sangat kecil** ke sinyal, sehingga perubahan pada pola I/Q nyaris tidak terlihat.

Model ada di **zona abu-abu**: confidence hanya ~67% (seharusnya >90% kalau yakin). Setiap kali sinyal di-load ulang dari cache, variasi floating-point kecil bisa menggeser keputusan bolak-balik antara Normal dan CW.

> **Kesimpulan:** Ini adalah **batas deteksi** (detection threshold) model 1D-CNN untuk CW jamming.

---

### Temuan #2: 1D-CNN + Barrage + SJR +10 → Selalu "Normal"

![1D-CNN Barrage +10 dB](file:///C:/Users/Acer/.gemini/antigravity/brain/16a951e0-5913-41db-a5c7-3523fccfa3a2/1d_barrage_plus10.png)

**Penjelasan:**
Barrage jamming pada SJR +10 dB hanya menambahkan **noise level sangat rendah**. Masalahnya: dataset DeepSig **sudah mengandung noise** (karena berbagai SNR). Jadi model melihat noise tambahan ini sebagai bagian natural dari sinyal.

Berbeda dengan CW yang punya **pola spesifik** (gelombang sinus = spike di frekuensi tertentu), barrage hanya noise acak — **tidak ada "tanda tangan" unik** yang bisa dideteksi di power rendah.

> **Kesimpulan:** Barrage jamming **lebih susah dideteksi** daripada CW pada power rendah, karena polanya mirip noise natural.

---

### Temuan #3 & #4: LSTM + SJR +10 → Sedikit Lebih Baik

**Penjelasan:**
LSTM memiliki keunggulan di sini karena arsitekturnya menganalisis **urutan temporal** (pola berurutan dalam sinyal). CW jamming, meskipun lemah, tetap membuat **pergeseran fase periodik** yang bisa ditangkap LSTM tapi terlewat oleh CNN yang melihat pola lokal.

Untuk barrage, LSTM juga sedikit lebih baik (confidence Normal ~49% vs CNN ~96%) — artinya LSTM "lebih ragu" dan kadang bisa mendeteksi barrage, meski masih sering salah.

> **Kesimpulan:** LSTM lebih sensitif terhadap jamming lemah, terutama CW.

---

### Temuan #5: Skenario 18 & 20 — CW + Efek Dunia Nyata → Berubah-ubah

**Penjelasan:**
Pada skenario 18 (CW + Doppler 60 + Multipath 3), ada **dua sumber distorsi** yang saling berinteraksi:
- **Doppler shift** menggeser frekuensi → komponen CW bergeser
- **Multipath fading** menciptakan interferensi antar path

Kadang kombinasi ini **memperkuat** pola CW (confidence tinggi ke CW), kadang **meredam** pola CW tapi **menciptakan pola baru** yang mirip barrage (confidence ke barrage). Ini menunjukkan model **sensitif terhadap fase acak** dari multipath.

---

### Temuan #6: Skenario 21 — Normal + Extreme Effects → "CW Jamming"

![Skenario 21: False Positive](file:///C:/Users/Acer/.gemini/antigravity/brain/16a951e0-5913-41db-a5c7-3523fccfa3a2/skenario_21.png)

**Ini temuan paling penting!** Sinyal normal + Doppler 200 km/h + Multipath 6 paths → model prediksi **CW Jamming 82.8%**.

**Mengapa ini terjadi:**
Doppler shift pada 200 km/h menghasilkan **pergeseran frekuensi konstan** pada sinyal. Dari perspektif model, pergeseran frekuensi konstan ini terlihat **sangat mirip** dengan CW jamming (yang juga menambahkan komponen frekuensi tunggal). Model tidak pernah dilatih dengan efek Doppler, jadi tidak bisa membedakan keduanya.

> **Ini adalah kelemahan fundamental** dari training dengan data sintetis tanpa efek dunia nyata.

---

## 2. Perbandingan Model

| Aspek | 1D-CNN | LSTM | Pemenang |
|-------|--------|------|----------|
| **Akurasi Test Set** | 97.0% | 98.2% | 🏆 LSTM |
| **Deteksi CW (SJR 0)** | ✅ Sangat baik | ✅ Sangat baik | Seri |
| **Deteksi CW (SJR +10)** | ⚠️ Tidak konsisten (~67%) | ⚠️ Tidak konsisten (~65%) | Seri |
| **Deteksi Barrage (SJR 0)** | ✅ Baik | ✅ Baik | Seri |
| **Deteksi Barrage (SJR +10)** | ❌ Selalu Normal (96%) | ⚠️ Ragu-ragu (49%) | 🏆 LSTM |
| **Tahan Doppler** | ❌ False positive CW | ❌ False positive CW | Seri (sama-sama gagal) |
| **Tahan Multipath** | ⚠️ Sedikit terpengaruh | ⚠️ Sedikit terpengaruh | Seri |
| **Latency** | 🏆 ~36 ms | ~47 ms | 🏆 1D-CNN |
| **Stabilitas Prediksi** | Lebih stabil | Lebih fluktuatif | 🏆 1D-CNN |

---

## 3. Kesimpulan Utama

### ✅ Kekuatan
1. **Sangat efektif pada jamming kuat** (SJR ≤ 0): akurasi 97-98%
2. **CW lebih mudah dideteksi** daripada barrage karena punya "tanda tangan" frekuensi yang jelas
3. **LSTM sedikit lebih sensitif** terhadap jamming lemah karena menangkap pola temporal

### ⚠️ Kelemahan
1. **Batas deteksi di SJR +10 dB**: kedua model kesulitan mendeteksi jamming lemah
2. **Barrage lebih susah dideteksi** daripada CW pada power rendah
3. **Efek Doppler menyebabkan false positive CW**: model menganggap pergeseran frekuensi konstan dari Doppler sebagai CW jamming

### 💡 Rekomendasi untuk Improvement
1. **Data Augmentation**: Latih ulang model dengan **menambahkan efek Doppler dan Multipath ke data training** → ini akan mengurangi false positive
2. **Threshold Adaptif**: Implementasi confidence threshold (misal: hanya alarm jika confidence > 80%)
3. **Feature Engineering**: Tambahkan fitur yang bisa membedakan Doppler dari CW (misal: rate perubahan fase)

### 🎯 Kapan Pakai Model Yang Mana?

| Skenario | Pilihan | Alasan |
|----------|---------|--------|
| Real-time / latency critical | **1D-CNN** | 30% lebih cepat |
| Perlu sensitivitas tinggi | **LSTM** | Lebih bisa mendeteksi jamming lemah |
| Barrage detection penting | **LSTM** | CNN hampir tidak bisa deteksi barrage lemah |
| Butuh hasil stabil | **1D-CNN** | Prediksi lebih konsisten |
