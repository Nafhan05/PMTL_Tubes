# 📚 Jawaban Pertanyaan Dasar Proyek Deteksi Jamming

---

## 1️⃣ Bentuk Data Mentah (Raw Data)

### Apa itu I/Q?

**I/Q** adalah cara merepresentasikan sinyal radio secara digital:
- **I = In-phase** (komponen sefasa) → bayangkan sebagai "sumbu X" dari sinyal
- **Q = Quadrature** (komponen kuadratur) → bayangkan sebagai "sumbu Y" dari sinyal

Kenapa harus 2 komponen? Karena sinyal radio itu **berputar** (seperti jarum jam). Untuk merekam "posisi jarum" di setiap waktu, kita butuh 2 koordinat: posisi horizontal (I) dan vertikal (Q).

> **Analogi**: Kalau kamu mau catat posisi orang yang berlari melingkar di lapangan, kamu butuh 2 angka: jarak ke timur (I) dan jarak ke utara (Q).

### Arti `(1024, 2)`

Satu sample = **tabel 2 kolom dengan 1024 baris**:
- **1024 baris** = 1024 titik waktu berurutan (time series)
- **2 kolom** = komponen I dan Q

![Data mentah 1 sample](file:///C:/Users/Acer/.gemini/antigravity/brain/16a951e0-5913-41db-a5c7-3523fccfa3a2/q1_raw_data.png)

### Contoh Data Mentah (Baris 0-4)

```
Index    I (In-phase)    Q (Quadrature)
  0       0.042027        0.234763
  1      -0.272883        0.405135
  2      -0.267073        0.227499
  3      -0.314851       -0.176459
  4       0.963341       -1.025710
```

Setiap angka adalah **nilai amplitudo sinyal** pada titik waktu tersebut.

### Dimensi Seluruh Dataset

```
Seluruh dataset:  (2,555,904 × 1024 × 2)
                   ↑            ↑      ↑
              2.5 juta     1024 titik  I & Q
              sample        waktu    komponen
```

---

## 2️⃣ Hubungan Antar Sample

### Apakah setiap sample punya matriks I/Q sendiri?

**Ya!** Setiap sample dari 2.5 juta sample adalah **sinyal tersendiri** dengan matriks (1024, 2) sendiri.

### Di mana time series-nya?

Time series ada **DI DALAM** setiap sample, **bukan** antar sample.

![Hubungan antar sample](file:///C:/Users/Acer/.gemini/antigravity/brain/16a951e0-5913-41db-a5c7-3523fccfa3a2/q2_samples_relation.png)

### Penjelasan Rinci

| | Benar | Salah |
|---|---|---|
| **Time series** | 1024 titik waktu **di dalam** 1 sample berurutan | Sample #0 → Sample #1 → Sample #2 bukan urutan waktu |
| **Hubungan antar sample** | Setiap sample **independen** (tidak terhubung) | Bukan sinyal kontinyu yang dipotong-potong |

> **Analogi**: Bayangkan 2.5 juta **foto** yang masing-masing ukurannya 1024 × 2 piksel. Setiap foto berdiri sendiri. Tapi **di dalam** setiap foto, piksel-pikselnya berurutan dari kiri ke kanan.

---

## 3️⃣ Arsitektur Model

### Mengapa Arsitektur Ini?

| Model | Alasan Pemilihan |
|-------|-----------------|
| **LSTM** | Sinyal I/Q adalah time series → butuh model yang bisa **mengingat konteks urutan** |
| **1D-CNN** | Eksperimen alternatif → ternyata CNN hanya melihat pola lokal, tidak cukup untuk mendeteksi jamming |
| **2D-CNN** | Eksperimen dengan representasi visual (spektrogram) → CNN bagus untuk pengenalan gambar |

### Parameter Penting

| Parameter | Arti | Nilai | Mengapa? |
|-----------|------|-------|----------|
| **seq_len** | Panjang input (jumlah titik waktu) | 256 | Downsampled dari 1024 untuk mengurangi noise |
| **L2 regularization** | "Penalti" agar bobot tidak terlalu besar | 1e-4 | Mencegah model menghafal noise (overfitting) |
| **Dropout** | Matikan secara acak sebagian neuron saat training | 0.5 | Memaksa model belajar fitur yang redundan |
| **BatchNorm** | Normalisasi output setiap layer | - | Mempercepat training dan menstabilkan |
| **SpatialDropout1D** | Matikan seluruh filter channel, bukan per-neuron | 0.1-0.2 | Lebih efektif untuk data sekuensial |

### Mengapa Downsampling?

Lihat penjelasan detail di artifact sebelumnya (`downsampling_explanation.md`). Ringkasnya: sinyal 1024 titik terlalu detail → model menghafal noise → downsampling ke 256 membuang noise tapi mempertahankan pola penting.

---

## 4️⃣ Input Model: Apa Bedanya?

![Perbandingan input 3 model](file:///C:/Users/Acer/.gemini/antigravity/brain/16a951e0-5913-41db-a5c7-3523fccfa3a2/q4_model_inputs.png)

### Input Setiap Model

| Model | Input Shape | Bentuk Data | Perlu Diubah? |
|-------|------------|-------------|---------------|
| **LSTM** | (256, 2) | Sinyal I/Q (downsampled) | Ya, downsampled dari (1024,2) |
| **1D-CNN** | (256, 2) | Sinyal I/Q (downsampled) | Ya, downsampled dari (1024,2) |
| **2D-CNN** | (64, 61, 1) | Spektrogram (gambar) | Ya, STFT dari sinyal I/Q |

### Apakah data awalnya 1 dimensi?

**Tidak tepat.** Data awalnya sudah **(1024, 2)** — ini bisa disebut **data 1D dengan 2 channel**, mirip audio stereo (kiri-kanan). Model 1D-CNN dan LSTM menerima data ini langsung. "1D" di sini merujuk ke **1 sumbu waktu**, bukan benar-benar 1 angka.

### 2D-CNN: Diubah menjadi bentuk apa?

Sinyal (1024, 2) → **STFT** → Gambar (64, 61, 1):
- **64** = jumlah frekuensi yang dianalisis (sumbu Y)
- **61** = jumlah "jendela waktu" (sumbu X)
- **1** = 1 channel warna (grayscale)

Hasilnya: "foto sinar-X" dari sinyal → warna terang = frekuensi aktif.

### Perbedaan Fundamental: CNN vs LSTM

![CNN vs LSTM cara melihat data](file:///C:/Users/Acer/.gemini/antigravity/brain/16a951e0-5913-41db-a5c7-3523fccfa3a2/q4_cnn_vs_lstm.png)

| Aspek | 1D-CNN | LSTM |
|-------|--------|------|
| **Cara melihat data** | **Filter geser** — lihat potongan kecil sinyal (misal 7 titik), lalu geser | **Membaca berurutan** — dari titik 0 sampai titik terakhir |
| **Apa yang dipelajari** | Pola **lokal** (spike, edge, perubahan mendadak) | Pola **temporal** (alur perubahan, tren jangka panjang) |
| **Memori** | ❌ Tidak punya memori antar posisi | ✅ Punya "memori" (cell state) yang menyimpan konteks |
| **Analogi** | Membaca buku dengan **kaca pembesar** — hanya lihat 1 paragraf | Membaca buku **halaman per halaman** — paham ceritanya |
| **Cocok untuk** | Sinyal dengan fitur lokal jelas | Sinyal yang polanya ada di urutan waktu |

### Kenapa LSTM Menang?

Jamming mengubah **karakteristik sinyal secara keseluruhan**, bukan di 1 titik:
- **CW Jamming**: menambah gelombang sinus → perlu melihat **konteks panjang** untuk mendeteksi frekuensi tambahan
- **Barrage Jamming**: menaikkan noise level → perlu **membandingkan** level noise di seluruh sinyal

LSTM bisa melakukan ini karena punya memori. CNN hanya melihat 7 titik sekaligus → tidak bisa menangkap perubahan yang tersebar di seluruh sinyal.

---

## 🎯 Ringkasan Jawaban Cepat (Jika Ditanya Dosen)

> **"Apa itu I/Q?"**
> I/Q adalah representasi digital sinyal radio. I = komponen sefasa, Q = komponen kuadratur. Keduanya merepresentasikan posisi sinyal yang berputar di setiap titik waktu.

> **"Apa arti 1024 × 2?"**
> 1024 titik waktu berurutan, masing-masing punya 2 nilai (I dan Q). Satu sample = tabel 1024 baris × 2 kolom.

> **"Time series di mana?"**
> Time series ada di dalam setiap sample (1024 titik berurutan). Antar sample tidak terhubung — setiap sample adalah sinyal independen.

> **"Kenapa LSTM lebih baik dari CNN?"**
> Karena jamming mengubah sinyal secara keseluruhan, bukan di satu titik. LSTM membaca sinyal dari awal sampai akhir dan mengingat konteks, sehingga bisa mendeteksi perubahan yang tersebar. CNN hanya melihat potongan kecil, sehingga kehilangan gambaran besar.

> **"Kenapa downsampling?"**
> Mengurangi titik dari 1024 ke 256 menghilangkan detail noise yang tidak informatif. Model dipaksa belajar pola sinyal yang penting, bukan menghafal noise.
