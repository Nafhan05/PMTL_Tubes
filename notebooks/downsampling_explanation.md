# 📐 Penjelasan Downsampling 1024 → 256

## Apa itu Downsampling?

**Downsampling** = mengambil **sebagian titik** dari data asli secara merata.

### Analogi Sederhana

Bayangkan kamu punya **buku 1024 halaman**. Kamu ingin membuat ringkasan.

| Metode | Penjelasan |
|--------|-----------|
| **Baca semua 1024 halaman** | Kamu tahu setiap detail, termasuk typo dan kotoran di halaman. Tapi kamu jadi **ingat kotoran-nya** juga, bukan cuma isi ceritanya |
| **Baca setiap halaman ke-4 (256 halaman)** | Kamu masih paham alur ceritanya, tapi **tidak terganggu** oleh detail yang tidak penting |

Itulah yang terjadi pada sinyal kita:

![Visualisasi Downsampling](file:///C:/Users/Acer/.gemini/antigravity/brain/16a951e0-5913-41db-a5c7-3523fccfa3a2/downsampling_visual.png)

### Secara Teknis

```
Sinyal asli:    [titik_0, titik_1, titik_2, titik_3, titik_4, ... titik_1023]
                 1024 titik data

Downsampled:    [titik_0,          titik_4,          titik_8, ... titik_1020]
                 256 titik data (ambil setiap ~4 titik secara merata)
```

Menggunakan `np.linspace(0, 1023, 256)` → mengambil 256 index yang **terdistribusi merata** dari 0 sampai 1023.

---

## Mengapa Downsampling Membantu?

![Mengapa Downsampling Membantu](file:///C:/Users/Acer/.gemini/antigravity/brain/16a951e0-5913-41db-a5c7-3523fccfa3a2/downsampling_analogy.png)

### Pada sinyal 1024 titik:
- Noise (kebisingan acak) punya **detail yang sangat kaya** — setiap titik noise beda-beda
- CNN yang kuat bisa **menghafal pola noise spesifik** dari data training
- Saat data baru (validasi/test) punya noise yang berbeda → CNN bingung → **overfit**

### Pada sinyal 256 titik:
- Detail noise **berkurang** (karena kita skip 3 dari 4 titik)
- Yang tersisa adalah **pola besar** yang stabil: bentuk gelombang CW, distribusi barrage noise
- Model **dipaksa belajar pola**, bukan menghafal noise
- Hasilnya: model lebih **generalize** (bagus di data baru)

---

## Apakah LSTM 1024 Akan Lebih Bagus dari LSTM 256?

**Jawaban singkat: Belum tentu, bahkan bisa lebih buruk.**

### Mengapa?

| Aspek | LSTM 256 | LSTM 1024 |
|-------|----------|-----------|
| **Training time** | ~21 jam | **~84+ jam** (4x lebih lambat!) |
| **Risiko overfit** | Rendah ✅ | Lebih tinggi ⚠️ |
| **Pola yang ditangkap** | Pola besar (cukup untuk deteksi) | Pola besar + detail noise |
| **Akurasi** | 97.7% | Kemungkinan serupa atau sedikit lebih tinggi |

### Penjelasan:

1. **Informasi penting untuk deteksi jamming itu berskala besar:**
   - CW Jamming = ada gelombang sinus tambahan → terlihat jelas bahkan di 256 titik
   - Barrage Jamming = noise level meningkat → juga terlihat di 256 titik
   - Kita tidak butuh detail per-titik untuk mendeteksi ini

2. **LSTM secara natural sudah cocok dengan sequence pendek:**
   - LSTM memproses data **secara berurutan** (timestep demi timestep)
   - Semakin panjang sequence → semakin susah LSTM **mengingat konteks awal** (vanishing gradient)
   - 256 timestep sudah cukup panjang untuk LSTM menangkap pola

3. **Diminishing returns:**
   - Dari 256 → 1024, informasi **tambahan** yang didapat kebanyakan adalah noise
   - Training **4x lebih lama** untuk peningkatan yang mungkin hanya 0.1-0.5%
   - Tidak worth it secara praktis

### Analogi:

> Bayangkan kamu diminta **menebak apakah ada gempa** dengan melihat grafik seismograf.
>
> - **256 titik**: Kamu bisa lihat getaran besar dengan jelas → cukup untuk bilang "ada gempa!"
> - **1024 titik**: Kamu bisa lihat getaran besar DAN riak-riak kecil → tapi riak kecil tidak membantu menebak gempa, malah bikin bingung kalau terlalu diperhatikan

---

## Ringkasan untuk Presentasi

### Poin-poin kunci:

1. **Downsampling** = mengurangi jumlah titik data dari 1024 → 256 secara merata
2. **Tujuan**: menghilangkan detail noise yang tidak penting, memaksa model fokus ke pola sinyal
3. **Efek pada CNN**: tanpa downsampling → CNN menghafal noise → overfit (75% val acc). Dengan downsampling → CNN belajar pola → diharapkan lebih baik
4. **Efek pada LSTM**: LSTM 256 sudah 97.7% karena:
   - LSTM natural untuk sequence pendek (menghindari vanishing gradient)
   - Informasi penting untuk deteksi jamming sudah tercakup di 256 titik
5. **LSTM 1024 tidak akan jauh lebih baik** — diminishing returns, training 4x lebih lama, risiko overfit lebih tinggi

### Jika dosen bertanya: "Kenapa tidak pakai 1024 saja?"

> "Kami melakukan downsampling dari 1024 ke 256 sebagai teknik **regularisasi implisit**. Pada eksperimen kami, CNN dengan input 1024 mengalami overfitting parah (val accuracy 75% vs train 97%) karena model menghafal pola noise spesifik pada data training. Dengan mengurangi resolusi ke 256, detail noise berkurang sehingga model dipaksa mempelajari fitur yang lebih general seperti pola frekuensi CW atau distribusi energi barrage. LSTM dengan input 256 mencapai 97.7% accuracy, menunjukkan bahwa informasi relevan untuk deteksi jamming sudah cukup terwakili pada resolusi tersebut."
