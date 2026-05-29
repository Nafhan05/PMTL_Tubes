"""
Buat ulang 2 gambar PPT-ready dengan data asli dari dataset.
"""
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = r"C:\Users\Acer\.gemini\antigravity\brain\16a951e0-5913-41db-a5c7-3523fccfa3a2"
HDF5 = r"D:\Coding\College Coding vibes\PMTL_Tubes\data\GOLD_XYZ_OSC.0001_1024.hdf5"

NPERSEG = 64
HOP = 16

# Load sample asli
with h5py.File(HDF5, "r") as f:
    sample = f["X"][100]  # (1024, 2)

# Downsampled version
idx_256 = np.linspace(0, 1023, 256, dtype=int)
sample_256 = sample[idx_256]  # (256, 2)

# Spektrogram
complex_sig = sample[:, 0] + 1j * sample[:, 1]
frame_starts = np.arange(0, 1024 - NPERSEG + 1, HOP)
frame_idx = frame_starts[:, None] + np.arange(NPERSEG)
window = np.hanning(NPERSEG)
frames = complex_sig[frame_idx] * window[None, :]
spec = np.fft.fft(frames, axis=1)
mag = np.abs(spec).T  # (64, 61)
log_mag = np.log10(mag + 1e-10)
log_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-10)


# ================================================================
# GAMBAR 1: Perbandingan Input 3 Model (versi data asli)
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")

# Panel kiri: Input LSTM & 1D-CNN
ax = axes[0]
ax.plot(sample_256[:, 0], linewidth=1.0, color="#64B5F6", alpha=0.85, label="I (In-phase)")
ax.plot(sample_256[:, 1], linewidth=1.0, color="#EF9A9A", alpha=0.85, label="Q (Quadrature)")
ax.set_title("Input LSTM & 1D-CNN\n(256 titik × 2 channel)", fontsize=12, fontweight="bold", color="white")
ax.set_xlabel("Waktu (sample index)", color="#aaaaaa")
ax.set_ylabel("Amplitudo", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#333366")
ax.legend(fontsize=9, facecolor="#0f3460", labelcolor="white")
ax.grid(True, alpha=0.15, color="white")
ax.text(0.5, -0.20, "I dan Q masuk BERSAMAAN\n(seperti audio stereo)",
        ha="center", fontsize=10, color="#FFD54F", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f3460", edgecolor="#FFD54F", linewidth=1.2))

# Panel tengah: Input 2D-CNN (spektrogram)
ax = axes[1]
im = ax.imshow(log_mag, aspect="auto", cmap="plasma", origin="lower")
ax.set_title("Input 2D-CNN\n(Spektrogram 64×61 px)", fontsize=12, fontweight="bold", color="white")
ax.set_xlabel("Waktu (frame)", color="#aaaaaa")
ax.set_ylabel("Frekuensi (bin)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#333366")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Log-magnitude", color="#aaaaaa")
cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")
ax.text(0.5, -0.20, "Sinyal I/Q diubah jadi GAMBAR\n(peta frekuensi vs waktu)",
        ha="center", fontsize=10, color="#80DEEA", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f3460", edgecolor="#80DEEA", linewidth=1.2))

# Panel kanan: Cara model melihat
ax = axes[2]
ax.set_facecolor("#0f3460")
ax.axis("off")
comparison_text = (
    "CARA MODEL MELIHAT DATA:\n\n"
    "1D-CNN:\n"
    "  Filter geser kiri -> kanan\n"
    "  Cari pola LOKAL (potongan kecil)\n"
    "  Tidak punya memori konteks\n\n"
    "LSTM:\n"
    "  Baca dari awal sampai akhir\n"
    "  Ingat konteks sebelumnya\n"
    "  Tangkap pola TEMPORAL panjang\n\n"
    "2D-CNN:\n"
    "  Lihat gambar spektrogram\n"
    "  Kenali pola VISUAL frekuensi\n"
    "  Seperti klasifikasi gambar"
)
ax.text(0.08, 0.95, comparison_text, fontsize=10.5, fontfamily="monospace",
        va="top", color="white", transform=ax.transAxes, linespacing=1.6)
ax.set_title("Ringkasan Cara Kerja", fontsize=12, fontweight="bold", color="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333366")

plt.suptitle("Perbandingan Input 3 Model — Data Asli (Sample #100, OOK)",
             fontsize=14, fontweight="bold", color="white", y=1.02)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/ppt_model_inputs_real.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Gambar 1 saved: ppt_model_inputs_real.png")


# ================================================================
# GAMBAR 2: CNN vs LSTM cara melihat data (versi data asli)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("#1a1a2e")
for ax in axes:
    ax.set_facecolor("#16213e")

signal = sample_256[:50, 0]  # 50 titik pertama komponen I untuk kejelasan

# --- Panel kiri: 1D-CNN ---
ax = axes[0]
ax.plot(signal, color="#64B5F6", linewidth=1.8, zorder=2)
ax.set_title("1D-CNN: Filter Geser (Sliding Window)", fontsize=13, fontweight="bold", color="white")
ax.set_xlabel("Waktu (sample index)", color="#aaaaaa")
ax.set_ylabel("Amplitudo (I)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
ax.grid(True, alpha=0.15, color="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333366")

# Highlight 3 window filter
windows = [(3, "#FF7043"), (12, "#FFB300"), (21, "#AB47BC")]
for start, color in windows:
    size = 8
    ax.axvspan(start, start + size, alpha=0.35, color=color, zorder=1)
    mid = start + size // 2
    ax.annotate(f"Filter\n[{start}:{start+size}]",
                xy=(mid, signal[mid]),
                xytext=(mid, signal[mid] + 0.7),
                ha="center", fontsize=9, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", edgecolor=color))

ax.text(0.5, -0.18,
        "Melihat potongan kecil (misal 8 titik) lalu geser\nTidak tahu konteks di luar jendela",
        ha="center", fontsize=10.5, color="#FF8A65", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#3E2723", edgecolor="#FF7043", linewidth=1.2))

# --- Panel kanan: LSTM ---
ax = axes[1]
ax.plot(signal, color="#81C784", linewidth=1.8, zorder=2)
ax.set_title("LSTM: Membaca Berurutan (Sequential)", fontsize=13, fontweight="bold", color="white")
ax.set_xlabel("Waktu (sample index)", color="#aaaaaa")
ax.set_ylabel("Amplitudo (I)", color="#aaaaaa")
ax.tick_params(colors="#aaaaaa")
ax.grid(True, alpha=0.15, color="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333366")

# Panah berurutan menandakan pembacaan sekuensial
arrow_positions = list(range(0, 46, 5))
for i in range(len(arrow_positions) - 1):
    x0, x1 = arrow_positions[i], arrow_positions[i + 1]
    y0, y1 = signal[x0], signal[x1]
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#A5D6A7",
                                lw=1.5, mutation_scale=14))

ax.scatter([0], [signal[0]], s=120, color="#00E676", zorder=5, label="Mulai", edgecolors="white", linewidth=1)
ax.scatter([49], [signal[49]], s=120, color="#FF1744", zorder=5, label="Selesai", edgecolors="white", linewidth=1)
ax.legend(fontsize=10, facecolor="#0f3460", labelcolor="white", loc="upper right")

# Memory annotation
ax.annotate("Ingat konteks\ndari sini...",
            xy=(10, signal[10]), xytext=(5, -1.2),
            fontsize=9, color="#FFD54F", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#FFD54F", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", edgecolor="#FFD54F"))
ax.annotate("...sampai sini",
            xy=(40, signal[40]), xytext=(30, 1.3),
            fontsize=9, color="#FFD54F", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#FFD54F", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a2e", edgecolor="#FFD54F"))

ax.text(0.5, -0.18,
        "Membaca dari AWAL sampai AKHIR\nMenyimpan memori konteks -> paham pola temporal",
        ha="center", fontsize=10.5, color="#A5D6A7", transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1B5E20", edgecolor="#81C784", linewidth=1.2))

plt.suptitle("Perbedaan Cara 1D-CNN vs LSTM Melihat Data — Komponen I, Sample #100 (OOK)",
             fontsize=14, fontweight="bold", color="white", y=1.02)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/ppt_cnn_vs_lstm_real.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Gambar 2 saved: ppt_cnn_vs_lstm_real.png")
print("Selesai!")
