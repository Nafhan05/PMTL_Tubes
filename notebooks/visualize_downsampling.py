"""Visualisasi downsampling 1024 → 256 untuk penjelasan ke dosen."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# Buat sinyal contoh: sinyal bersih + noise
t_1024 = np.linspace(0, 1, 1024)
clean_signal = np.sin(2 * np.pi * 5 * t_1024)  # gelombang 5 Hz
noise = np.random.randn(1024) * 0.3
noisy_signal = clean_signal + noise

# Downsample: ambil 256 titik dari 1024 (setiap ~4 titik)
indices = np.linspace(0, 1023, 256, dtype=int)
t_256 = t_1024[indices]
downsampled_signal = noisy_signal[indices]

# ===== PLOT 1: Original vs Downsampled =====
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Sinyal asli 1024 titik
axes[0].plot(t_1024, noisy_signal, linewidth=0.5, color="#2196F3", alpha=0.8)
axes[0].plot(t_1024, clean_signal, linewidth=2, color="#FF5722", alpha=0.6, label="Pola asli (tanpa noise)")
axes[0].set_title("Sinyal ASLI — 1024 titik (yang dilihat CNN v1)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Amplitudo")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)
axes[0].annotate("Banyak detail noise\n→ CNN bisa menghafal noise ini", 
                xy=(0.6, 1.0), fontsize=11, color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# Downsampled 256 titik
axes[1].plot(t_256, downsampled_signal, linewidth=1.0, color="#4CAF50", alpha=0.8)
axes[1].plot(t_1024, clean_signal, linewidth=2, color="#FF5722", alpha=0.6, label="Pola asli (tanpa noise)")
axes[1].set_title("Sinyal DOWNSAMPLED — 256 titik (yang dilihat CNN v3 & LSTM)", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Amplitudo")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)
axes[1].annotate("Detail noise berkurang\n→ Model fokus ke POLA sinyal", 
                xy=(0.6, 1.0), fontsize=11, color="green",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew"))

# Overlay perbandingan
axes[2].plot(t_1024, noisy_signal, linewidth=0.3, color="#2196F3", alpha=0.4, label="1024 titik (detail)")
axes[2].plot(t_256, downsampled_signal, 'o-', markersize=2, linewidth=1.0, color="#4CAF50", alpha=0.8, label="256 titik (downsampled)")
axes[2].set_title("PERBANDINGAN — Overlay", fontsize=14, fontweight="bold")
axes[2].set_xlabel("Waktu (detik)")
axes[2].set_ylabel("Amplitudo")
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\Acer\.gemini\antigravity\brain\16a951e0-5913-41db-a5c7-3523fccfa3a2\downsampling_visual.png", dpi=150, bbox_inches="tight")
print("Plot 1 saved!")

# ===== PLOT 2: Analogi sederhana =====
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Analogi: foto resolusi tinggi vs rendah
# Kiri: "1024 titik" - terlalu banyak detail
ax = axes2[0]
x_dense = np.linspace(0, 2*np.pi, 1024)
y_pattern = np.sin(x_dense) + 0.5*np.sin(3*x_dense)
y_noisy = y_pattern + np.random.randn(1024) * 0.4
ax.fill_between(x_dense, y_noisy, alpha=0.3, color="#2196F3")
ax.plot(x_dense, y_noisy, linewidth=0.3, color="#2196F3")
ax.plot(x_dense, y_pattern, linewidth=3, color="#FF5722", label="Pola penting")
ax.set_title("1024 titik: Noise MENDOMINASI\n→ CNN menghafal noise", fontsize=12, fontweight="bold", color="red")
ax.legend()
ax.grid(True, alpha=0.2)

# Kanan: "256 titik" - pola terlihat jelas
ax = axes2[1]
idx_256 = np.linspace(0, 1023, 256, dtype=int)
x_sparse = x_dense[idx_256]
y_sparse = y_noisy[idx_256]
y_pat_sparse = y_pattern[idx_256]
ax.fill_between(x_sparse, y_sparse, alpha=0.3, color="#4CAF50")
ax.plot(x_sparse, y_sparse, linewidth=1, color="#4CAF50")
ax.plot(x_sparse, y_pat_sparse, linewidth=3, color="#FF5722", label="Pola penting")
ax.set_title("256 titik: POLA terlihat jelas\n→ Model belajar pola", fontsize=12, fontweight="bold", color="green")
ax.legend()
ax.grid(True, alpha=0.2)

plt.suptitle("Mengapa Downsampling Membantu?", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(r"C:\Users\Acer\.gemini\antigravity\brain\16a951e0-5913-41db-a5c7-3523fccfa3a2\downsampling_analogy.png", dpi=150, bbox_inches="tight")
print("Plot 2 saved!")
