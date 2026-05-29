import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SAVE_DIR = r"C:\Users\Acer\.gemini\antigravity\brain\16a951e0-5913-41db-a5c7-3523fccfa3a2"
HDF5 = r"D:\Coding\College Coding vibes\PMTL_Tubes\data\GOLD_XYZ_OSC.0001_1024.hdf5"

BG = "#0a0a23"
PANEL = "#11133a"
ACCENT1 = "#FF6B6B"
ACCENT2 = "#4ECDC4"
ACCENT3 = "#FFE66D"
ACCENT4 = "#A78BFA"
TEXT_W = "#EAEAEA"
TEXT_G = "#999999"

def style_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a5a")
    ax.tick_params(colors=TEXT_G, labelsize=11)

# Load sample once
with h5py.File(HDF5, "r") as f:
    sample = f["X"][100]

# ================================================================
# IMAGE 1: Data Dimensionality (Downsampling & STFT)
# ================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax1)

idx_256 = np.linspace(0, 1023, 256, dtype=int)
sample_256 = sample[idx_256]
t_full = np.linspace(0, 1, 1024)
t_ds = np.linspace(0, 1, 256)

ax1.plot(t_full, sample[:, 0] + 0.15, color=ACCENT1, alpha=0.5, linewidth=0.5,
         label="Raw I-channel (1024 titik) - Noisy")
ax1.plot(t_ds, sample_256[:, 0] - 0.15, color=ACCENT2, linewidth=1.8,
         label="Downsampled (256 titik) - Pola Esensial")
ax1.legend(loc="upper right", facecolor="#1a1a40", labelcolor="white", fontsize=11,
           framealpha=0.95, edgecolor="#333366")
ax1.set_title("Data Dimensionality (Downsampling & STFT)", color="white",
              fontweight="bold", fontsize=16, pad=15)
ax1.set_yticks([])
ax1.set_xlabel("Waktu (Normalized)", color=TEXT_G, fontsize=11)

# Inset: STFT spectrogram
ins = ax1.inset_axes([0.60, 0.03, 0.37, 0.45])
NPERSEG = 64
HOP = 16
cplx = sample[:, 0] + 1j * sample[:, 1]
starts = np.arange(0, 1024 - NPERSEG + 1, HOP)
idx_mat = starts[:, None] + np.arange(NPERSEG)
window = np.hanning(NPERSEG)
frames = cplx[idx_mat] * window[None, :]
spec = np.abs(np.fft.fft(frames, axis=1)).T
log_spec = np.log10(spec + 1e-10)
ins.imshow(log_spec, aspect="auto", cmap="magma", origin="lower")
ins.set_xticks([]); ins.set_yticks([])
ins.set_title("2D-CNN: Sinyal Utuh -> STFT Spektrogram (64,61,1)",
              color=ACCENT3, fontsize=9, fontweight="bold")
for s in ins.spines.values():
    s.set_edgecolor(ACCENT3)
    s.set_linewidth(2)

# Annotation
ax1.annotate("1D-CNN & LSTM:\n1024 -> 256 (downsampled)", xy=(0.15, -0.25),
             fontsize=11, color=ACCENT2, fontweight="bold", ha="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a40",
                       edgecolor=ACCENT2, linewidth=1.2))
ax1.annotate("2D-CNN:\n1024 -> STFT (64x61)", xy=(0.75, -0.25),
             fontsize=11, color=ACCENT3, fontweight="bold", ha="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a40",
                       edgecolor=ACCENT3, linewidth=1.2))

plt.tight_layout()
fig1.savefig(f"{SAVE_DIR}/ppt_hpo_1_dimensionality.png", dpi=250,
             bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print("1/4 saved: ppt_hpo_1_dimensionality.png")


# ================================================================
# IMAGE 2: Model Sizing ("Diet" Kapasitas Parameter)
# ================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax2)

categories = ["1D-CNN\n(Filter Max)", "LSTM\n(Hidden Units)", "2D-CNN\n(Filter Blocks)"]
before_vals = [256, 256, 256]
after_vals  = [128, 128, 128]
after_labels = ["128", "128 -> 64", "32->64->128"]

x = np.arange(len(categories))
w = 0.30

bars_b = ax2.bar(x - w/2, before_vals, w, label="Desain Awal (Overfit Risk)",
                 color=ACCENT1, alpha=0.7, edgecolor="white", linewidth=0.5)
bars_a = ax2.bar(x + w/2, after_vals, w, label='Setelah "Diet" Kapasitas',
                 color=ACCENT2, alpha=0.9, edgecolor="white", linewidth=0.5)

for bar, lbl in zip(bars_a, after_labels):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             lbl, ha="center", va="bottom", color=ACCENT3, fontweight="bold", fontsize=13)

for i in range(len(categories)):
    ax2.annotate("", xy=(x[i] + w/2, after_vals[i] + 2),
                 xytext=(x[i] - w/2, before_vals[i] - 2),
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.8,
                                connectionstyle="arc3,rad=-0.3"))

ax2.set_ylabel("Kapasitas (Units / Filters)", color=TEXT_G, fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, color="white", fontsize=13)
ax2.legend(facecolor="#1a1a40", labelcolor="white", loc="upper right", fontsize=11,
           edgecolor="#333366")
ax2.set_title('Model Sizing ("Diet" Kapasitas Parameter)', color="white",
              fontweight="bold", fontsize=16, pad=15)

# Insight box
ax2.text(0.5, -0.18, "Tujuan: Memaksa model mencari pola ESENSIAL, bukan menghafal noise",
         transform=ax2.transAxes, ha="center", fontsize=12, color=ACCENT3, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a40",
                   edgecolor=ACCENT3, linewidth=1.2))

plt.tight_layout()
fig2.savefig(f"{SAVE_DIR}/ppt_hpo_2_model_sizing.png", dpi=250,
             bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print("2/4 saved: ppt_hpo_2_model_sizing.png")


# ================================================================
# IMAGE 3: Regularisasi Agresif (L2 & Dropout)
# ================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax3)
ax3.axis("off")
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_title("Regularisasi Agresif (L2 + Dropout)", color="white",
              fontweight="bold", fontsize=16, pad=15)

# --- LEFT HALF: Standard Dropout ---
ax3.text(0.22, 0.93, "Standard Dropout", color=ACCENT2, ha="center",
         fontsize=15, fontweight="bold")
ax3.text(0.22, 0.86, "(Digunakan pada LSTM: Rate 0.3 - 0.4)", color=TEXT_G,
         ha="center", fontsize=10)

np.random.seed(7)
for r in range(4):
    for c in range(5):
        cx = 0.06 + c * 0.08
        cy = 0.48 + r * 0.09
        is_dropped = np.random.rand() < 0.35
        if is_dropped:
            ax3.plot(cx, cy, 'o', color=ACCENT1, markersize=18, alpha=0.35)
            ax3.plot([cx - 0.018, cx + 0.018], [cy - 0.018, cy + 0.018],
                     color="white", lw=2.5)
            ax3.plot([cx + 0.018, cx - 0.018], [cy - 0.018, cy + 0.018],
                     color="white", lw=2.5)
        else:
            ax3.plot(cx, cy, 'o', color=ACCENT2, markersize=18)

ax3.text(0.22, 0.36, "Mematikan neuron secara ACAK\nsaat training berlangsung",
         color=TEXT_G, ha="center", fontsize=11, linespacing=1.5)
ax3.text(0.22, 0.26, "-> Mencegah model menghafal pola tertentu",
         color=ACCENT2, ha="center", fontsize=11, fontweight="bold")

# Separator line
ax3.plot([0.49, 0.49], [0.25, 0.95], color="#333366", lw=1.5, linestyle="--")

# --- RIGHT HALF: Spatial Dropout ---
ax3.text(0.75, 0.93, "Spatial Dropout", color=ACCENT4, ha="center",
         fontsize=15, fontweight="bold")
ax3.text(0.75, 0.86, "(Digunakan pada 1D/2D-CNN: Rate 0.1 - 0.2)", color=TEXT_G,
         ha="center", fontsize=10)

fmap_colors = [ACCENT2, ACCENT1, ACCENT4, ACCENT2, "#66BB6A"]
fmap_dropped = [False, True, False, False, False]
for i in range(5):
    x_start = 0.55 + i * 0.085
    alpha = 0.2 if fmap_dropped[i] else 0.9
    rect = patches.FancyBboxPatch(
        (x_start, 0.48), 0.065, 0.32,
        boxstyle="round,pad=0.01",
        facecolor=fmap_colors[i], alpha=alpha,
        edgecolor="white", linewidth=1.8
    )
    ax3.add_patch(rect)
    if fmap_dropped[i]:
        ax3.plot([x_start, x_start + 0.065], [0.48, 0.80], color="white", lw=3.5)
        ax3.plot([x_start + 0.065, x_start], [0.48, 0.80], color="white", lw=3.5)
        ax3.text(x_start + 0.0325, 0.44, "OFF", ha="center", color=ACCENT1,
                 fontsize=10, fontweight="bold")
    else:
        ax3.text(x_start + 0.0325, 0.44, f"FM{i+1}", ha="center", color=TEXT_G,
                 fontsize=8)

ax3.text(0.75, 0.36, "Mematikan satu PETA FITUR (Feature Map)\nsecara utuh per batch",
         color=TEXT_G, ha="center", fontsize=11, linespacing=1.5)
ax3.text(0.75, 0.26, "-> Mencegah model menghafal noise channel",
         color=ACCENT4, ha="center", fontsize=11, fontweight="bold")

# L2 box at bottom
ax3.text(0.5, 0.08, "+ L2 Regularizer (1e-4) pada kernel CNN & Dense\n"
         "-> Menahan bobot model agar tidak membengkak (weight decay)",
         color=ACCENT3, ha="center", fontsize=12, fontweight="bold", linespacing=1.5,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a40",
                   edgecolor=ACCENT3, linewidth=1.5))

plt.tight_layout()
fig3.savefig(f"{SAVE_DIR}/ppt_hpo_3_regularization.png", dpi=250,
             bbox_inches="tight", facecolor=BG)
plt.close(fig3)
print("3/4 saved: ppt_hpo_3_regularization.png")


# ================================================================
# IMAGE 4: Dynamic Learning Rate (ReduceLROnPlateau)
# ================================================================
fig4, ax4 = plt.subplots(figsize=(10, 6), facecolor=BG)
style_ax(ax4)

epochs = np.arange(0, 25)
np.random.seed(42)

val_loss = np.zeros(25)
val_loss[0] = 0.55
for i in range(1, 25):
    if i < 5:
        val_loss[i] = val_loss[i-1] - 0.06 + np.random.randn()*0.01
    elif i < 12:
        val_loss[i] = val_loss[i-1] - 0.01 + np.random.randn()*0.008
    elif i < 16:
        val_loss[i] = val_loss[i-1] + np.random.randn()*0.005
    elif i < 20:
        val_loss[i] = val_loss[i-1] - 0.015 + np.random.randn()*0.005
    else:
        val_loss[i] = val_loss[i-1] - 0.003 + np.random.randn()*0.003
val_loss = np.clip(val_loss, 0.05, 0.6)

lr = np.ones(25) * 3e-4
lr[14:] = 1.5e-4
lr[19:] = 7.5e-5

ax4.plot(epochs, val_loss, color=ACCENT1, linewidth=2.5, marker="o", markersize=4,
         label="Validation Loss", zorder=3)
ax4.set_xlabel("Epoch", color=TEXT_G, fontsize=12)
ax4.set_ylabel("Loss", color=ACCENT1, fontsize=13)
ax4.tick_params(axis="y", labelcolor=ACCENT1)

ax4_2 = ax4.twinx()
ax4_2.set_facecolor("none")
ax4_2.step(epochs, lr, where="post", color=ACCENT4, linewidth=2.8, linestyle="--",
           label="Learning Rate", zorder=2)
ax4_2.set_ylabel("Learning Rate", color=ACCENT4, fontsize=13)
ax4_2.tick_params(axis="y", labelcolor=ACCENT4)
ax4_2.set_yscale("log")
ax4_2.set_ylim(3e-5, 8e-4)

# Highlight plateau zones
ax4.axvspan(12, 14, alpha=0.15, color=ACCENT3, zorder=1)
ax4.axvspan(17, 19, alpha=0.10, color=ACCENT3, zorder=1)

ax4.axvline(x=14, color=ACCENT3, linestyle=":", alpha=0.7, lw=1.5)
ax4.axvline(x=19, color=ACCENT3, linestyle=":", alpha=0.5, lw=1.5)

ax4.annotate("Loss mulai stagnan\n-> LR dipangkas 50%\n(3e-4 -> 1.5e-4)",
             xy=(14, val_loss[14]),
             xytext=(5, 0.15), fontsize=11, color=ACCENT3, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=2),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a40",
                       edgecolor=ACCENT3, linewidth=1))

ax4.annotate("Stagnan lagi\n-> LR dipangkas lagi\n(1.5e-4 -> 7.5e-5)",
             xy=(19, val_loss[19]),
             xytext=(20, 0.38), fontsize=10, color=ACCENT3,
             arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=1.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a40",
                       edgecolor=ACCENT3, linewidth=1, alpha=0.9))

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_2.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
           facecolor="#1a1a40", labelcolor="white", fontsize=12, edgecolor="#333366")
ax4.set_title("Dynamic Learning Rate (ReduceLROnPlateau)", color="white",
              fontweight="bold", fontsize=16, pad=15)
ax4.grid(True, alpha=0.08)

# Insight text
ax4.text(0.5, -0.15, "Callback: Jika val_loss stagnan 3 epoch berturut-turut -> LR x 0.5",
         transform=ax4.transAxes, ha="center", fontsize=12, color=ACCENT3, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a40",
                   edgecolor=ACCENT3, linewidth=1.2))

plt.tight_layout()
fig4.savefig(f"{SAVE_DIR}/ppt_hpo_4_learning_rate.png", dpi=250,
             bbox_inches="tight", facecolor=BG)
plt.close(fig4)
print("4/4 saved: ppt_hpo_4_learning_rate.png")

print("\nDone! All 4 images saved separately.")
