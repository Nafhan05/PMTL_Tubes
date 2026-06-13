"""
Visualisasi untuk slide PPT final — HPO Tables & Comparison Charts
Menghasilkan gambar-gambar tabel yang siap dimasukkan ke PowerPoint.
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import json

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "hpo")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Tabel Perbandingan Grid vs Random vs Bayesian
# ============================================================
def plot_hpo_method_comparison():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    col_labels = ["Aspek", "Grid Search", "Random Search", "Bayesian Optimization ✓"]
    row_data = [
        ["Cara kerja", "Coba SEMUA\nkombinasi", "Coba kombinasi\nacak", "Belajar dari\npercobaan sebelumnya"],
        ["Efisiensi", "Sangat lambat\n(ribuan trial)", "Sedang\n(untung-untungan)", "Cerdas & efisien\n(30 trial cukup)"],
        ["Kualitas hasil", "Pasti optimal\n(tapi mahal)", "Bisa miss\nkonfigurasi terbaik", "Mendekati optimal\ndengan cepat"],
        ["Cocok untuk", "Search space\nkecil (<4 HP)", "Eksplorasi awal\n(banyak HP)", "Search space besar\n(14 HP, seperti proyek ini)"],
    ]

    colors = [["#E3F2FD"] * 4 for _ in range(4)]
    for i in range(4):
        colors[i][3] = "#C8E6C9"  # Bayesian column green

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellColours=colors,
        colColours=["#1565C0", "#BBDEFB", "#BBDEFB", "#66BB6A"],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    # Style header
    for j in range(4):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white' if j in [0, 3] else 'black')

    plt.title("Perbandingan Metode Hyperparameter Optimization", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "ppt_hpo_method_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 2. Tabel Search Space & Best HP — 1D-CNN
# ============================================================
def plot_search_space_1dcnn():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')

    col_labels = ["Hyperparameter", "Search Range", "Baseline", "Best (HPO)"]
    row_data = [
        ["Jumlah blok konvolusi", "2 – 4", "4", "4"],
        ["Ukuran kernel", "3, 5, 7", "7, 5, 3, 3", "7 (semua)"],
        ["Filter per blok", "32 – 256", "64→128→128→128", "192→256→256→32"],
        ["Spatial Dropout", "0.05 – 0.4", "0.1 – 0.2", "0.05 (semua)"],
        ["Dense units", "32, 64, 128", "64", "128"],
        ["Dense Dropout", "0.3 – 0.6", "0.5", "0.3"],
        ["L2 regularization", "1e-5, 1e-4, 1e-3", "1e-4", "1e-5"],
        ["Learning rate", "0.0001 – 0.01", "0.0003", "0.00183"],
    ]

    colors = []
    for i, row in enumerate(row_data):
        row_colors = ["#E3F2FD", "#F5F5F5", "#FFF3E0", "#C8E6C9"]
        colors.append(row_colors)

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellColours=colors,
        colColours=["#1565C0", "#757575", "#E65100", "#2E7D32"],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for j in range(4):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')

    plt.title("1D-CNN — Search Space & Best Hyperparameters", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "ppt_search_space_1dcnn.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 3. Tabel Search Space & Best HP — 2D-CNN
# ============================================================
def plot_search_space_2dcnn():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')

    col_labels = ["Hyperparameter", "Search Range", "Baseline", "Best (HPO)"]
    row_data = [
        ["Jumlah blok konvolusi", "3 – 5", "3", "5"],
        ["Ukuran kernel", "3, 5", "3", "5"],
        ["Filter per blok", "16 – 128", "32→64→128", "32→64→32→32→32"],
        ["Use L2 reg.", "True / False", "True (1e-4)", "False"],
        ["Dense units", "64, 128, 256", "64", "256"],
        ["Dense Dropout", "0.3 – 0.6", "0.5", "0.5"],
        ["Learning rate", "0.0001 – 0.01", "0.0003", "0.00881"],
        ["STFT nperseg", "32, 64, 128", "64", "64"],
    ]

    colors = []
    for i, row in enumerate(row_data):
        row_colors = ["#E3F2FD", "#F5F5F5", "#FFF3E0", "#FFCDD2"]
        colors.append(row_colors)

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellColours=colors,
        colColours=["#1565C0", "#757575", "#E65100", "#C62828"],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for j in range(4):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')

    plt.title("2D-CNN — Search Space & Best Hyperparameters", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "ppt_search_space_2dcnn.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 4. Bar Chart Perbandingan Baseline vs HPO (semua model)
# ============================================================
def plot_comparison_bar():
    # Load baseline results
    with open(os.path.join(BASE_DIR, "results", "evaluation_results.json")) as f:
        baseline_data = json.load(f)
    with open(os.path.join(BASE_DIR, "results", "evaluation_results_2dcnn.json")) as f:
        baseline_2d = json.load(f)
    with open(os.path.join(OUT_DIR, "evaluation_results_all.json")) as f:
        hpo_data = json.load(f)

    models = ["LSTM", "1D-CNN\n(Baseline)", "1D-CNN\n(HPO)", "2D-CNN\n(Baseline)", "2D-CNN\n(HPO)"]

    # Gather all 4 metrics
    sources = [
        baseline_data["LSTM"],
        baseline_data["1D-CNN"],
        hpo_data["1D-CNN HPO"],
        baseline_2d["2D-CNN"],
        hpo_data["2D-CNN HPO"],
    ]
    accuracy  = [s["accuracy"] * 100 for s in sources]
    precision = [s["precision_macro"] * 100 for s in sources]
    recall    = [s["recall_macro"] * 100 for s in sources]
    f1        = [s["f1_macro"] * 100 for s in sources]

    x = np.arange(len(models))
    n_metrics = 4
    width = 0.18
    offsets = [-(1.5*width), -(0.5*width), (0.5*width), (1.5*width)]

    metric_data   = [accuracy, precision, recall, f1]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, (data, label, color, off) in enumerate(zip(metric_data, metric_labels, metric_colors, offsets)):
        bars = ax.bar(x + off, data, width, label=label, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Perbandingan Performa - Baseline vs HPO (Semua Metrik)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(70, 103)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "ppt_comparison_bar.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 5. Tabel Ringkasan Hasil Akhir (semua model)
# ============================================================
def plot_final_summary_table():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis('off')

    col_labels = ["Rank", "Model", "Accuracy", "F1-Score", "Δ dari Baseline", "Status"]
    row_data = [
        ["#1", "LSTM", "97.74%", "0.977", "- (tidak di-HPO)", "TERBAIK"],
        ["#2", "1D-CNN (HPO)", "96.31%", "0.963", "+20.8%", "SUKSES HPO"],
        ["#3", "2D-CNN (HPO)", "83.36%", "0.823", "+0.1% (tetap)", "Bottleneck data"],
        ["#4", "1D-CNN (Baseline)", "75.49%", "0.756", "- (sebelum HPO)", "Overfitting"],
    ]

    colors = [
        ["#FFF9C4", "#FFF9C4", "#C8E6C9", "#C8E6C9", "#F5F5F5", "#C8E6C9"],
        ["#E3F2FD", "#E3F2FD", "#C8E6C9", "#C8E6C9", "#C8E6C9", "#C8E6C9"],
        ["#FFF3E0", "#FFF3E0", "#FFECB3", "#FFECB3", "#FFCDD2", "#FFECB3"],
        ["#F5F5F5", "#F5F5F5", "#FFCDD2", "#FFCDD2", "#F5F5F5", "#FFCDD2"],
    ]

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellColours=colors,
        colColours=["#1565C0"] * 6,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.9)

    for j in range(6):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')

    plt.title("Ringkasan Performa Akhir — Semua Model", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "ppt_final_summary.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 6. Diagram alur Bayesian Optimization
# ============================================================
def plot_bayesian_flow():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')

    steps = [
        ("1. Inisialisasi\nRandom", "Coba 5 konfigurasi\nacak sebagai titik awal", "#42A5F5"),
        ("2. Surrogate\nModel", "Bangun model prediksi\n(Gaussian Process)", "#66BB6A"),
        ("3. Acquisition\nFunction", "Pilih konfigurasi\npaling menjanjikan", "#FFA726"),
        ("4. Evaluasi\nModel", "Training dengan HP\nterpilih, ukur val_acc", "#EF5350"),
        ("5. Update &\nUlangi", "Perbarui surrogate,\nulangi hingga 30 trial", "#AB47BC"),
    ]

    for i, (title, desc, color) in enumerate(steps):
        x = 0.08 + i * 0.19
        # Box
        rect = plt.Rectangle((x, 0.25), 0.15, 0.55, facecolor=color, alpha=0.85,
                             edgecolor='white', linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.075, 0.62, title, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(x + 0.075, 0.40, desc, ha='center', va='center', fontsize=8.5,
                color='white', transform=ax.transAxes)
        # Arrow
        if i < 4:
            ax.annotate('', xy=(x + 0.175, 0.52), xytext=(x + 0.15, 0.52),
                       arrowprops=dict(arrowstyle='->', color='#333', lw=2),
                       transform=ax.transAxes)

    plt.title("Alur Bayesian Optimization (Keras Tuner)", fontsize=14, fontweight='bold', pad=10)
    save_path = os.path.join(OUT_DIR, "ppt_bayesian_flow.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
if __name__ == "__main__":
    print("Generating PPT visualizations...\n")
    plot_hpo_method_comparison()
    plot_search_space_1dcnn()
    plot_search_space_2dcnn()
    plot_comparison_bar()
    plot_final_summary_table()
    plot_bayesian_flow()
    print("\nDone! All images saved to results/hpo/")
