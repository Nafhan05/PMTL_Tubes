"""
Generate updated PPT visualizations:
1. ppt_final_summary.png — with Precision & Recall columns, 2D-CNN baseline row
2. ppt_computational_cost.png — parameter count, inference latency, training epochs
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


# ============================================================
# 1. Updated Final Summary Table
# ============================================================
def plot_final_summary_table():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis('off')

    col_labels = ["Rank", "Model", "Accuracy", "Precision", "Recall", "F1-Score", "vs Baseline", "Status"]
    row_data = [
        ["#1", "LSTM",               "97.74%", "97.77%", "97.76%", "0.977", "- (tanpa HPO)", "Terbaik"],
        ["#2", "1D-CNN (HPO)",       "96.31%", "96.51%", "96.35%", "0.963", "+20.8%",        "Sukses HPO"],
        ["#3", "2D-CNN (Baseline)",  "83.27%", "88.74%", "83.01%", "0.822", "- (sebelum HPO)","Saturasi"],
        ["#4", "2D-CNN (HPO)",       "83.36%", "88.83%", "83.10%", "0.823", "+0.1% (tetap)", "Bottleneck input"],
        ["#5", "1D-CNN (Baseline)",  "75.49%", "85.59%", "75.90%", "0.756", "- (sebelum HPO)","Overfitting"],
    ]

    # Color coding per row
    colors = [
        ["#FFF9C4", "#FFF9C4", "#C8E6C9", "#C8E6C9", "#C8E6C9", "#C8E6C9", "#F5F5F5", "#C8E6C9"],  # LSTM gold
        ["#E3F2FD", "#E3F2FD", "#C8E6C9", "#C8E6C9", "#C8E6C9", "#C8E6C9", "#C8E6C9", "#C8E6C9"],  # 1D HPO blue
        ["#FFF3E0", "#FFF3E0", "#FFECB3", "#FFECB3", "#FFECB3", "#FFECB3", "#F5F5F5", "#FFECB3"],  # 2D baseline
        ["#FFF3E0", "#FFF3E0", "#FFECB3", "#FFECB3", "#FFECB3", "#FFECB3", "#FFCDD2", "#FFCDD2"],  # 2D HPO
        ["#F5F5F5", "#F5F5F5", "#FFCDD2", "#FFCDD2", "#FFCDD2", "#FFCDD2", "#F5F5F5", "#FFCDD2"],  # 1D baseline red
    ]

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellColours=colors,
        colColours=["#1565C0"] * 8,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for j in range(8):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')

    plt.title("Ringkasan Performa Akhir - Semua Model", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "ppt_final_summary.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 2. Computational Cost Table
# ============================================================
def plot_computational_cost_table():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    col_labels = ["Model", "Parameters", "Trainable", "Training\nEpochs", "Inference\nLatency (ms)", "Accuracy"]
    row_data = [
        ["LSTM",               "120,835",  "120,835",  "21",  "61.5 ms",  "97.74%"],
        ["1D-CNN (Baseline)",  "150,851",  "149,955",  "6*",  "36.9 ms",  "75.49%"],
        ["1D-CNN (HPO)",       "871,139",  "869,667",  "30",  "61.8 ms",  "96.31%"],
        ["2D-CNN (Baseline)",  "102,019",  "101,571",  "30",  "34.3 ms",  "83.27%"],
        ["2D-CNN (HPO)",       "164,579",  "164,195",  "40",  "66.4 ms",  "83.36%"],
    ]

    # Color code: green = good, yellow = medium, red = bad
    colors = [
        ["#E3F2FD", "#C8E6C9", "#C8E6C9", "#C8E6C9", "#FFECB3", "#C8E6C9"],  # LSTM
        ["#E3F2FD", "#C8E6C9", "#C8E6C9", "#FFCDD2", "#C8E6C9", "#FFCDD2"],  # 1D base
        ["#E3F2FD", "#FFECB3", "#FFECB3", "#FFECB3", "#FFECB3", "#C8E6C9"],  # 1D HPO
        ["#E3F2FD", "#C8E6C9", "#C8E6C9", "#FFECB3", "#C8E6C9", "#FFECB3"],  # 2D base
        ["#E3F2FD", "#C8E6C9", "#C8E6C9", "#FFECB3", "#FFECB3", "#FFCDD2"],  # 2D HPO
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
    table.scale(1, 1.8)

    for j in range(6):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')

    plt.title("Computational Cost - Perbandingan Semua Model", fontsize=14, fontweight='bold', pad=20)

    # Footnote
    fig.text(0.12, 0.02, "* 1D-CNN Baseline: early stopping pada epoch 6 karena overfitting parah (val_loss meningkat terus)",
             fontsize=9, style='italic', color='#666')

    save_path = os.path.join(OUT_DIR, "ppt_computational_cost.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 3. Computational Cost Bar Chart (visual complement)
# ============================================================
def plot_computational_cost_bars():
    models = ["LSTM", "1D-CNN\nBaseline", "1D-CNN\nHPO", "2D-CNN\nBaseline", "2D-CNN\nHPO"]
    params = [120835, 150851, 871139, 102019, 164579]
    latency = [61.5, 36.9, 61.8, 34.3, 66.4]
    accuracy = [97.74, 75.49, 96.31, 83.27, 83.36]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Bar 1: Parameter Count
    bar_colors = ["#4CAF50", "#FF9800", "#2196F3", "#FF9800", "#f44336"]
    bars = axes[0].bar(models, [p/1000 for p in params], color=bar_colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title("Parameter Count (K)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Thousands")
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, params):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{val/1000:.0f}K', ha='center', va='bottom', fontsize=9)

    # Bar 2: Inference Latency
    bars = axes[1].bar(models, latency, color=bar_colors, edgecolor='white', linewidth=0.5)
    axes[1].set_title("Inference Latency (ms)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Milliseconds")
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, latency):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Bar 3: Accuracy
    bars = axes[2].bar(models, accuracy, color=bar_colors, edgecolor='white', linewidth=0.5)
    axes[2].set_title("Accuracy (%)", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_ylim(70, 102)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, accuracy):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.suptitle("Computational Cost vs Performance", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "ppt_computational_cost_bars.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
if __name__ == "__main__":
    print("Generating updated visualizations...\n")
    plot_final_summary_table()
    plot_computational_cost_table()
    plot_computational_cost_bars()
    print("\nDone!")
