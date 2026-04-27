import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    results_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'evaluation_results.json')
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    class_names = ["Normal", "CW Jamming", "Barrage Jamming"]
    metrics_names = ["Precision", "Recall", "F1-Score"]
    
    for model_name, metrics in data.items():
        # Buat matriks 3x3 (Baris: Kelas, Kolom: Metrik)
        matrix = np.zeros((3, 3))
        matrix[:, 0] = metrics["precision_per_class"]
        matrix[:, 1] = metrics["recall_per_class"]
        matrix[:, 2] = metrics["f1_per_class"]
        
        plt.figure(figsize=(7, 5))
        # Gunakan colormap yang bagus dengan limit vmin=0.9 agar perbedaan kecil terlihat
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".4f",
            cmap="YlGnBu",
            xticklabels=metrics_names,
            yticklabels=class_names,
            vmin=0.90, vmax=1.00
        )
        
        plt.title(f"{model_name} — Classification Report Matrix")
        plt.xlabel("Metrics")
        plt.ylabel("Classes")
        plt.tight_layout()
        
        save_filename = f"classification_report_{model_name.lower().replace('-', '')}.png"
        save_path = os.path.join(save_dir, save_filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Berhasil menyimpan {save_filename}")

if __name__ == "__main__":
    # Karena backend mungkin non-interaktif
    import matplotlib
    matplotlib.use("Agg")
    main()
