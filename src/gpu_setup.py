"""
gpu_setup.py — Otomatis menambahkan path NVIDIA DLL ke PATH agar TF 2.10 mendeteksi GPU.
Import modul ini SEBELUM import tensorflow.

Juga mengatur matplotlib ke backend non-interaktif (Agg) saat dijalankan via script.
"""

import os
import sys
import site


def setup_nvidia_dll_paths():
    """Tambahkan semua folder bin NVIDIA pip packages ke os.environ['PATH'] dan os.add_dll_directory."""
    sp = site.getsitepackages()
    if not sp:
        return

    nvidia_base = os.path.join(sp[0], "nvidia")
    if not os.path.isdir(nvidia_base):
        return

    added = []
    for pkg in os.listdir(nvidia_base):
        bin_dir = os.path.join(nvidia_base, pkg, "bin")
        if os.path.isdir(bin_dir):
            # Tambahkan ke PATH
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            # Gunakan os.add_dll_directory (Python 3.8+, Windows)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(bin_dir)
                except OSError:
                    pass
            added.append(pkg)

    if added:
        print(f"[GPU Setup] Added {len(added)} NVIDIA DLL paths")


def setup_matplotlib_backend():
    """Set matplotlib ke backend non-interaktif agar plt.show() tidak blocking."""
    import matplotlib
    matplotlib.use("Agg")


# Auto-run saat di-import
setup_nvidia_dll_paths()
setup_matplotlib_backend()
