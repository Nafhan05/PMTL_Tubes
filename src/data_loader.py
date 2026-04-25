"""
data_loader.py — Re-export modul dari 1_data_generator.py.

Python tidak mengizinkan import langsung dari modul yang namanya dimulai dengan angka.
File ini berfungsi sebagai bridge/proxy agar modul lain bisa melakukan:

    from src.data_loader import JammingDataGenerator, ...
"""

import importlib
import os
import sys

# Tambahkan src directory ke path
_src_dir = os.path.dirname(__file__)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Import dari 1_data_generator menggunakan importlib
_module = importlib.import_module("1_data_generator")

# Re-export semua yang dibutuhkan
inject_cw_jamming = _module.inject_cw_jamming
inject_barrage_jamming = _module.inject_barrage_jamming
JammingDataGenerator = _module.JammingDataGenerator
create_train_val_test_split = _module.create_train_val_test_split
HDF5_FILE = _module.HDF5_FILE
NUM_CLASSES = _module.NUM_CLASSES
LABEL_NORMAL = _module.LABEL_NORMAL
LABEL_CW = _module.LABEL_CW
LABEL_BARRAGE = _module.LABEL_BARRAGE
