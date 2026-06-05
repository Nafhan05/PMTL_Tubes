#!/bin/bash
# run_gpu.sh — Aktivasi venv + set NVIDIA LD_LIBRARY_PATH untuk GPU support di Linux (Ubuntu)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv_py310"
NVIDIA_BASE="$VENV_DIR/lib/python3.10/site-packages/nvidia"

export LD_LIBRARY_PATH="$NVIDIA_BASE/cuda_runtime/lib:$NVIDIA_BASE/cudnn/lib:$NVIDIA_BASE/cublas/lib:$NVIDIA_BASE/cusolver/lib:$NVIDIA_BASE/cusparse/lib:$NVIDIA_BASE/cufft/lib:$NVIDIA_BASE/curand/lib:$NVIDIA_BASE/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

source "$VENV_DIR/bin/activate"

echo "[GPU] NVIDIA LD_LIBRARY_PATH loaded. GPU should be available."
echo ""

exec "$@"
