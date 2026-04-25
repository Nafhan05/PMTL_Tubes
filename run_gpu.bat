@echo off
REM Aktivasi venv + set NVIDIA DLL PATH untuk GPU support
REM Usage: run_gpu.bat python src/2_train_1dcnn.py --dry-run

set VENV_DIR=%~dp0.venv
set NVIDIA_BASE=%VENV_DIR%\Lib\site-packages\nvidia

set PATH=%NVIDIA_BASE%\cuda_runtime\bin;%NVIDIA_BASE%\cudnn\bin;%NVIDIA_BASE%\cublas\bin;%NVIDIA_BASE%\cusolver\bin;%NVIDIA_BASE%\cusparse\bin;%NVIDIA_BASE%\cufft\bin;%NVIDIA_BASE%\curand\bin;%NVIDIA_BASE%\cuda_nvrtc\bin;%PATH%

call %VENV_DIR%\Scripts\activate.bat

echo [GPU] NVIDIA DLL paths loaded. GPU should be available.
echo.

%*
