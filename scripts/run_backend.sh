#!/usr/bin/env bash
# 設好 CUDA 函式庫路徑後執行後端。GUI 一律透過這支腳本呼叫 WSL。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
SITE="$(echo "$VENV"/lib/python3*/site-packages)"

# faster-whisper 的 GPU 後端需要 pip 裝進來的 cuDNN / cuBLAS
export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ROOT/bin:$PATH"
export HF_HOME="$ROOT/models"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd "$ROOT"
exec "$VENV/bin/python" -u -m backend.cli "$@"
