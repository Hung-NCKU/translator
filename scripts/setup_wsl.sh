#!/usr/bin/env bash
# 在 WSL 內建立本專案的獨立執行環境。
# 所有東西都裝在專案資料夾內，不需要 sudo，也不會動到系統。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
BIN="$ROOT/bin"
PY="$VENV/bin/python"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

log "專案根目錄: $ROOT"

# ---------------------------------------------------------------- ffmpeg
if [ ! -x "$BIN/ffmpeg" ]; then
    log "下載可攜式 ffmpeg (static build)"
    mkdir -p "$BIN"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    url="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    curl -fL --retry 3 -o "$tmp/ffmpeg.tar.xz" "$url"
    tar -xJf "$tmp/ffmpeg.tar.xz" -C "$tmp"
    src="$(find "$tmp" -maxdepth 1 -type d -name 'ffmpeg-*-static' | head -1)"
    install -m755 "$src/ffmpeg" "$src/ffprobe" "$BIN/"
    trap - EXIT; rm -rf "$tmp"
else
    log "ffmpeg 已存在，略過"
fi
"$BIN/ffmpeg" -version | head -1

# ---------------------------------------------------------------- venv
if [ ! -x "$PY" ]; then
    log "建立 Python 虛擬環境 (.venv)"
    python3 -m venv "$VENV"
fi
"$PY" -m pip install --quiet --upgrade pip wheel

# ---------------------------------------------------------------- 套件
log "安裝 Python 套件（第一次會下載約 1~2 GB，請耐心等候）"
"$PY" -m pip install \
    "faster-whisper>=1.1.0" \
    "ctranslate2>=4.5.0" \
    "nvidia-cublas-cu12" \
    "nvidia-cudnn-cu12>=9.0" \
    "transformers>=4.40" \
    "sentencepiece" \
    "jinja2" \
    "opencc-python-reimplemented" \
    "deep-translator>=1.11.4" \
    "srt>=3.5.3" \
    "tqdm"

log "完成"
"$PY" - <<'EOF'
import faster_whisper, ctranslate2
print("faster-whisper", faster_whisper.__version__)
print("ctranslate2   ", ctranslate2.__version__)
print("CUDA devices  ", ctranslate2.get_cuda_device_count())
EOF
