"""集中管理路徑、語言清單與預設值。"""
from __future__ import annotations

import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
LOGS = ROOT / "logs"
FONTS = BIN / "fonts"

FFMPEG = BIN / "ffmpeg"
FFPROBE = BIN / "ffprobe"

# 模型與 HuggingFace 快取一律留在專案資料夾內
os.environ.setdefault("HF_HOME", str(MODELS))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS / "hub"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODELS / "cache"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# transformers 只拿來做斷詞，不需要 PyTorch，也不用它報一堆無關的提醒
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def quiet_hub_warnings() -> None:
    """關掉「沒帶 HF_TOKEN」那句警告。

    下載模型本來就不需要登入，那只是速率限制的提醒，卻會洗版執行紀錄。
    huggingface_hub 會在自己被匯入時重設 logger 等級，所以這件事一定要等到
    匯入之後才做，在這個模組的頂層設是沒有用的。
    使用者自己設了 HF_TOKEN 的話就不動它。
    """
    if os.environ.get("HF_TOKEN"):
        return
    for name in ("huggingface_hub", "huggingface_hub.utils._http"):
        logging.getLogger(name).setLevel(logging.ERROR)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm", ".m4v",
              ".mpg", ".mpeg", ".ts", ".m2ts", ".vob", ".3gp", ".mts"}

# Whisper 模型（顯示名稱 -> 模型 ID）
WHISPER_MODELS = {
    "tiny（最快 / 品質最低）": "tiny",
    "base（很快）": "base",
    "small（平衡）": "small",
    "medium（較準）": "medium",
    "large-v3（最準 / 建議）": "large-v3",
    "large-v3-turbo（快且準 / 推薦）": "deepdml/faster-whisper-large-v3-turbo-ct2",
}

# 目標語言：顯示名稱 -> (Google 代碼, 給 LLM 看的名稱, Whisper 代碼)
LANGUAGES = {
    "繁體中文": ("zh-TW", "Traditional Chinese (Taiwan)", "zh"),
    "簡體中文": ("zh-CN", "Simplified Chinese", "zh"),
    "英文": ("en", "English", "en"),
    "日文": ("ja", "Japanese", "ja"),
    "韓文": ("ko", "Korean", "ko"),
    "西班牙文": ("es", "Spanish", "es"),
    "法文": ("fr", "French", "fr"),
    "德文": ("de", "German", "de"),
    "越南文": ("vi", "Vietnamese", "vi"),
    "泰文": ("th", "Thai", "th"),
    "印尼文": ("id", "Indonesian", "id"),
    "俄文": ("ru", "Russian", "ru"),
    "葡萄牙文": ("pt", "Portuguese", "pt"),
    "義大利文": ("it", "Italian", "it"),
    "阿拉伯文": ("ar", "Arabic", "ar"),
    "印地文": ("hi", "Hindi", "hi"),
}

# 來源語言（影片原本的語言），auto 代表自動偵測
SOURCE_LANGUAGES = {"自動偵測": "auto"} | {
    name: code for name, (_g, _n, code) in LANGUAGES.items()
}

DEFAULTS = {
    "model": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "source_lang": "auto",
    "target_lang": "繁體中文",
    "translator": "google",
    "subtitle_mode": "soft",     # srt | soft | hard
    "layout": "bilingual",       # target | bilingual
    "font_size": 20,
}

for d in (MODELS, OUTPUT, LOGS, FONTS):
    d.mkdir(parents=True, exist_ok=True)
