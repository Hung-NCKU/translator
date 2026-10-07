"""用 faster-whisper 把影片語音轉成帶時間軸的逐句文字。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import MODELS, quiet_hub_warnings

_CONVERTER = None


def to_traditional(text: str) -> str:
    """Whisper 的中文輸出幾乎都是簡體，台灣使用者要的是繁體。

    用 OpenCC 的 s2twp：除了字形，連詞彙都會換成台灣說法（网络→網路）。
    本來就是繁體的句子丟進去不會被改壞。
    """
    global _CONVERTER
    if _CONVERTER is None:
        import opencc
        _CONVERTER = opencc.OpenCC("s2twp")
    return _CONVERTER.convert(text)


@dataclass
class Cue:
    """一句字幕：起訖秒數與文字。"""
    start: float
    end: float
    text: str
    translation: str = ""


_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


def _load_model(name: str, device: str, compute_type: str):
    key = (name, device, compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel
        quiet_hub_warnings()      # 必須在 huggingface_hub 被匯入之後才有效
        _MODEL_CACHE[key] = WhisperModel(
            name, device=device, compute_type=compute_type,
            download_root=str(MODELS / "whisper"),
        )
    return _MODEL_CACHE[key]


def pick_device() -> tuple[str, str]:
    """有 GPU 就用 GPU（float16），否則退回 CPU（int8）。"""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def transcribe(audio: Path, model_name: str, source_lang: str = "auto",
               duration: float = 0.0,
               on_progress: Callable[[float, str], None] | None = None,
               device: str | None = None,
               traditional_chinese: bool = True) -> tuple[list[Cue], str]:
    """回傳 (字幕清單, 偵測到的語言代碼)。"""
    dev, compute = (device, "float16" if device == "cuda" else "int8") if device \
        else pick_device()
    try:
        model = _load_model(model_name, dev, compute)
    except Exception:
        if dev == "cpu":
            raise
        # GPU 起不來（驅動/cuDNN 問題）就自動降級到 CPU，不要整個失敗
        dev, compute = "cpu", "int8"
        model = _load_model(model_name, dev, compute)

    segments, info = model.transcribe(
        str(audio),
        language=None if source_lang == "auto" else source_lang,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        # 關掉「參考前文」可以大幅減少日語等語言的幻聽式重複字幕
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        # 先用一句繁體提示引導，模型比較不會一路吐簡體
        initial_prompt="以下是繁體中文的內容。" if source_lang == "zh" else None,
    )

    total = duration or float(getattr(info, "duration", 0.0)) or 0.0
    cues: list[Cue] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            cues.append(Cue(start=seg.start, end=seg.end, text=text))
        if on_progress and total > 0:
            on_progress(min(1.0, seg.end / total),
                        f"辨識中（{dev.upper()}）：{len(cues)} 句")

    detected = info.language or source_lang
    if detected == "zh" and traditional_chinese:
        for cue in cues:
            cue.text = to_traditional(cue.text)
    return cues, detected
