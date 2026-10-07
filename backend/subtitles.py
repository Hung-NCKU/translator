"""把辨識/翻譯結果排版成 SRT 檔。"""
from __future__ import annotations

import math
from pathlib import Path

from .transcribe import Cue

# 每行最多幾個「字寬」，中日韓字算 2、其他算 1（36 約等於 18 個中文字）
_MAX_WIDTH = 36
_MIN_DURATION = 0.8
_PUNCT = "，。、！？；：,.!?;:"


def _width(text: str) -> int:
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _choose_cut(text: str, target: float, max_width: int) -> int:
    """挑一個斷行位置：盡量接近理想長度，但在標點或空白處斷更好看。"""
    best, best_score, acc = 0, -1e9, 0
    for i, ch in enumerate(text):
        acc += 2 if ord(ch) > 0x2E80 else 1
        if acc > max_width:
            break
        following = text[i + 1] if i + 1 < len(text) else ""
        # 中日韓沒有詞距，任意字之間都能斷；拉丁文只能在空白或標點後斷
        breakable = (ch in _PUNCT or ch == " " or ord(ch) > 0x2E80
                     or (following and ord(following) > 0x2E80))
        if not breakable:
            continue
        score = -abs(acc - target)
        if ch in _PUNCT:
            score += max_width * 0.35
        elif ch == " ":
            score += max_width * 0.10
        if score > best_score:
            best, best_score = i + 1, score
    return best or max(1, len(text) // 2)


def _wrap(text: str, max_width: int = _MAX_WIDTH) -> str:
    """把過長的字幕折成幾行長度相近的短行。"""
    text = " ".join(text.split())
    total = _width(text)
    if total <= max_width:
        return text

    # 先算需要幾行，讓每行都朝「平均長度」靠攏，避免第一行爆滿、第二行只有幾個字
    target = total / math.ceil(total / max_width)
    lines, rest = [], text
    while _width(rest) > max_width:
        cut = _choose_cut(rest, target, max_width)
        lines.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest.strip():
        lines.append(rest.strip())
    return "\n".join(line for line in lines if line)


def _timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _tidy(cues: list[Cue]) -> list[Cue]:
    """補足過短的顯示時間，並消除時間軸重疊。"""
    out: list[Cue] = []
    for cue in cues:
        end = max(cue.end, cue.start + _MIN_DURATION)
        if out and cue.start < out[-1].end:
            out[-1].end = max(out[-1].start + 0.3, cue.start - 0.02)
        out.append(Cue(cue.start, end, cue.text, cue.translation))
    return out


def render(cues: list[Cue], layout: str = "target") -> str:
    """layout: target=只有譯文, source=只有原文, bilingual=譯文在上原文在下。"""
    blocks = []
    for index, cue in enumerate(_tidy(cues), 1):
        translated = cue.translation or cue.text
        if layout == "source" or not cue.translation:
            body = _wrap(cue.text)
        elif layout == "bilingual":
            body = f"{_wrap(translated)}\n{_wrap(cue.text)}"
        else:
            body = _wrap(translated)
        blocks.append(f"{index}\n{_timestamp(cue.start)} --> "
                      f"{_timestamp(cue.end)}\n{body}\n")
    return "\n".join(blocks)


def write(cues: list[Cue], path: Path, layout: str = "target") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(cues, layout), encoding="utf-8")
    return path
