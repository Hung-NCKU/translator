"""ffmpeg / ffprobe 的封裝：探測、抽音軌、掛字幕、燒字幕。"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable

from .config import FFMPEG, FFPROBE, FONTS

# Windows 內建的中日韓字型，依偏好排序
_WINDOWS_FONT_CANDIDATES = [
    ("msjh.ttc", "Microsoft JhengHei"),
    ("msyh.ttc", "Microsoft YaHei"),
    ("meiryo.ttc", "Meiryo"),
    ("malgun.ttf", "Malgun Gothic"),
    ("arial.ttf", "Arial"),
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


def probe(path: Path) -> dict:
    """回傳影片的時長、有無音軌等資訊。"""
    cp = run([FFPROBE, "-v", "error", "-print_format", "json",
              "-show_format", "-show_streams", str(path)])
    if cp.returncode != 0:
        raise RuntimeError(f"ffprobe 讀取失敗：{cp.stderr.strip()[:500]}")
    data = json.loads(cp.stdout or "{}")
    streams = data.get("streams", [])
    return {
        "duration": float(data.get("format", {}).get("duration") or 0.0),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "has_video": any(s.get("codec_type") == "video" for s in streams),
    }


def extract_audio(src: Path, dst: Path) -> Path:
    """抽成 16kHz 單聲道 wav，這是 Whisper 要的格式。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cp = run([FFMPEG, "-y", "-v", "error", "-i", str(src),
              "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000",
              "-c:a", "pcm_s16le", str(dst)])
    if cp.returncode != 0:
        raise RuntimeError(f"抽取音訊失敗：{cp.stderr.strip()[:500]}")
    return dst


def ensure_fonts() -> str:
    """把 Windows 的中文字型複製進專案，讓 WSL 裡的 libass 找得到。

    回傳可用的字型名稱（給 libass 的 force_style 用）。
    """
    winfonts = Path("/mnt/c/Windows/Fonts")
    chosen = "Arial"
    for filename, family in _WINDOWS_FONT_CANDIDATES:
        target = FONTS / filename
        source = winfonts / filename
        if target.exists():
            chosen = family
            break
        if source.exists():
            try:
                shutil.copy2(source, target)
                chosen = family
                break
            except OSError:
                continue
    return chosen


def _has_encoder(name: str) -> bool:
    cp = run([FFMPEG, "-hide_banner", "-encoders"])
    return name in (cp.stdout or "")


_ENCODER_CACHE: dict[str, bool] = {}


def pick_video_encoder() -> list[str]:
    """能用 GPU 就用 GPU 編碼，快非常多。"""
    if "nvenc" not in _ENCODER_CACHE:
        _ENCODER_CACHE["nvenc"] = _has_encoder("h264_nvenc")
    if _ENCODER_CACHE["nvenc"]:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]


def _run_with_progress(cmd: list[str], total_sec: float,
                       on_progress: Callable[[float], None] | None,
                       cwd: Path | None = None) -> None:
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            cwd=str(cwd) if cwd else None)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        m = re.match(r"out_time_ms=(\d+)", line.strip())
        if m and total_sec > 0 and on_progress:
            on_progress(min(1.0, int(m.group(1)) / 1_000_000 / total_sec))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg 失敗：\n" + "".join(tail)[-1500:])


def mux_soft_subtitles(src: Path, subs: list[tuple[Path, str, str]], dst: Path,
                       on_progress: Callable[[float], None] | None = None,
                       duration: float = 0.0) -> Path:
    """把字幕以「軟字幕」方式包進影片，不重新編碼影像，很快。

    subs: [(srt 路徑, 語言代碼, 軌道標題), ...]
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-hide_banner", "-v", "error", "-progress", "pipe:1",
           "-i", str(src)]
    for srt_path, _lang, _title in subs:
        cmd += ["-i", str(srt_path)]
    cmd += ["-map", "0"]
    for i in range(len(subs)):
        cmd += ["-map", str(i + 1)]
    sub_codec = "mov_text" if dst.suffix.lower() in {".mp4", ".m4v", ".mov"} else "srt"
    cmd += ["-c", "copy", "-c:s", sub_codec]
    for i, (_p, lang, title) in enumerate(subs):
        cmd += [f"-metadata:s:s:{i}", f"language={lang}",
                f"-metadata:s:s:{i}", f"title={title}"]
    cmd += ["-disposition:s:0", "default", str(dst)]
    _run_with_progress(cmd, duration, on_progress)
    return dst


def burn_subtitles(src: Path, srt_path: Path, dst: Path, font_name: str,
                   font_size: int = 20,
                   on_progress: Callable[[float], None] | None = None,
                   duration: float = 0.0) -> Path:
    """把字幕燒進畫面，任何播放器都看得到，但需要重新編碼。

    subtitles 濾鏡對路徑裡的冒號、引號、逗號很敏感，跳脫規則又依 ffmpeg 版本
    而異。所以這裡不跳脫，而是要求 srt_path 位於一個乾淨的暫存目錄，
    把工作目錄切過去後只用相對檔名，專案放在含空白或中文的路徑也不會出錯。
    """
    src, dst = src.resolve(), dst.resolve()   # 等下會切工作目錄，路徑必須是絕對的
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = srt_path.parent
    fonts_dir = work / "fonts"
    if not fonts_dir.exists():
        try:
            os.symlink(FONTS, fonts_dir)
        except OSError:
            shutil.copytree(FONTS, fonts_dir)

    style = (f"FontName={font_name},FontSize={font_size},"
             "PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,BorderStyle=1,"
             "Outline=2,Shadow=0,MarginV=28,Alignment=2")
    vf = (f"subtitles={srt_path.name}:fontsdir=fonts:force_style='{style}'")
    cmd = [FFMPEG, "-y", "-hide_banner", "-v", "error", "-progress", "pipe:1",
           "-i", str(src), "-vf", vf, *pick_video_encoder(),
           "-c:a", "copy", "-movflags", "+faststart", str(dst)]
    _run_with_progress(cmd, duration, on_progress, cwd=work)
    return dst
