"""後端入口：從 stdin 讀一份 JSON 工作單，逐檔處理並把進度以 JSON 逐行輸出。

GUI 只負責蒐集設定與顯示進度，真正的工作都在這裡完成。
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

from . import media, subtitles, translate
from .config import LANGUAGES, OUTPUT
from .transcribe import Cue, transcribe

# 各階段在單一檔案進度條中佔的比例
STAGES = {"probe": (0.00, 0.02), "audio": (0.02, 0.08),
          "asr": (0.08, 0.55), "mt": (0.55, 0.82),
          "srt": (0.82, 0.85), "render": (0.85, 1.00)}


def emit(**payload) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _stage_reporter(stage: str, index: int):
    """回傳一個回報函式，順便依實際速度推估這個階段還要多久。"""
    lo, hi = STAGES[stage]
    started = time.monotonic()

    def report(fraction: float, message: str = "") -> None:
        fraction = max(0.0, min(1.0, fraction))
        eta = None
        elapsed = time.monotonic() - started
        # 太早推估會亂跳，等跑過 5% 且至少兩秒才開始報
        if 0.05 <= fraction < 1.0 and elapsed >= 2.0:
            eta = elapsed * (1.0 - fraction) / fraction
        emit(type="progress", index=index, stage=stage,
             pct=lo + (hi - lo) * fraction, msg=message, eta=eta,
             frac=fraction)
    return report


def _unique_stem(path: Path, taken: set[str]) -> str:
    """所有檔案都輸出到同一個資料夾時，不同來源的同名影片會互相覆蓋。

    例如「第一季/01.mp4」和「第二季/01.mp4」都會想寫成 01.zh-TW.srt。
    這裡遇到撞名就冠上來源資料夾名（第一季_01），再撞才加編號。
    """
    stem = path.stem
    if stem not in taken:
        taken.add(stem)
        return stem

    candidate = f"{path.parent.name}_{stem}"
    serial = 2
    while candidate in taken:
        candidate = f"{path.parent.name}_{stem} ({serial})"
        serial += 1
    taken.add(candidate)
    return candidate


def process_one(path: Path, index: int, total: int, job: dict,
                translator, taken: set[str] | None = None) -> list[str]:
    emit(type="file_start", index=index, total=total, name=path.name)

    custom_dir = bool(job.get("output_dir"))
    out_dir = Path(job["output_dir"]) if custom_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # 各自輸出回原資料夾時不可能撞名（同一層不會有兩個同名檔），不用改名
    stem = _unique_stem(path, taken) if custom_dir and taken is not None \
        else path.stem
    if stem != path.stem:
        emit(type="log", index=index,
             msg=f"輸出資料夾已有同名檔案，這個檔改存成「{stem}」")
    target_lang = job["target_lang"]
    lang_code, _model_name, _w = LANGUAGES[target_lang]
    produced: list[str] = []

    _stage_reporter("probe", index)(0.5, "讀取影片資訊")
    info = media.probe(path)
    if not info["has_audio"]:
        raise RuntimeError("這個檔案沒有音軌，無法辨識語音")
    duration = info["duration"]

    with tempfile.TemporaryDirectory(prefix="subtrans_") as tmpdir:
        tmp = Path(tmpdir)
        _stage_reporter("audio", index)(0.3, "抽取音訊")
        wav = media.extract_audio(path, tmp / "audio.wav")

        cues, detected = transcribe(
            wav, job["model"], job.get("source_lang", "auto"),
            duration=duration, on_progress=_stage_reporter("asr", index),
            # 只有在使用者就是要簡體時才保留簡體
            traditional_chinese=target_lang != "簡體中文")
        emit(type="log", index=index,
             msg=f"辨識完成：{len(cues)} 句，語言 {detected}")
        # 讓介面能在翻譯開始前就估出大概要等多久。
        # 本地模型的時間主要由字數決定（要逐字生成），不是由句數決定
        emit(type="cue_count", index=index, count=len(cues), lang=detected,
             chars=sum(len(c.text) for c in cues))
        if not cues:
            raise RuntimeError("這個檔案沒有辨識到任何語音")

        if translator is not None:
            try:
                translator.translate(cues, target_lang, source_name=detected,
                                     on_progress=_stage_reporter("mt", index))
            except Exception:
                # 語音辨識可能已經跑了很久，別讓翻譯失敗把那份成果一起賠掉。
                # 先把原文字幕寫出來，之後修好問題可以直接拿去翻，不用重跑辨識。
                rescue = out_dir / f"{stem}.{detected}.srt"
                subtitles.write(cues, rescue, "source")
                emit(type="log", index=index,
                     msg=f"翻譯失敗，但已保留辨識好的原文字幕：{rescue.name}")
                raise
        else:
            _stage_reporter("mt", index)(1.0, "略過翻譯")

        layout = job.get("layout", "target")
        _stage_reporter("srt", index)(0.5, "產生字幕檔")
        srt_path = out_dir / f"{stem}.{lang_code}.srt"
        subtitles.write(cues, srt_path, layout)
        produced.append(str(srt_path))

        mode = job.get("subtitle_mode", "srt")
        if mode in {"soft", "hard"}:
            render = _stage_reporter("render", index)
            work_srt = tmp / "subs.srt"          # 避開中文與空白路徑造成的濾鏡跳脫問題
            shutil.copy2(srt_path, work_srt)
            if mode == "soft":
                dst = out_dir / f"{stem}.subbed{path.suffix}"
                if dst.resolve() == path.resolve():
                    dst = out_dir / f"{stem}.subbed.mkv"
                render(0.02, "嵌入軟字幕（不重新編碼）")
                media.mux_soft_subtitles(
                    path, [(work_srt, lang_code, target_lang)], dst,
                    on_progress=lambda f: render(f, "嵌入軟字幕"),
                    duration=duration)
            else:
                font = media.ensure_fonts()
                dst = out_dir / f"{stem}.hardsub.mp4"
                render(0.02, f"燒錄字幕（字型 {font}）")
                media.burn_subtitles(
                    path, work_srt, dst, font,
                    font_size=int(job.get("font_size", 20)),
                    on_progress=lambda f: render(f, "燒錄字幕"),
                    duration=duration)
            produced.append(str(dst))
        else:
            _stage_reporter("render", index)(1.0, "完成")

    emit(type="file_done", index=index, outputs=produced)
    return produced


def main() -> int:
    job = json.loads(sys.stdin.read() or "{}")
    files = [Path(f) for f in job.get("files", [])]
    if not files:
        emit(type="fatal", error="沒有指定任何影片檔")
        return 2
    if job.get("output_dir") == "__project__":
        job["output_dir"] = str(OUTPUT)

    def setup_progress(fraction: float, message: str) -> None:
        # 本地引擎第一次要下載好幾 GB 的模型，沒有進度會讓人以為當掉了
        emit(type="setup", pct=max(0.0, min(1.0, fraction)), msg=message)

    try:
        translator = translate.build(job.get("engine", "google"), job,
                                     on_setup=setup_progress)
    except Exception as exc:
        emit(type="fatal", error=f"翻譯引擎初始化失敗：{exc}")
        return 2

    failures = 0
    spent: list[float] = []
    taken: set[str] = set()      # 這一輪已經用掉的輸出檔名
    for i, path in enumerate(files):
        started = time.monotonic()
        # 處理過幾個檔案之後，就能用實際平均推估整批還要多久
        if spent:
            average = sum(spent) / len(spent)
            emit(type="overall_eta", eta=average * (len(files) - i))
        try:
            if not path.exists():
                raise RuntimeError("找不到這個檔案")
            process_one(path, i, len(files), job, translator, taken)
            spent.append(time.monotonic() - started)
        except translate.FatalTranslationError as exc:
            # 金鑰或權限問題，剩下的檔案跑下去也只會一樣失敗
            emit(type="file_error", index=i, name=path.name, error=str(exc))
            emit(type="fatal", error=f"{exc}\n已中止後續檔案。")
            failures += len(files) - i
            break
        except Exception as exc:
            failures += 1
            emit(type="file_error", index=i, name=path.name, error=str(exc),
                 detail=traceback.format_exc()[-1200:])

    emit(type="all_done", failed=failures, total=len(files))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
