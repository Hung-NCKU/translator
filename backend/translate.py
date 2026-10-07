"""翻譯引擎的挑選，以及譯文的台灣在地化處理。

實際翻譯由 ct2_translate 的本地開源模型負責，這裡另外提供 Google 免費翻譯當備援。
"""
from __future__ import annotations

import time
from typing import Callable

from .config import LANGUAGES
from .transcribe import Cue


class FatalTranslationError(RuntimeError):
    """設定或環境層級的問題：換下一個檔案也只會一樣失敗，直接中止整批。"""


def _target_names(target_lang: str) -> tuple[str, str]:
    """回傳 (Google 語言代碼, 給模型看的語言名稱)。"""
    google_code, model_name, _whisper = LANGUAGES[target_lang]
    return google_code, model_name


def target_display_name(target_lang: str) -> str:
    """給語言模型看的目標語言名稱，例如「繁體中文」→ Traditional Chinese。"""
    return _target_names(target_lang)[1]


# 模型（尤其是小模型）常不照提示用台灣詞彙，翻完再掃一次比較可靠。
# 只收「台灣幾乎不會這樣講、而且沒有別的意思」的詞；
# 像「質量」（物理學的 mass）、「數據」、「用戶」台灣也在用，就不要動。
_CN_TO_TW = {
    "早上好": "早安", "晚上好": "晚安", "視頻": "影片", "網絡": "網路",
    "軟件": "軟體", "硬件": "硬體", "鼠標": "滑鼠", "屏幕": "螢幕",
    "服務器": "伺服器", "激光": "雷射", "出租車": "計程車",
    "互聯網": "網際網路", "默認": "預設", "內存": "記憶體",
    "打印": "列印", "回車": "Enter", "菜單": "選單", "博客": "部落格",
}


def localize_taiwan(text: str) -> str:
    """把譯文整理成台灣讀者習慣的樣子。

    先用 OpenCC 處理字形與大部分詞彙（簡體→台灣正體，网络→網路），
    再補上它不管的口語講法（早上好→早安）。
    """
    from .transcribe import to_traditional

    text = to_traditional(text)
    for mainland, taiwan in _CN_TO_TW.items():
        if mainland in text:
            text = text.replace(mainland, taiwan)
    return text


class GoogleTranslator:
    """免費備援，不需金鑰，但語氣與斷句品質普通，而且很容易被限流。"""

    def __init__(self, source: str = "auto"):
        self.source = source

    def translate(self, cues: list[Cue], target_lang: str, source_name: str = "",
                  on_progress: Callable[[float, str], None] | None = None) -> None:
        from deep_translator import GoogleTranslator as _G
        code, _name = _target_names(target_lang)
        to_taiwan = target_lang == "繁體中文"
        engine = _G(source=self.source or "auto", target=code)

        failed = 0
        for i, cue in enumerate(cues, 1):
            text = ""
            for attempt in range(3):
                try:
                    text = (engine.translate(cue.text) or "").strip()
                    break
                except Exception:
                    time.sleep(1.5 * (attempt + 1))   # 免費端點限流，退避後重試
            if text:
                cue.translation = localize_taiwan(text) if to_taiwan else text
            else:
                failed += 1
                cue.translation = cue.text
            time.sleep(0.25)                          # 主動節流，別把額度用爆
            if on_progress and (i % 5 == 0 or i == len(cues)):
                on_progress(i / max(1, len(cues)),
                            f"翻譯中：{i}/{len(cues)} 句"
                            + (f"（{failed} 句失敗）" if failed else ""))

        # 整批翻不動時直接報錯，不要默默交出一份原文字幕讓人以為翻好了
        if failed > len(cues) // 2:
            raise RuntimeError(
                f"Google 免費翻譯有 {failed}/{len(cues)} 句失敗（通常是被限流）。"
                "請稍後再試，或改用本地開源模型翻譯。")


def build(engine: str, settings: dict,
          on_setup: Callable[[float, str], None] | None = None) -> object | None:
    """依設定建立翻譯器；engine 為 none 時代表只出原文字幕。"""
    if engine == "none":
        return None
    if engine == "local":
        from .ct2_translate import build_local
        return build_local(
            repo=settings.get("local_model", "ctranslate2-4you/Qwen3-4B-ct2-AWQ"),
            glossary=settings.get("glossary", ""),
            on_setup=on_setup,
        )
    return GoogleTranslator(source=settings.get("source_code", "auto"))
