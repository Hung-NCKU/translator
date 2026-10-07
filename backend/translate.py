"""翻譯引擎：Claude API（主力）與 Google 免費翻譯（備援）。"""
from __future__ import annotations

import json
import os
import time
from typing import Callable

from .config import LANGUAGES
from .transcribe import Cue

class FatalTranslationError(RuntimeError):
    """金鑰、額度或權限問題：再多檔案也只會一樣失敗，直接中止整批。"""


def _is_unsupported_param(exc: Exception) -> bool:
    """判斷這個 400 是不是「伺服器不認得某個參數」造成的。"""
    text = str(exc).lower()
    return "fallbacks" in text or "beta" in text or "unexpected" in text


def _friendly_error(exc: Exception) -> str:
    """把 API 的英文錯誤翻成看得懂、而且知道下一步該做什麼的訊息。"""
    text = str(exc).lower()
    if "credit balance" in text or "billing" in text:
        return ("Anthropic 帳戶的 API 額度不足。\n\n"
                "請到 console.anthropic.com → Plans & Billing 儲值。\n"
                "注意：Claude Pro / Max 訂閱不包含 API 額度，兩者是分開計費的，"
                "所以有訂閱也還是要另外買 API credits。\n\n"
                "不想付費的話，可以把翻譯引擎改成「Google 免費翻譯」。")
    if "authentication" in text or "invalid x-api-key" in text:
        return ("API Key 不正確。請確認是從 console.anthropic.com 複製的完整金鑰"
                "（開頭是 sk-ant-），中間沒有多餘空白。")
    if "permission" in text or "not allowed" in text:
        return "這把 API Key 沒有使用這個模型的權限，請換一把金鑰或改選其他模型。"
    if "model" in text and "not found" in text:
        return f"找不到這個模型（{exc}）。請在設定裡改選其他 Claude 模型。"
    return str(exc)


CLAUDE_MODELS = {
    "Claude Opus 5（品質最好 / 推薦）": "claude-opus-5",
    "Claude Sonnet 5（快且便宜）": "claude-sonnet-5",
    "Claude Haiku 4.5（最便宜）": "claude-haiku-4-5",
}

_SYSTEM = """你是專業影視字幕譯者，專精於將對白翻譯成{target}。

翻譯規則：
1. 一句原文對應一句譯文，絕不合併、拆分或增刪句子。
2. 字幕要口語、自然、簡潔，符合觀眾閱讀速度，不要書面語或冗長解釋。
3. 保留說話者的語氣與情緒（敬語、粗話、猶豫、開玩笑都要譯出對應感覺）。
4. 人名、地名、作品名前後一致；沒有通用譯名時保留原文。
5. 語音辨識可能有錯字或斷句不完整，請依上下文合理推斷，不要照錯字直譯。
6. 只輸出譯文本身，不要加註解、引號或說明。
7. 若某句是無意義的語助詞或雜音，輸出最貼近的簡短對應詞即可，不要留空。"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


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


class ClaudeTranslator:
    """用 Claude 翻譯，整批送出並帶上下文，字幕品質明顯優於逐句機翻。"""

    def __init__(self, api_key: str, model: str = "claude-opus-5",
                 effort: str = "medium", batch_size: int = 40,
                 glossary: str = ""):
        import anthropic
        if not api_key:
            raise RuntimeError("沒有填 Anthropic API key")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.anthropic = anthropic
        self.model = model
        self.effort = effort
        self.batch_size = max(5, batch_size)
        self.glossary = glossary.strip()
        self.use_fallbacks = True

    def _call(self, system: str, user: str) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA},
                           "effort": self.effort},
        )
        if self.use_fallbacks:
            try:
                # 影片內容可能觸發安全分類器，開啟伺服器端 fallback 才不會整批失敗
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default", **kwargs)
                return next(b.text for b in resp.content if b.type == "text")
            except TypeError:
                # SDK 版本不吃這個參數，之後都走標準呼叫
                self.use_fallbacks = False
            except self.anthropic.BadRequestError as exc:
                # 只有「這個參數不能用」才降級重打；餘額不足之類的 400 直接往上丟，
                # 否則會被誤判成參數問題，白白多打一次必然失敗的請求
                if not _is_unsupported_param(exc):
                    raise
                self.use_fallbacks = False
        resp = self.client.messages.create(**kwargs)
        return next(b.text for b in resp.content if b.type == "text")

    def translate(self, cues: list[Cue], target_lang: str, source_name: str = "",
                  on_progress: Callable[[float, str], None] | None = None) -> None:
        _code, target_name = _target_names(target_lang)
        to_taiwan = target_lang == "繁體中文"
        system = _SYSTEM.format(target=target_name)
        if self.glossary:
            system += f"\n\n【本片專用指示與術語表，優先遵守】\n{self.glossary}"
        if source_name:
            system += f"\n\n原文語言：{source_name}。"

        done = 0
        for start in range(0, len(cues), self.batch_size):
            batch = cues[start:start + self.batch_size]
            # 帶上前 5 句原文當作上下文，讓代名詞與語氣能接續
            context = [c.text for c in cues[max(0, start - 5):start]]
            payload = {
                "previous_lines_for_context_only": context,
                "lines_to_translate": [
                    {"id": i, "text": c.text} for i, c in enumerate(batch)
                ],
            }
            user = ("請翻譯 lines_to_translate 裡的每一句，"
                    "輸出的 translations 陣列必須包含每一個 id，順序與數量完全一致。\n\n"
                    + json.dumps(payload, ensure_ascii=False, indent=1))

            text = self._retry(system, user)
            mapping = {}
            try:
                for item in json.loads(text).get("translations", []):
                    mapping[int(item["id"])] = str(item["text"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                mapping = {}
            for i, cue in enumerate(batch):
                text = mapping.get(i, "") or cue.text
                cue.translation = localize_taiwan(text) if to_taiwan else text

            done += len(batch)
            if on_progress:
                on_progress(done / max(1, len(cues)),
                            f"翻譯中：{done}/{len(cues)} 句")

    def _retry(self, system: str, user: str, attempts: int = 3) -> str:
        # 這幾種都是請求本身有問題（金鑰、額度、權限、模型），重試再多次也一樣
        fatal = (self.anthropic.AuthenticationError,
                 self.anthropic.PermissionDeniedError,
                 self.anthropic.NotFoundError,
                 self.anthropic.BadRequestError)
        last: Exception | None = None
        for n in range(attempts):
            try:
                return self._call(system, user)
            except fatal as exc:
                raise FatalTranslationError(
                    f"Claude 翻譯失敗：{_friendly_error(exc)}") from exc
            except Exception as exc:  # 速率限制或暫時性錯誤就退避重試
                last = exc
                if n < attempts - 1:
                    time.sleep(2 ** n * 3)
        raise RuntimeError(f"Claude 翻譯失敗：{last}")


class GoogleTranslator:
    """免費備援，不需金鑰，但語氣與斷句品質普通。"""

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
                "請稍後再試，或改用 Claude API 翻譯。")


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
    if engine == "claude":
        key = settings.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        return ClaudeTranslator(
            api_key=key,
            model=settings.get("claude_model", "claude-opus-5"),
            effort=settings.get("effort", "medium"),
            batch_size=int(settings.get("batch_size", 40)),
            glossary=settings.get("glossary", ""),
        )
    return GoogleTranslator(source=settings.get("source_code", "auto"))
